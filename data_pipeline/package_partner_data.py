"""Stream a verified v2 dataset (and optional local base model) into Release assets.

No network access, original databases, derived caches, or old trained checkpoints
are involved. Each compressed part is at most 512 MiB. ``verify`` checks every
part and every archived file; ``install`` stages verified files before moving
the two fixed asset roots into an existing checkout. Existing asset roots are
never overwritten. The transfer manifest is an integrity receipt, not encryption.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OWNER = "patton_partner_release_v1"
MANIFEST = "partner-transfer.json"
DATA_PREFIX = "data/patent/cooperation_v2"
MODEL_PREFIX = "ckpt/chinese-roberta-wwm-ext"
PART_PREFIX = "partner-assets.tar.gz.part"
MAX_PART = 512 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}")
DATA_FILES = (
    "dataset_manifest.json", "tokenization_manifest.json", "data_audit.json",
    "dataset_report.md", "evidence.jsonl.gz", "evidence.tokens.jsonl.gz",
) + tuple(name for split in ("train", "val", "test") for name in (
    "documents." + split + ".jsonl", "histories." + split + ".jsonl",
    "queries." + split + ".jsonl", split + ".pairs.jsonl",
)) + ("baseline/metrics.val.json", "baseline/metrics.test.json")
MODEL_FILES = ("config.json", "tokenizer.json", "vocab.txt", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            "Invalid archive path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(part not in ("", ".", "..") for part in name.split("/"))
            and path.as_posix() == name, "Unsafe archive path: " + name)
    return name


def regular_under(root, name):
    safe_name(name)
    path = root / name
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        require(not current.is_symlink(), "Refusing symbolic link: " + str(current))
    require(path.is_file() and path.resolve().is_relative_to(root), "Missing regular source: " + str(path))
    return path


def snapshot(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_ino


def remove_staging(path, parent):
    # Delete only this tool's own temporary directory, never a computed data root.
    path, parent = Path(path), Path(parent).resolve()
    require(not path.is_symlink() and path.resolve().parent == parent and
            path.name.startswith(".partner-stage-"), "Unsafe staging cleanup")
    shutil.rmtree(str(path))


def collect_sources(data_dir, model_dir=None):
    directory = Path(data_dir).resolve()
    sources = {DATA_PREFIX + "/" + name: regular_under(directory, name) for name in DATA_FILES}
    dataset = load_json(directory / "dataset_manifest.json")
    tokens = load_json(directory / "tokenization_manifest.json")
    audit = load_json(directory / "data_audit.json")
    require(dataset.get("protocol") == "patent_conditioned_observed_partner_v2.0", "Unknown dataset protocol")
    require(audit.get("status") == "passed" and audit.get("tokenization", {}).get("checked") is True,
            "Completed dataset/token audit is required")
    require(tokens.get("includes_cls_sep") is True and tokens.get("unique_patents") == dataset.get("unique_input_patents"),
            "Tokenization/dataset counts differ")
    canonical_files = {name for name in DATA_FILES if name.endswith(".jsonl") or name == "evidence.jsonl.gz"}
    require(isinstance(dataset.get("artifacts"), dict) and canonical_files <= set(dataset["artifacts"]),
            "Dataset manifest is missing canonical artifact fingerprints")
    if model_dir is not None:
        model = Path(model_dir).resolve()
        weights = [name for name in ("pytorch_model.bin", "model.safetensors") if (model / name).is_file()]
        require(len(weights) == 1, "Expected exactly one complete base-model weight file")
        for name in MODEL_FILES + tuple(weights):
            sources[MODEL_PREFIX + "/" + name] = regular_under(model, name)
        for name, key in (("tokenizer.json", "tokenizer_json_sha256"), ("vocab.txt", "vocab_sha256")):
            require(sha256_file(model / name) == tokens.get(key), "Base-model tokenizer differs from encoded data: " + name)
    return sources, dataset, tokens


class SplitWriter:
    """Write a gzip stream directly to bounded parts, without a joined archive."""
    def __init__(self, directory, part_size):
        self.directory, self.part_size = directory, part_size
        self.parts, self.stream = [], None
        self.count, self.total = 0, 0
        self.digest, self.part_digest = hashlib.sha256(), None

    def writable(self):
        return True

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def _finish_part(self):
        if self.stream is not None:
            self.stream.close()
            self.parts.append({"name": self.name, "bytes": self.count,
                               "sha256": self.part_digest.hexdigest()})
            self.stream = None

    def write(self, data):
        original_size = len(data)
        view = memoryview(data)
        while view:
            if self.stream is None:
                self.name = PART_PREFIX + format(len(self.parts) + 1, "03d")
                self.stream = (self.directory / self.name).open("xb")
                self.count, self.part_digest = 0, hashlib.sha256()
            block = view[:self.part_size - self.count]
            self.stream.write(block)
            self.digest.update(block)
            self.part_digest.update(block)
            self.count += len(block)
            self.total += len(block)
            view = view[len(block):]
            if self.count == self.part_size:
                self._finish_part()
        return original_size

    def close(self):
        self._finish_part()


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.count += len(block)
        return block


def export_bundle(data_dir, output_dir, model_dir=None, part_size=MAX_PART):
    require(integer(part_size) and 0 < part_size <= MAX_PART, "Part size must be in 1..512 MiB")
    output = Path(output_dir).resolve()
    require(not output.exists(), "Refusing existing output directory: " + str(output))
    source_root = Path(data_dir).resolve()
    require(not output.is_relative_to(source_root), "Output must be outside the source data directory")
    sources, dataset, tokens = collect_sources(data_dir, model_dir)
    records, snapshots = [], {}
    for name, path in sorted(sources.items()):
        before = snapshot(path)
        digest = sha256_file(path)
        require(before == snapshot(path), "Source changed while hashing: " + name)
        records.append({"name": name, "bytes": before[0], "sha256": digest})
        snapshots[name] = before
        relative = name.removeprefix(DATA_PREFIX + "/")
        if name.startswith(DATA_PREFIX + "/") and relative in dataset.get("artifacts", {}):
            expected = dataset["artifacts"][relative]
            require(expected.get("sha256") == digest and expected.get("bytes") == before[0],
                    "Canonical artifact differs from dataset manifest: " + relative)
        if relative == "evidence.jsonl.gz":
            require(digest == tokens.get("source_sha256"), "Evidence fingerprint differs")
        if relative == "evidence.tokens.jsonl.gz":
            require(digest == tokens.get("output_sha256") and before[0] == tokens.get("output_bytes"),
                    "Encoded evidence fingerprint differs")
        for model_name, key in (("tokenizer.json", "tokenizer_json_sha256"), ("vocab.txt", "vocab_sha256")):
            if name == MODEL_PREFIX + "/" + model_name:
                require(digest == tokens.get(key), "Base-model tokenizer changed before archiving")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".partner-stage-", dir=str(output.parent)))
    writer = SplitWriter(staging, part_size)
    try:
        with gzip.GzipFile(filename="", mode="wb", fileobj=writer, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for record in records:
                    name, path = record["name"], sources[record["name"]]
                    require(snapshot(path) == snapshots[name], "Source changed before archiving: " + name)
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = record["bytes"], 0o644, 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as stream:
                        reader = HashingReader(stream)
                        archive.addfile(info, reader)
                        require(reader.count == record["bytes"] and reader.digest.hexdigest() == record["sha256"],
                                "Source changed during archiving: " + name)
        writer.close()
        for record in records:
            path = sources[record["name"]]
            require(snapshot(path) == snapshots[record["name"]] and sha256_file(path) == record["sha256"],
                    "Source changed during export: " + record["name"])
        manifest = {"format": OWNER, "archive_format": "tar+gzip", "part_size_bytes": part_size,
                    "archive_bytes": writer.total, "archive_sha256": writer.digest.hexdigest(),
                    "files_total_bytes": sum(record["bytes"] for record in records),
                    "model_included": model_dir is not None, "files": records, "parts": writer.parts,
                    "dataset_protocol": dataset["protocol"], "unique_input_patents": dataset["unique_input_patents"],
                    "split_reports": dataset["split_reports"], "canonical_sources_modified": False}
        (staging / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "LOCAL_PLAINTEXT_DO_NOT_PUBLISH.txt").write_text(
            "LOCAL PLAINTEXT — DO NOT PUBLISH\n"
            "This folder contains purchased patent data and base-model assets.\n"
            "Do not upload these plaintext parts or partner-transfer.json to a public repository or Release.\n"
            "Use a private destination, or encrypt this complete folder locally before public transfer.\n"
            "Keep the manifest inside the encrypted container. The transfer tool does not encrypt files.\n\n"
            "本目录是本地明文传输包，包含购买的数据；不要将分片及清单直接上传公开仓库或公开Release。\n"
            "使用私有目标，或先将整个目录加密再上传；清单也应放在加密包内。\n",
            encoding="utf-8")
        verify_bundle(staging)
        require(not output.exists(), "Output appeared during export")
        staging.rename(output)
        staging = None
        return manifest
    finally:
        writer.close()
        if staging is not None:
            remove_staging(staging, output.parent)


def read_manifest(directory):
    manifest = load_json(regular_under(directory, MANIFEST))
    require(manifest.get("format") == OWNER and manifest.get("archive_format") == "tar+gzip", "Unknown transfer format")
    require(type(manifest.get("model_included")) is bool, "Missing model_included flag")
    expected = {DATA_PREFIX + "/" + name for name in DATA_FILES}
    records = manifest.get("files")
    require(isinstance(records, list) and records, "Missing file list")
    names = [safe_name(record.get("name")) for record in records]
    require(len(names) == len(set(names)), "Duplicate file names in manifest")
    if manifest["model_included"]:
        expected |= {MODEL_PREFIX + "/" + name for name in MODEL_FILES}
        weights = {MODEL_PREFIX + "/pytorch_model.bin", MODEL_PREFIX + "/model.safetensors"} & set(names)
        require(len(weights) == 1, "Expected one base-model weight file")
        expected |= weights
    require(set(names) == expected, "Unexpected or missing asset paths")
    for record in records:
        require(integer(record.get("bytes")) and record["bytes"] >= 0 and
                isinstance(record.get("sha256"), str) and SHA.fullmatch(record["sha256"]), "Invalid file fingerprint")
    require(manifest.get("files_total_bytes") == sum(record["bytes"] for record in records), "Invalid total source bytes")
    part_size, parts = manifest.get("part_size_bytes"), manifest.get("parts")
    require(integer(part_size) and 0 < part_size <= MAX_PART and isinstance(parts, list) and parts, "Invalid part list/size")
    for index, part in enumerate(parts, 1):
        require(part.get("name") == PART_PREFIX + format(index, "03d"), "Invalid or unordered part name")
        require(integer(part.get("bytes")) and 0 < part["bytes"] <= part_size and
                (index == len(parts) or part["bytes"] == part_size), "Invalid part byte count")
        require(isinstance(part.get("sha256"), str) and SHA.fullmatch(part["sha256"]), "Invalid part fingerprint")
    require(manifest.get("archive_bytes") == sum(part["bytes"] for part in parts) and
            isinstance(manifest.get("archive_sha256"), str) and SHA.fullmatch(manifest["archive_sha256"]),
            "Invalid archive fingerprint")
    return manifest


class PartReader(io.RawIOBase):
    def __init__(self, directory, parts):
        self.paths = iter(directory / part["name"] for part in parts)
        self.stream = None

    def readable(self):
        return True

    def readinto(self, target):
        while True:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.stream = path.open("rb")
            count = self.stream.readinto(target)
            if count:
                return count
            self.stream.close()
            self.stream = None

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


def check_parts(directory, manifest):
    archive_digest = hashlib.sha256()
    for part in manifest["parts"]:
        path = regular_under(directory, part["name"])
        require(path.stat().st_size == part["bytes"], "Part size mismatch: " + part["name"])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                archive_digest.update(block)
        require(digest.hexdigest() == part["sha256"], "Part fingerprint mismatch: " + part["name"])
    require(archive_digest.hexdigest() == manifest["archive_sha256"], "Joined archive fingerprint mismatch")


def scan_archive(directory, manifest, destination=None):
    expected = {record["name"]: record for record in manifest["files"]}
    seen = set()
    with PartReader(directory, manifest["parts"]) as raw:
        with io.BufferedReader(raw) as buffered:
            with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
                for member in archive:
                    name = safe_name(member.name)
                    require(member.isfile() and name in expected and name not in seen,
                            "Unexpected, duplicate, or non-regular archive member: " + name)
                    record = expected[name]
                    require(member.size == record["bytes"], "Archive file size mismatch: " + name)
                    seen.add(name)
                    output = None
                    if destination is not None:
                        path = destination / name
                        require(path.resolve().is_relative_to(destination), "Archive path escaped staging")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        output = path.open("xb")
                    digest, count = hashlib.sha256(), 0
                    try:
                        with archive.extractfile(member) as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(block)
                                count += len(block)
                                if output is not None:
                                    output.write(block)
                    finally:
                        if output is not None:
                            output.close()
                    require(count == record["bytes"] and digest.hexdigest() == record["sha256"],
                            "Archived file fingerprint mismatch: " + name)
    require(seen == set(expected), "Missing archive files")


def verify_bundle(bundle_dir):
    directory = Path(bundle_dir).resolve()
    manifest = read_manifest(directory)
    check_parts(directory, manifest)
    scan_archive(directory, manifest)
    return manifest


def absent_asset_root(repo, name):
    target = repo / name
    current = repo
    for part in PurePosixPath(name).parts:
        current = current / part
        require(not current.is_symlink(), "Refusing symbolic-link installation path")
    require(target.resolve().is_relative_to(repo) and not target.exists(),
            "Refusing existing asset root: " + str(target))
    return target


def install_bundle(bundle_dir, repo_dir):
    directory, repo = Path(bundle_dir).resolve(), Path(repo_dir).resolve()
    require(repo.is_dir(), "Repository destination must already exist")
    manifest = read_manifest(directory)
    names = [DATA_PREFIX] + ([MODEL_PREFIX] if manifest["model_included"] else [])
    targets = {name: absent_asset_root(repo, name) for name in names}
    check_parts(directory, manifest)
    staging = Path(tempfile.mkdtemp(prefix=".partner-stage-", dir=str(repo)))
    moved = []
    try:
        scan_archive(directory, manifest, staging)
        receipt = {"format": OWNER, "status": "installed", "bundle_manifest_sha256": sha256_file(directory / MANIFEST),
                   "files_verified": len(manifest["files"]), "canonical_sources_modified": False}
        (staging / DATA_PREFIX / "transfer_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name in names:
            target = absent_asset_root(repo, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).rename(target)
            moved.append((name, target))
        return receipt
    except Exception:
        # Roll back only directories installed by this invocation; leave existing
        # checkout files and unrelated data untouched.
        for name, target in reversed(moved):
            target.rename(staging / name)
        raise
    finally:
        remove_staging(staging, repo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Create verified, bounded GitHub Release assets locally")
    export.add_argument("--data-dir", type=Path, default=ROOT / DATA_PREFIX)
    export.add_argument("--model-dir", type=Path)
    export.add_argument("--output-dir", type=Path, default=ROOT / "tmp/partner_release")
    export.add_argument("--part-mib", type=int, default=512)
    verify = commands.add_parser("verify", help="Read and check every compressed part and archived file")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    install = commands.add_parser("install", help="Verify and install assets into an existing checkout")
    install.add_argument("--bundle-dir", type=Path, required=True)
    install.add_argument("--repo-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "export":
        require(1 <= args.part_mib <= 512, "--part-mib must be 1..512")
        result = export_bundle(args.data_dir, args.output_dir, args.model_dir, args.part_mib * 1024 * 1024)
        print(json.dumps({"status": "exported_and_verified", "output_dir": str(args.output_dir.resolve()),
                          "file_count": len(result["files"]), "source_bytes": result["files_total_bytes"],
                          "archive_bytes": result["archive_bytes"], "parts": result["parts"],
                          "model_included": result["model_included"]}, ensure_ascii=False, indent=2))
    elif args.command == "verify":
        result = verify_bundle(args.bundle_dir)
        print(json.dumps({"status": "verified", "file_count": len(result["files"]),
                          "archive_bytes": result["archive_bytes"]}, indent=2))
    else:
        print(json.dumps(install_bundle(args.bundle_dir, args.repo_dir), indent=2))


if __name__ == "__main__":
    main()
