"""Transfer checks cover actual round trips and refusal of unsafe/corrupt assets."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
import package_partner_data as transfer


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class PartnerTransferTest(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="transfer-test-", dir=ROOT / "tmp")
        self.root = Path(self.temporary.name)
        self.data = self.root / "source"
        self.data.mkdir()
        for name in transfer.DATA_FILES:
            path = self.data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("历史证据 " + name + "\n").encode("utf-8"))
        for name in ("evidence.jsonl.gz", "evidence.tokens.jsonl.gz"):
            (self.data / name).write_bytes(gzip.compress((name + "正文").encode("utf-8"), mtime=0))
        artifacts = {name: {"bytes": (self.data / name).stat().st_size,
                            "sha256": transfer.sha256_file(self.data / name)}
                     for name in transfer.DATA_FILES if name.endswith(".jsonl") or name == "evidence.jsonl.gz"}
        self.dataset = {"protocol": "patent_conditioned_observed_partner_v2.0", "unique_input_patents": 1,
                        "artifacts": artifacts, "split_reports": {"train": {"queries": 1}}}
        dump(self.data / "dataset_manifest.json", self.dataset)
        self.tokenizer = {"includes_cls_sep": True, "unique_patents": 1,
                          "source_sha256": transfer.sha256_file(self.data / "evidence.jsonl.gz"),
                          "output_sha256": transfer.sha256_file(self.data / "evidence.tokens.jsonl.gz"),
                          "output_bytes": (self.data / "evidence.tokens.jsonl.gz").stat().st_size}
        dump(self.data / "tokenization_manifest.json", self.tokenizer)
        dump(self.data / "data_audit.json", {"status": "passed", "tokenization": {"checked": True}})

    def tearDown(self):
        self.temporary.cleanup()

    def bundle(self, name="bundle", model=None):
        output = self.root / name
        transfer.export_bundle(self.data, output, model, part_size=300)
        return output

    def test_round_trip_deterministic_bounded_parts_and_utf8(self):
        bundle = self.bundle()
        second = self.bundle("second")
        first_manifest = transfer.verify_bundle(bundle)
        self.assertEqual(first_manifest, transfer.verify_bundle(second))
        self.assertGreater(len(first_manifest["parts"]), 1)
        self.assertTrue(all(part["bytes"] <= 300 for part in first_manifest["parts"]))
        repo = self.root / "checkout"
        repo.mkdir()
        (repo / "code.py").write_text("unchanged", encoding="utf-8")
        receipt = transfer.install_bundle(bundle, repo)
        self.assertEqual(receipt["files_verified"], len(transfer.DATA_FILES))
        for name in transfer.DATA_FILES:
            self.assertEqual((self.data / name).read_bytes(), (repo / transfer.DATA_PREFIX / name).read_bytes())
        self.assertEqual((repo / "code.py").read_text(), "unchanged")
        self.assertFalse(list(repo.glob(".partner-stage-*")))

    def test_part_corruption_refused_before_installation(self):
        bundle = self.bundle()
        manifest = transfer.read_manifest(bundle)
        path = bundle / manifest["parts"][0]["name"]
        data = bytearray(path.read_bytes())
        data[15] ^= 1
        path.write_bytes(data)
        repo = self.root / "checkout"
        repo.mkdir()
        with self.assertRaisesRegex(ValueError, "Part fingerprint"):
            transfer.install_bundle(bundle, repo)
        self.assertEqual(list(repo.iterdir()), [])

    def test_archive_member_tamper_and_missing_member_rejected_even_with_rehashed_parts(self):
        for mutation in ("changed", "missing", "duplicate", "symlink", "escape"):
            with self.subTest(mutation=mutation):
                bundle = self.bundle("bundle-" + mutation)
                manifest = transfer.read_manifest(bundle)
                archive_bytes = b"".join((bundle / part["name"]).read_bytes() for part in manifest["parts"])
                with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as old:
                    content = [(member.name, old.extractfile(member).read()) for member in old]
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode="w:gz") as new:
                    for index, (name, data) in enumerate(content):
                        if mutation == "missing" and index == 0:
                            continue
                        if mutation == "changed" and index == 0:
                            data = bytes([data[0] ^ 1]) + data[1:]
                        member = tarfile.TarInfo("../outside" if mutation == "escape" and index == 0 else name)
                        member.size = len(data)
                        if mutation == "symlink" and index == 0:
                            member.type, member.linkname, member.size = tarfile.SYMTYPE, "/tmp/outside", 0
                        new.addfile(member, io.BytesIO(data))
                        if mutation == "duplicate" and index == 0:
                            new.addfile(member, io.BytesIO(data))
                self.rewrite_archive(bundle, manifest, stream.getvalue())
                with self.assertRaises(ValueError):
                    transfer.verify_bundle(bundle)

    def rewrite_archive(self, bundle, manifest, data):
        for part in manifest["parts"]:
            (bundle / part["name"]).unlink()
        parts = []
        for index, offset in enumerate(range(0, len(data), 300), 1):
            block = data[offset:offset + 300]
            name = transfer.PART_PREFIX + format(index, "03d")
            (bundle / name).write_bytes(block)
            parts.append({"name": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
        manifest.update(parts=parts, archive_bytes=len(data), archive_sha256=hashlib.sha256(data).hexdigest())
        dump(bundle / transfer.MANIFEST, manifest)

    def test_manifest_path_duplicate_and_part_path_refused(self):
        mutations = (
            lambda manifest: manifest["files"][0].update(name="../escape"),
            lambda manifest: manifest["files"].append(dict(manifest["files"][0])),
            lambda manifest: manifest["parts"][0].update(name="../outside"),
            lambda manifest: manifest["files"][0].update(name="/absolute"),
            lambda manifest: manifest["files"][0].update(name="C:/absolute"),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                bundle = self.bundle("bad-manifest-" + str(index))
                manifest = transfer.read_manifest(bundle)
                mutation(manifest)
                dump(bundle / transfer.MANIFEST, manifest)
                with self.assertRaises(ValueError):
                    transfer.verify_bundle(bundle)

    def test_source_change_during_archive_refused_and_staging_cleaned(self):
        original = transfer.HashingReader.read
        triggered = [False]

        def mutate(reader, size=-1):
            if not triggered[0]:
                triggered[0] = True
                target = self.data / "documents.train.jsonl"
                target.write_bytes(target.read_bytes() + b"changed")
            return original(reader, size)

        with patch.object(transfer.HashingReader, "read", mutate):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                self.bundle()
        self.assertFalse((self.root / "bundle").exists())
        self.assertFalse(list(self.root.glob(".partner-stage-*")))

    def test_source_canonical_tamper_and_existing_destination_refused(self):
        bundle = self.bundle()
        repo = self.root / "checkout"
        existing = repo / transfer.DATA_PREFIX
        existing.mkdir(parents=True)
        sentinel = existing / "mine.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "existing asset root"):
            transfer.install_bundle(bundle, repo)
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(ValueError, "existing output"):
            transfer.export_bundle(self.data, bundle)
        (self.data / "documents.train.jsonl").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Canonical artifact differs"):
            self.bundle("tampered")

    def test_model_requires_matching_tokenizer_and_installs_under_fixed_root(self):
        model = self.root / "base-model"
        model.mkdir()
        for name in transfer.MODEL_FILES + ("pytorch_model.bin",):
            (model / name).write_bytes((name + " local weights").encode())
        self.tokenizer["tokenizer_json_sha256"] = transfer.sha256_file(model / "tokenizer.json")
        self.tokenizer["vocab_sha256"] = transfer.sha256_file(model / "vocab.txt")
        dump(self.data / "tokenization_manifest.json", self.tokenizer)
        bundle = self.bundle(model=model)
        repo = self.root / "checkout"
        repo.mkdir()
        transfer.install_bundle(bundle, repo)
        self.assertEqual((repo / transfer.MODEL_PREFIX / "pytorch_model.bin").read_bytes(),
                         (model / "pytorch_model.bin").read_bytes())
        (model / "vocab.txt").write_bytes(b"different vocab")
        with self.assertRaisesRegex(ValueError, "tokenizer differs"):
            self.bundle("bad-model", model)

    def test_install_move_failure_rolls_back_only_new_roots(self):
        model = self.root / "base-model"
        model.mkdir()
        for name in transfer.MODEL_FILES + ("pytorch_model.bin",):
            (model / name).write_bytes(name.encode())
        self.tokenizer.update(tokenizer_json_sha256=transfer.sha256_file(model / "tokenizer.json"),
                              vocab_sha256=transfer.sha256_file(model / "vocab.txt"))
        dump(self.data / "tokenization_manifest.json", self.tokenizer)
        bundle = self.bundle(model=model)
        repo = self.root / "checkout"
        repo.mkdir()
        (repo / "mine.txt").write_text("keep")
        rename = Path.rename

        def fail_model(path, target):
            if str(target).replace("\\", "/").endswith(transfer.MODEL_PREFIX):
                raise OSError("simulated target move failure")
            return rename(path, target)

        with patch.object(Path, "rename", fail_model):
            with self.assertRaisesRegex(OSError, "simulated"):
                transfer.install_bundle(bundle, repo)
        self.assertFalse((repo / transfer.DATA_PREFIX).exists())
        self.assertFalse((repo / transfer.MODEL_PREFIX).exists())
        self.assertEqual((repo / "mine.txt").read_text(), "keep")
        self.assertFalse(list(repo.glob(".partner-stage-*")))


if __name__ == "__main__":
    unittest.main()
