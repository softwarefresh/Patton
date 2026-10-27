"""Rebuild full cleaned patent text for the cooperation_v2 evidence IDs.

The canonical evidence.jsonl.gz stores text[:1600] (see build_patent_partner_data.py
--text-chars default 1600) but also records full_cleaned_text_characters.  This
script recovers the full cleaned text from g_domain.db, verifies the raw-text SHA
and the cleaned length, and writes a separate cache.  Canonical data and labels
are read-only; no validation/test labels are opened.
"""
import argparse
import gzip
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path


def clean_patent_text(text):
    # Keep exactly the canonical cleaning rule from build_patent_partner_data.py.
    text = (text or "").replace("草莓科研服务网", "")
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def source_signature(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


def run(args):
    data_dir = Path(args.data_dir).resolve()
    evidence_path = data_dir / "evidence.jsonl.gz"
    db_path = Path(args.g_domain_db).resolve()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else output.parent / (output.name + ".manifest.json")
    if not evidence_path.is_file() or not db_path.is_file():
        raise ValueError("Missing canonical evidence or g_domain.db")
    if output.exists() or manifest_path.exists():
        raise ValueError("Refusing existing output/manifest; choose fresh paths")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive or omitted")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    seen = set()
    counts = {
        "patents": 0,
        "full_text_longer_than_stored": 0,
        "recovered_characters": 0,
        "max_full_characters": 0,
        "max_stored_characters": 0,
    }
    connection = sqlite3.connect("file:{}?mode=ro".format(db_path), uri=True)
    cursor = connection.cursor()
    try:
        with gzip.open(evidence_path, "rt", encoding="utf-8") as source, \
                gzip.open(output, "wt", encoding="utf-8", newline="\n") as sink:
            for line in source:
                row = json.loads(line)
                application = row.get("application_number")
                raw_sha = row.get("raw_text_sha256")
                stored = row.get("text") or ""
                full_characters = row.get("full_cleaned_text_characters")
                if not isinstance(application, str) or not application or application in seen:
                    raise ValueError("Invalid or duplicate evidence application")
                seen.add(application)
                record = cursor.execute(
                    "SELECT q_text FROM patents_g WHERE 申请号=?", (application,)
                ).fetchone()
                if record is None:
                    raise ValueError("Missing g_domain record: " + application)
                raw_text = record[0] or ""
                if sha256_text(raw_text) != raw_sha:
                    raise ValueError("Raw text SHA mismatch: " + application)
                cleaned = clean_patent_text(raw_text)
                if len(cleaned) != full_characters:
                    raise ValueError("Full cleaned length mismatch: " + application)
                if cleaned[:len(stored)] != stored:
                    raise ValueError("Stored text is not a prefix of cleaned text: " + application)
                recovered = max(0, len(cleaned) - len(stored))
                counts["patents"] += 1
                counts["max_full_characters"] = max(counts["max_full_characters"], len(cleaned))
                counts["max_stored_characters"] = max(counts["max_stored_characters"], len(stored))
                if recovered:
                    counts["full_text_longer_than_stored"] += 1
                    counts["recovered_characters"] += recovered
                sink.write(json.dumps({
                    "application_number": application,
                    "raw_text_sha256": raw_sha,
                    "text": cleaned,
                    "full_cleaned_text_characters": len(cleaned),
                    "stored_text_characters": len(stored),
                    "recovered_characters": recovered,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                if args.limit is not None and counts["patents"] >= args.limit:
                    break
                if counts["patents"] % 50000 == 0:
                    print("recovered {:,} patents".format(counts["patents"]), flush=True)
    finally:
        connection.close()
    if counts["patents"] == 0:
        raise ValueError("No patents recovered")
    manifest = {
        "format": "partner_full_cleaned_text_v1",
        "scope": "full" if args.limit is None else "sample_limit_{}".format(args.limit),
        "data_dir": str(data_dir),
        "g_domain_db": str(db_path),
        "source_fingerprints": {
            "evidence.jsonl.gz": fingerprint(evidence_path),
            "g_domain.db": source_signature(db_path),
        },
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": fingerprint(output)["sha256"],
        },
        "counts": counts,
        "canonical_sources_modified": False,
        "labels_accessed": False,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/patent/cooperation_v2"))
    parser.add_argument("--g-domain-db", type=Path, default=Path("data_pipeline/g_domain.db"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/patent/cooperation_v2/full_text/evidence.full.jsonl.gz"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, help="Optional smoke limit; omit for full recovery")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
