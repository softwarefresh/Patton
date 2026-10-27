"""Audit natural token lengths of the recovered full patent text.

Use the new encoder's own tokenizer, with truncation disabled, to choose a max
length from the actual distribution instead of setting 2048 blindly.  This is a
read-only text audit: no embeddings, labels, training, or canonical writes.
"""
import argparse
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path


BUDGETS = (256, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096, 8192, 16384, 32768)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def quantiles(values):
    values = sorted(values)
    return {
        name: values[round((len(values) - 1) * fraction)]
        for name, fraction in (
            ("min", 0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75),
            ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("max", 1.0),
        )
    }


def select_rows(path, sample_size):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(row)
    if sample_size and sample_size < len(rows):
        rows = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                ("full-text-token-audit-v1:" + row["application_number"]).encode()
            ).digest(),
        )[:sample_size]
    return rows


def run(args):
    from tokenizers import Tokenizer

    cache = Path(args.text_cache).resolve()
    tokenizer_path = Path(args.tokenizer_json).resolve()
    output = Path(args.output).resolve()
    if not cache.is_file() or not tokenizer_path.is_file():
        raise ValueError("Missing text cache or tokenizer.json")
    if output.exists():
        raise ValueError("Refusing existing audit output")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be 0 (all) or positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    started = time.monotonic()
    rows = select_rows(cache, args.sample_size)
    if not rows:
        raise ValueError("No text rows selected")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lengths, char_lengths, recovered = [], [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        encoded = tokenizer.encode_batch([row["text"] for row in batch], add_special_tokens=True)
        lengths.extend(len(result.ids) for result in encoded)
        char_lengths.extend(len(row["text"]) for row in batch)
        recovered.extend(row.get("recovered_characters", 0) for row in batch)
        del encoded
    n = len(lengths)
    coverage = {
        str(budget): {
            "at_or_below": sum(1 for value in lengths if value <= budget) / n,
            "above": sum(1 for value in lengths if value > budget) / n,
        }
        for budget in BUDGETS
    }
    report = {
        "format": "partner_full_text_token_audit_v1",
        "scope": "full" if not args.sample_size or args.sample_size >= n else "hash_sample_{}".format(n),
        "source_fingerprints": {
            "text_cache": fingerprint(cache),
            "tokenizer_json": fingerprint(tokenizer_path),
        },
        "counts": {
            "patents": n,
            "recovered_patents": sum(1 for value in recovered if value > 0),
            "recovered_characters": sum(recovered),
        },
        "characters": {"quantiles": quantiles(char_lengths)},
        "tokens_including_special": {
            "quantiles": quantiles(lengths),
            "mean": statistics.mean(lengths),
            "coverage": coverage,
        },
        "tokens_per_character": {
            "median": statistics.median(
                value / max(1, chars) for value, chars in zip(lengths, char_lengths)
            ),
        },
        "interpretation_limits": [
            "Token lengths are tokenizer-specific; this audit uses the supplied tokenizer.json.",
            "Coverage is token count, not semantic completeness.",
            "Choose max_length from the coverage table; a larger cap adds no information for texts already below it.",
            "No labels, embeddings, training, or canonical files were touched.",
        ],
        "canonical_sources_modified": False,
        "labels_accessed": False,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "patents": n,
        "char_quantiles": report["characters"]["quantiles"],
        "token_quantiles": report["tokens_including_special"]["quantiles"],
        "coverage": coverage,
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-cache", type=Path,
        default=Path("data/patent/cooperation_v2/full_text/evidence.full.jsonl.gz"),
    )
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=0, help="0 = all patents")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
