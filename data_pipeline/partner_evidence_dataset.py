"""Read canonical v2 evidence sets without loading the token corpus into RAM.

EvidenceDataset needs only the standard library. Returned token IDs already
include CLS/SEP and are never retokenized or concatenated. Each item is:
  {query_id, features: {source_doc_id, query, query_neighbors,
   query_neighbor_mask, candidate_doc_ids, candidate_evidence},
   targets: {positive_mask}}.
Each candidate_evidence row has 20 separate patent slots and an evidence_mask;
each patent has variable-length input_ids/attention_mask and dated IPC metadata.
Missing slots have empty token lists and None metadata. Positive masks are
targets; label_evidence, company names, and outcome dates are never features.
"""
import argparse
from array import array
from datetime import date
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import zlib


OWNER = "patton_partner_evidence_token_cache_v1"
SCHEMA = "2"
SHA256 = re.compile(r"[0-9a-f]{64}")
SPLIT_YEARS = {"train": 2022, "val": 2023, "test": 2024}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                item = json.loads(line)
                require(isinstance(item, dict), "Expected JSON object at {}:{}".format(path, number))
                yield item


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def id_list(values, name):
    require(isinstance(values, list) and all(integer(value) for value in values),
            name + " must be a list of integer document IDs")
    return list(dict.fromkeys(values))


def application_list(values, name):
    require(isinstance(values, list) and all(isinstance(value, str) and value for value in values),
            name + " must be a list of application IDs")
    require(len(values) == len(set(values)), name + " contains duplicate applications")
    return values


def valid_date(value, name):
    require(isinstance(value, str), name + " must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(name + " must be an ISO date") from exc
    require(parsed.isoformat() == value, name + " must use YYYY-MM-DD")
    return value


class EvidenceDataset:
    """ID-joined training records backed by a verified, read-only token cache.

    tokenizer may be a local tokenizer directory (or tokenizer.json path) to
    additionally verify its SHA against tokenization_manifest.json. It is not
    needed to encode text: the shared tokens are already encoded. Cache source
    fingerprint changes raise ValueError; existing foreign/stale caches are
    never replaced. The cache is shared across splits in data_dir/token_cache.
    """

    def __init__(self, data_dir, split="train", tokenizer=None, max_neighbors=5):
        require(split in ("train", "val", "test"), "split must be train, val, or test")
        require(integer(max_neighbors) and 0 <= max_neighbors <= 5, "max_neighbors must be 0..5")
        self.data_dir = Path(data_dir).resolve()
        self.split = split
        self.max_neighbors = max_neighbors
        self.evidence_slots = 20
        self.cache_path = self.data_dir / "token_cache/evidence.tokens.sqlite3"
        self._connection = None
        self._pid = None
        self._closed = False
        manifest_path = self.data_dir / "tokenization_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(self.manifest.get("includes_cls_sep") is True, "Token source must already include CLS/SEP")
        self.cls_id, self.sep_id = self.manifest.get("cls_token_id"), self.manifest.get("sep_token_id")
        self.max_length = self.manifest.get("max_length")
        require(integer(self.cls_id) and integer(self.sep_id) and self.cls_id != self.sep_id,
                "Manifest must declare distinct CLS/SEP token IDs")
        require(integer(self.max_length) and 2 <= self.max_length <= 256, "Invalid manifest max_length")
        require(integer(self.manifest.get("unique_patents")) and self.manifest["unique_patents"] > 0,
                "Manifest must declare the shared evidence count")
        self.fingerprints = {manifest_path.name: sha256_file(manifest_path)}
        for filename, key in (("evidence.tokens.jsonl.gz", "output_sha256"), ("evidence.jsonl.gz", "source_sha256")):
            expected = self.manifest.get(key)
            require(isinstance(expected, str) and SHA256.fullmatch(expected), "Missing valid " + key)
            actual = sha256_file(self.data_dir / filename)
            require(actual == expected, "Tokenization source fingerprint mismatch: " + filename)
            self.fingerprints[filename] = actual
        token_path = self.data_dir / "evidence.tokens.jsonl.gz"
        if "output_bytes" in self.manifest:
            require(token_path.stat().st_size == self.manifest["output_bytes"], "Token source byte count differs")
        if tokenizer is not None:
            tokenizer = Path(tokenizer).resolve()
            directory = tokenizer if tokenizer.is_dir() else tokenizer.parent
            for filename, key in (("tokenizer.json", "tokenizer_json_sha256"), ("vocab.txt", "vocab_sha256")):
                require(sha256_file(directory / filename) == self.manifest.get(key), "Tokenizer fingerprint mismatch: " + filename)
        self._signature = {
            "owner": OWNER, "schema": SCHEMA,
            "source_signature": json.dumps(self.fingerprints, sort_keys=True, separators=(",", ":")),
            "expected_records": str(self.manifest["unique_patents"]),
            "cls_id": str(self.cls_id), "sep_id": str(self.sep_id), "max_length": str(self.max_length),
            "encoding": "zlib1_little_endian_uint32",
        }
        self._load_metadata()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_hit = self.cache_path.exists()
        if self.cache_hit:
            self._validate_cache()
        else:
            self._build_cache()

    def _load_metadata(self):
        files = {
            "documents": self.data_dir / ("documents." + self.split + ".jsonl"),
            "histories": self.data_dir / ("histories." + self.split + ".jsonl"),
            "queries": self.data_dir / ("queries." + self.split + ".jsonl"),
            "pairs": self.data_dir / (self.split + ".pairs.jsonl"),
        }
        self.documents, self.histories, self.queries, self.pairs = {}, {}, {}, []
        for path in files.values():
            self.fingerprints[path.name] = sha256_file(path)
        dataset_manifest = self.data_dir / "dataset_manifest.json"
        if dataset_manifest.exists():
            self.fingerprints[dataset_manifest.name] = sha256_file(dataset_manifest)
            artifacts = json.loads(dataset_manifest.read_text(encoding="utf-8")).get("artifacts", {})
            for path in list(files.values()) + [self.data_dir / "evidence.jsonl.gz"]:
                signature = artifacts.get(path.name, {})
                require(signature.get("sha256") == self.fingerprints[path.name] and
                        signature.get("bytes") == path.stat().st_size,
                        "Canonical artifact fingerprint mismatch: " + path.name)
        for row in rows(files["documents"]):
            doc_id = row.get("doc_id")
            require(integer(doc_id) and doc_id not in self.documents, "Invalid or duplicate document ID")
            self.documents[doc_id] = {
                "applications": application_list(row.get("evidence_application_numbers"), "document evidence"),
                "company_code": row.get("company_code"),
                "excluded": set(id_list(row.get("excluded_doc_ids", []), "document exclusions")),
            }
        for row in rows(files["histories"]):
            doc_id = row.get("doc_id")
            require(integer(doc_id) and doc_id in self.documents and doc_id not in self.histories,
                    "History has missing or duplicate document ID")
            cutoff = valid_date(row.get("cutoff_exclusive"), "history cutoff")
            require(row.get("forecast_year") == SPLIT_YEARS[self.split] and
                    cutoff == "{}-01-01".format(SPLIT_YEARS[self.split]), "History cutoff/year/split mismatch")
            require(row.get("company_code") == self.documents[doc_id]["company_code"], "History/document identity mismatch")
            refs = row.get("patents")
            require(isinstance(refs, list) and 1 <= len(refs) <= self.evidence_slots, "History must retain 1..20 separate patents")
            apps = application_list([ref.get("application_number") for ref in refs], "history evidence")
            require(apps == self.documents[doc_id]["applications"], "Document/history application order mismatch")
            safe_refs = []
            for ref in refs:
                published = valid_date(ref.get("publication_date"), "history publication date")
                require(published < cutoff, "Future evidence reached input")
                require(integer(ref.get("event_id")), "Invalid historical event ID")
                require(isinstance(ref.get("raw_text_sha256"), str) and SHA256.fullmatch(ref["raw_text_sha256"]),
                        "Invalid historical text fingerprint")
                ipc = ref.get("ipc_main") or ""
                match = re.match(r"[A-HY][0-9]{2}[A-Z]", ipc.upper())
                require(ref.get("technical_group") == (match.group(0) if match else "UNKNOWN"), "IPC/technical-group mismatch")
                safe_refs.append({key: ref.get(key) for key in (
                    "application_number", "event_id", "publication_date", "ipc_main", "ipc_all", "technical_group", "raw_text_sha256")})
            require([(ref["publication_date"], ref["application_number"]) for ref in safe_refs] ==
                    sorted(((ref["publication_date"], ref["application_number"]) for ref in safe_refs), reverse=True),
                    "History patent order is not newest-first")
            self.histories[doc_id] = {"patents": safe_refs, "cutoff": cutoff, "year": row["forecast_year"]}
        require(self.documents and set(self.documents) == set(self.histories), "Candidate/history ID sets differ")
        candidate_ids = set(self.documents)
        for row in rows(files["queries"]):
            qid, source = row.get("query_id"), row.get("source_doc_id")
            require(isinstance(qid, str) and qid and qid not in self.queries, "Invalid or duplicate query ID")
            require(integer(source) and source in self.histories, "Query source has no matching history")
            history = self.histories[source]
            refs = history["patents"]
            require(application_list(row.get("input_application_numbers"), "query applications") ==
                    [ref["application_number"] for ref in refs], "Query/history application order mismatch")
            require(row.get("query_application_number") == refs[0]["application_number"], "Query center application mismatch")
            require(row.get("query_event_id") == refs[0]["event_id"] and
                    row.get("query_publication_date") == refs[0]["publication_date"], "Query center event/date mismatch")
            require(row.get("cutoff_exclusive") == history["cutoff"] and row.get("forecast_year") == history["year"],
                    "Query/history cutoff mismatch")
            positives = id_list(row.get("positive_doc_ids"), "query positives")
            excluded = set(id_list(row.get("excluded_doc_ids", []), "query exclusions")) | {source}
            require(excluded == self.documents[source]["excluded"] | {source}, "Query/document exclusions mismatch")
            require(positives and set(positives) <= candidate_ids and not set(positives) & excluded,
                    "Missing or excluded positive candidate")
            self.queries[qid] = {"source": source, "application": refs[0]["application_number"],
                                 "positives": positives, "excluded": excluded}
        seen = set()
        for row in rows(files["pairs"]):
            qid = row.get("query_id")
            require(qid in self.queries and qid not in seen, "Pair has missing or duplicate query ID")
            seen.add(qid)
            query = self.queries[qid]
            require(row.get("source_doc_id") == query["source"] and
                    row.get("query_application_number") == query["application"], "Pair/query source or application mismatch")
            positives = id_list(row.get("positive_doc_ids"), "pair positives")
            require(set(positives) == set(query["positives"]), "Pair loses or changes query positives")
            negatives = id_list(row.get("negative_doc_ids"), "pair negatives")
            hard = id_list(row.get("hard_negative_doc_ids"), "hard negatives")
            random_ids = id_list(row.get("random_negative_doc_ids"), "random negatives")
            require(set(negatives) == set(hard + random_ids), "Hard/random and combined negatives differ")
            require(not set(negatives) & (set(positives) | query["excluded"]), "Negative is positive or excluded")
            ordered = list(dict.fromkeys(positives + hard + random_ids))
            require(set(ordered) <= candidate_ids, "Pair candidate has no document/history")
            # Candidate position must not reveal the positive-first pair schema.
            # This fixed order uses only IDs and is independent of target masks.
            ordered.sort(key=lambda doc_id: (hashlib.sha256(
                (qid + "\0" + str(doc_id)).encode("utf-8")).digest(), doc_id))
            self.pairs.append({"query_id": qid, "candidate_ids": ordered, "positive_ids": set(positives)})
        require(seen and seen == set(self.queries), "Pair/query ID sets differ or are empty")

    def _validate_cache(self):
        connection = sqlite3.connect(self.cache_path.as_uri() + "?mode=ro", uri=True)
        try:
            try:
                metadata = dict(connection.execute("SELECT key,value FROM cache_meta"))
            except sqlite3.DatabaseError as exc:
                raise ValueError("Refusing unrecognized token cache: " + str(self.cache_path)) from exc
            require(metadata.get("owner") == OWNER, "Refusing foreign token cache")
            require(all(metadata.get(key) == value for key, value in self._signature.items()),
                    "Token cache fingerprint mismatch; use a fresh canonical data directory/cache")
            try:
                count = connection.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
                require(count == self.manifest["unique_patents"], "Token cache record count mismatch")
            except sqlite3.DatabaseError as exc:
                raise ValueError("Malformed token cache") from exc
        finally:
            connection.close()

    def _validate_ids(self, ids):
        require(isinstance(ids, list) and 2 <= len(ids) <= self.max_length and
                all(integer(value) and 0 <= value < 2 ** 32 for value in ids), "Invalid shared evidence token IDs")
        require(ids[0] == self.cls_id and ids[-1] == self.sep_id, "Shared evidence is missing declared CLS/SEP")

    def _build_cache(self):
        token_path = self.data_dir / "evidence.tokens.jsonl.gz"
        before = (token_path.stat().st_size, token_path.stat().st_mtime_ns)
        handle, name = tempfile.mkstemp(prefix="evidence.tokens.", suffix=".building", dir=str(self.cache_path.parent))
        os.close(handle)
        temporary, connection = Path(name), None
        count = 0
        try:
            connection = sqlite3.connect(str(temporary))
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE cache_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            # Ordinary rowid leaves avoid WITHOUT ROWID overflow pages for BLOBs.
            connection.execute("CREATE TABLE tokens(application_number TEXT PRIMARY KEY,raw_sha TEXT NOT NULL,"
                               "ids BLOB NOT NULL,token_count INTEGER NOT NULL,blob_sha TEXT NOT NULL)")
            for row in rows(token_path):
                app, raw_sha, ids = row.get("application_number"), row.get("raw_text_sha256"), row.get("input_ids")
                require(isinstance(app, str) and app, "Token row lacks application ID")
                require(isinstance(raw_sha, str) and SHA256.fullmatch(raw_sha), "Token row lacks raw text SHA")
                self._validate_ids(ids)
                encoded = array("I", ids)
                require(encoded.itemsize == 4, "Platform uint32 array format unavailable")
                if sys.byteorder != "little":
                    encoded.byteswap()
                blob = zlib.compress(encoded.tobytes(), 1)
                try:
                    connection.execute("INSERT INTO tokens VALUES(?,?,?,?,?)", (app, raw_sha, sqlite3.Binary(blob), len(ids), hashlib.sha256(blob).hexdigest()))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate token application ID: " + app) from exc
                count += 1
                if count % 5000 == 0:
                    connection.commit()
            require(count == self.manifest["unique_patents"], "Token source record count mismatch")
            require(before == (token_path.stat().st_size, token_path.stat().st_mtime_ns), "Token source changed while caching")
            connection.executemany("INSERT INTO cache_meta VALUES(?,?)", self._signature.items())
            connection.commit()
            connection.close()
            connection = None
            if self.cache_path.exists():
                self._validate_cache()
            else:
                temporary.replace(self.cache_path)
        finally:
            if connection is not None:
                connection.close()
            if temporary.exists():
                temporary.unlink()

    def _connect(self):
        require(not self._closed, "EvidenceDataset is closed")
        if self._connection is None or self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            self._connection = sqlite3.connect(self.cache_path.as_uri() + "?mode=ro", uri=True)
            self._pid = os.getpid()
        return self._connection

    def _fetch(self, references):
        expected = {}
        for ref in references:
            app, digest = ref["application_number"], ref["raw_text_sha256"]
            require(app not in expected or expected[app] == digest, "Application has conflicting body fingerprints")
            expected[app] = digest
        result, applications = {}, list(expected)
        connection = self._connect()
        for offset in range(0, len(applications), 512):
            batch = applications[offset:offset + 512]
            for app, raw_sha, blob, length, digest in connection.execute(
                    "SELECT application_number,raw_sha,ids,token_count,blob_sha FROM tokens WHERE application_number IN ({})".format(",".join("?" for _ in batch)), batch):
                require(raw_sha == expected[app], "Token/history body fingerprint mismatch: " + app)
                require(hashlib.sha256(blob).hexdigest() == digest, "Corrupt cached token BLOB")
                try:
                    uncompressed = zlib.decompress(blob)
                except zlib.error as exc:
                    raise ValueError("Corrupt compressed token BLOB") from exc
                require(len(uncompressed) == length * 4, "Corrupt cached token count")
                decoded = array("I")
                decoded.frombytes(uncompressed)
                if sys.byteorder != "little":
                    decoded.byteswap()
                ids = decoded.tolist()
                self._validate_ids(ids)
                result[app] = ids
        require(set(result) == set(expected), "No matching token application(s): " + ",".join(sorted(set(expected) - set(result))[:3]))
        return result

    @staticmethod
    def _patent(ref, tokens):
        keys = ("application_number", "event_id", "publication_date", "ipc_main", "ipc_all", "technical_group")
        if ref is None:
            return dict({key: None for key in keys}, input_ids=[], attention_mask=[])
        ids = list(tokens[ref["application_number"]])
        return dict({key: ref[key] for key in keys}, input_ids=ids, attention_mask=[1] * len(ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        require(integer(index), "Dataset index must be an integer")
        pair = self.pairs[index]
        query = self.queries[pair["query_id"]]
        source_refs = self.histories[query["source"]]["patents"]
        references = source_refs[:1 + self.max_neighbors]
        for doc_id in pair["candidate_ids"]:
            references += self.histories[doc_id]["patents"]
        tokens = self._fetch(references)
        neighbor_refs = source_refs[1:1 + self.max_neighbors]
        neighbors = [self._patent(ref, tokens) for ref in neighbor_refs]
        neighbors += [self._patent(None, tokens) for _ in range(self.max_neighbors - len(neighbors))]
        candidates = []
        for doc_id in pair["candidate_ids"]:
            refs = self.histories[doc_id]["patents"]
            patents = [self._patent(ref, tokens) for ref in refs]
            patents += [self._patent(None, tokens) for _ in range(self.evidence_slots - len(patents))]
            candidates.append({"doc_id": doc_id, "patents": patents,
                               "evidence_mask": [True] * len(refs) + [False] * (self.evidence_slots - len(refs))})
        return {"query_id": pair["query_id"], "features": {
            "source_doc_id": query["source"], "query": self._patent(source_refs[0], tokens),
            "query_neighbors": neighbors,
            "query_neighbor_mask": [True] * len(neighbor_refs) + [False] * (self.max_neighbors - len(neighbor_refs)),
            "candidate_doc_ids": list(pair["candidate_ids"]), "candidate_evidence": candidates},
            "targets": {"positive_mask": [doc_id in pair["positive_ids"] for doc_id in pair["candidate_ids"]]}}

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._closed = True

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_connection"], state["_pid"] = None, None
        return state

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def inspect_item(dataset, index):
    """Only structural counts: no text, names, IDs, or label evidence."""
    item = dataset[index]
    features = item["features"]
    candidates = features["candidate_evidence"]
    return {"split": dataset.split, "dataset_size": len(dataset), "index": index,
            "query_token_count": len(features["query"]["input_ids"]),
            "query_neighbor_slots": len(features["query_neighbors"]),
            "query_neighbor_count": sum(features["query_neighbor_mask"]),
            "candidate_count": len(candidates), "positive_count": sum(item["targets"]["positive_mask"]),
            "evidence_slots_per_candidate": dataset.evidence_slots,
            "candidate_evidence_counts": [sum(row["evidence_mask"]) for row in candidates],
            "max_evidence_token_count": max((len(patent["input_ids"]) for row in candidates for patent in row["patents"]), default=0),
            "token_cache_hit": dataset.cache_hit, "tokens_include_cls_sep": True,
            "label_evidence_in_features": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("inspect")
    command.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/patent/cooperation_v2")
    command.add_argument("--split", choices=("train", "val", "test"), default="train")
    command.add_argument("--index", type=int, default=0)
    command.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    with EvidenceDataset(args.data_dir, args.split, args.tokenizer) as dataset:
        print(json.dumps(inspect_item(dataset, args.index), indent=2))


if __name__ == "__main__":
    main()
