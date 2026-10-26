"""Boundary tests for canonical ID-joined, separately encoded patent sets."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("partner_evidence_dataset", ROOT / "data_pipeline/partner_evidence_dataset.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write_rows(path, records):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")


class EvidenceDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="evidence_dataset_test_", dir=str(temporary_root))
        self.directory = Path(self.temporary.name).resolve()
        self.directory.relative_to(temporary_root.resolve())
        self.histories, self.documents, self.queries, self.pairs, self.tokens, self.evidence = [], [], [], [], [], []
        for doc_id, count in ((0, 2), (1, 20), (2, 2), (3, 1)):
            patents = []
            for offset in range(count):
                app = "APP{}_{}".format(doc_id, offset)
                raw_sha = hashlib.sha256(app.encode()).hexdigest()
                patents.append({"application_number": app, "event_id": doc_id * 100 + offset,
                                "publication_date": "2021-12-{:02d}".format(28 - offset),
                                "ipc_main": "G06F1/00", "ipc_all": "G06F1/00;G06Q1/00",
                                "technical_group": "G06F", "raw_text_sha256": raw_sha,
                                "applicant_name_at_event": "SECRET_HISTORICAL_NAME"})
                self.tokens.append({"application_number": app, "input_ids": [101, 70000 + doc_id * 100 + offset, 102],
                                    "raw_text_sha256": raw_sha})
                self.evidence.append({"application_number": app, "raw_text_sha256": raw_sha, "text": "SECRET_TEXT"})
            self.histories.append({"doc_id": doc_id, "company_code": "C{}".format(doc_id),
                                   "forecast_year": 2022, "cutoff_exclusive": "2022-01-01", "patents": patents})
            self.documents.append({"doc_id": doc_id, "company_code": "C{}".format(doc_id),
                                   "evidence_application_numbers": [row["application_number"] for row in patents],
                                   "excluded_doc_ids": [doc_id], "company_name": "SECRET_CURRENT_NAME",
                                   "contents": "SECRET_TEXT"})
        for source in (0, 3):
            first = self.histories[source]["patents"][0]
            qid = "query_{}".format(source)
            self.queries.append({"query_id": qid, "source_doc_id": source,
                                 "query_application_number": first["application_number"],
                                 "query_event_id": first["event_id"], "query_publication_date": first["publication_date"],
                                 "input_application_numbers": self.documents[source]["evidence_application_numbers"],
                                 "forecast_year": 2022, "cutoff_exclusive": "2022-01-01",
                                 "positive_doc_ids": [1, 2], "excluded_doc_ids": [source],
                                 "q_text": "SECRET_QUERY_TEXT", "label_evidence": [{"date": "2099-01-01"}]})
            negative = 3 if source == 0 else 0
            self.pairs.append({"query_id": qid, "source_doc_id": source,
                               "query_application_number": first["application_number"],
                               "positive_doc_ids": [1, 2, 1], "negative_doc_ids": [negative, negative],
                               "hard_negative_doc_ids": [negative], "random_negative_doc_ids": [negative]})
        self.write_metadata()
        self.write_tokens()

    def tearDown(self):
        self.directory.relative_to((ROOT / "tmp").resolve())
        self.temporary.cleanup()

    def write_metadata(self):
        write_rows(self.directory / "documents.train.jsonl", self.documents)
        write_rows(self.directory / "histories.train.jsonl", self.histories)
        # Physical file order is deliberately different: the join must use IDs.
        write_rows(self.directory / "queries.train.jsonl", list(reversed(self.queries)))
        write_rows(self.directory / "train.pairs.jsonl", self.pairs)

    def write_tokens(self):
        source = self.directory / "evidence.jsonl.gz"
        output = self.directory / "evidence.tokens.jsonl.gz"
        write_rows(source, self.evidence)
        write_rows(output, self.tokens)
        manifest = {"unique_patents": len(self.tokens), "includes_cls_sep": True,
                    "cls_token_id": 101, "sep_token_id": 102, "max_length": 256,
                    "source_sha256": module.sha256_file(source), "output_sha256": module.sha256_file(output),
                    "output_bytes": output.stat().st_size}
        (self.directory / "tokenization_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def dataset(self):
        return module.EvidenceDataset(self.directory, "train")

    def test_all_positives_id_join_twenty_independent_patents_and_missing_masks(self):
        with self.dataset() as dataset:
            item = dataset[0]
            self.assertEqual(len(dataset), 2)
            self.assertEqual(item["query_id"], "query_0")
            features = item["features"]
            self.assertEqual(set(features["candidate_doc_ids"]), {1, 2, 3})
            self.assertEqual(dict(zip(features["candidate_doc_ids"], item["targets"]["positive_mask"])),
                             {1: True, 2: True, 3: False})
            self.assertEqual(features["query"]["input_ids"], [101, 70000, 102])
            self.assertEqual(features["query_neighbor_mask"], [True, False, False, False, False])
            self.assertEqual(features["query_neighbors"][1]["input_ids"], [])
            self.assertIsNone(features["query_neighbors"][1]["publication_date"])
            by_id = {row["doc_id"]: row for row in features["candidate_evidence"]}
            candidate = by_id[1]
            self.assertEqual(len(candidate["patents"]), 20)
            self.assertEqual(sum(candidate["evidence_mask"]), 20)
            self.assertEqual(len({row["application_number"] for row in candidate["patents"]}), 20)
            self.assertTrue(all(row["input_ids"] == [101, 70100 + i, 102] for i, row in enumerate(candidate["patents"])))
            short = by_id[2]
            self.assertEqual(sum(short["evidence_mask"]), 2)
            self.assertEqual(short["patents"][2]["input_ids"], [])
            serialized = json.dumps(features)
            self.assertNotIn("SECRET", serialized)
            self.assertNotIn("label_evidence", serialized)
            self.assertNotIn("positive", serialized)
            self.assertNotIn("2099", serialized)
            self.assertFalse(hasattr(dataset, "tokens"))
            connection = sqlite3.connect(str(dataset.cache_path))
            try:
                self.assertEqual(connection.execute("SELECT DISTINCT typeof(ids) FROM tokens").fetchall(), [("blob",)])
            finally:
                connection.close()
        with self.dataset() as dataset:
            self.assertTrue(dataset.cache_hit)
            self.assertEqual(module.inspect_item(dataset, 1)["query_neighbor_count"], 0)

    def test_reordered_applications_and_pair_lost_positive_are_rejected(self):
        self.documents[1]["evidence_application_numbers"].reverse()
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "application order"):
            self.dataset()
        self.documents[1]["evidence_application_numbers"].reverse()
        self.pairs[0]["positive_doc_ids"] = [1]
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "loses or changes"):
            self.dataset()

    def test_query_center_mismatch_and_future_history_are_rejected(self):
        self.pairs[0]["query_application_number"] = "WRONG_APPLICATION"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "application mismatch"):
            self.dataset()
        self.pairs[0]["query_application_number"] = self.queries[0]["query_application_number"]
        self.histories[1]["patents"][0]["publication_date"] = "2022-01-01"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "Future evidence"):
            self.dataset()

    def test_missing_token_application_and_body_mismatch_are_rejected(self):
        original = self.tokens.pop()
        self.write_tokens()
        with self.dataset() as dataset:
            with self.assertRaisesRegex(ValueError, "No matching token application"):
                dataset[0]
        # Metadata mismatches must also fail when the shared token cache is valid.
        self.histories[1]["patents"][0]["raw_text_sha256"] = "f" * 64
        self.write_metadata()
        with self.dataset() as dataset:
            with self.assertRaisesRegex(ValueError, "body fingerprint mismatch"):
                dataset[0]
        self.tokens.append(original)

    def test_stale_and_foreign_caches_are_not_overwritten(self):
        with self.dataset() as dataset:
            path = dataset.cache_path
        before = path.read_bytes()
        manifest_path = self.directory / "tokenization_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["new_source_revision"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "cache fingerprint mismatch"):
            self.dataset()
        self.assertEqual(path.read_bytes(), before)
        # Foreign-file refusal uses its own nested directory, no cache deletion.
        foreign = self.directory / "foreign"
        foreign.mkdir()
        for source in self.directory.iterdir():
            if source.is_file():
                (foreign / source.name).write_bytes(source.read_bytes())
        cache = foreign / "token_cache/evidence.tokens.sqlite3"
        cache.parent.mkdir()
        connection = sqlite3.connect(str(cache))
        connection.execute("CREATE TABLE important(value TEXT)")
        connection.commit()
        connection.close()
        foreign_before = cache.read_bytes()
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            module.EvidenceDataset(foreign, "train")
        self.assertEqual(cache.read_bytes(), foreign_before)

    def test_candidate_cannot_use_a_later_split_cutoff(self):
        self.histories[1]["forecast_year"] = 2024
        self.histories[1]["cutoff_exclusive"] = "2024-01-01"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "cutoff/year/split"):
            self.dataset()

    def test_candidate_order_is_stable_without_positive_position_shortcut(self):
        with self.dataset() as dataset:
            first = dataset[0]
        # Swap which of the existing candidates is positive, keeping the pool.
        self.queries[0]["positive_doc_ids"] = [2, 3]
        self.pairs[0]["positive_doc_ids"] = [3, 2]
        self.pairs[0]["negative_doc_ids"] = [1]
        self.pairs[0]["hard_negative_doc_ids"] = [1]
        self.pairs[0]["random_negative_doc_ids"] = []
        self.write_metadata()
        with self.dataset() as dataset:
            second = dataset[0]
        self.assertEqual(first["features"], second["features"])
        self.assertNotEqual(first["targets"], second["targets"])

    def test_dataset_manifest_artifact_hash_is_enforced(self):
        artifact_names = ["documents.train.jsonl", "histories.train.jsonl", "queries.train.jsonl",
                          "train.pairs.jsonl", "evidence.jsonl.gz"]
        artifacts = {name: {"sha256": module.sha256_file(self.directory / name),
                            "bytes": (self.directory / name).stat().st_size} for name in artifact_names}
        (self.directory / "dataset_manifest.json").write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
        with self.dataset() as dataset:
            dataset[0]
        self.documents[0]["contents"] = "TAMPERED_UNUSED_TEXT"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "Canonical artifact fingerprint mismatch"):
            self.dataset()

    def test_duplicate_or_special_token_invalid_source_leaves_no_final_cache(self):
        self.tokens.append(dict(self.tokens[0]))
        self.write_tokens()
        with self.assertRaisesRegex(ValueError, "Duplicate token application"):
            self.dataset()
        self.assertFalse((self.directory / "token_cache/evidence.tokens.sqlite3").exists())
        self.assertFalse(list((self.directory / "token_cache").glob("*.building")))
        self.tokens.pop()
        self.tokens[0]["input_ids"] = [70000, 102]
        self.write_tokens()
        with self.assertRaisesRegex(ValueError, "CLS/SEP"):
            self.dataset()
        self.assertFalse((self.directory / "token_cache/evidence.tokens.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
