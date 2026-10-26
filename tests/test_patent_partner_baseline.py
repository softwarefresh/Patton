"""Independent boundary checks for candidate retrieval and evaluation."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
SPEC = importlib.util.spec_from_file_location("patent_partner_baseline", ROOT / "data_pipeline/patent_partner_baseline.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def doc(doc_id, text):
    return {"doc_id": doc_id, "contents": text,
            "company_code": "C{}".format(doc_id), "company_name": "企业{}".format(doc_id)}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class PartnerFTSTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "tmp"
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="partner_fts_test_", dir=str(base))
        self.directory = Path(self.temporary.name).resolve()
        self.directory.relative_to(base.resolve())

    def tearDown(self):
        self.directory.relative_to((ROOT / "tmp").resolve())
        self.temporary.cleanup()

    def index(self, documents, fingerprint="one", budget=128):
        return baseline.FTSIndex(self.directory / "index.sqlite3", documents, fingerprint, budget)

    def test_chinese_match_exclusions_ties_and_zero_padding(self):
        with self.index([doc(7, "电池检测"), doc(2, "电池检测"),
                         doc(0, "水泥生产"), doc(3, "")]) as index:
            result = index.rank("电池检测", excluded=[7, 999], top_k=10)
            self.assertEqual([row[0] for row in result], [2, 0, 3])
            self.assertGreater(result[0][1], 0)
            self.assertEqual(result[1:], [(0, 0.0), (3, 0.0)])
            all_rows = index.rank("电池检测", top_k=None)
            self.assertEqual([row[0] for row in all_rows[:2]], [2, 7])
            self.assertEqual(index.rank("missing", excluded=[2], top_k=2), [(0, 0.0), (3, 0.0)])

    def test_low_df_budget_ignores_unseen_terms_and_is_order_invariant(self):
        with self.index([doc(0, "common rare"), doc(1, "common"), doc(2, "common")], budget=1) as index:
            first = index.rank("common rare unknown", top_k=3)
            self.assertEqual(index.last_query_terms, ("rare",))
            self.assertEqual(first[0][0], 0)
            self.assertGreater(first[0][1], 0)
            self.assertEqual(first[1:], [(1, 0.0), (2, 0.0)])
            self.assertEqual(index.rank("unknown rare common", top_k=3), first)
            long_query = " ".join("unknown{}".format(i) for i in range(1200)) + " rare"
            self.assertEqual(index.rank(long_query, top_k=1)[0][0], 0)
            self.assertEqual(index.last_query_terms, ("rare",))

    def test_match_expression_is_safe_for_quotes_operators_and_punctuation(self):
        with self.index([doc(1, "电池 rare"), doc(2, "其他")]) as index:
            result = index.rank('"rare" OR NEAR(*) - 电池 : [ ]', top_k=2)
            self.assertEqual(result[0][0], 1)
            self.assertGreater(result[0][1], 0)

    def test_excluded_source_only_terms_cannot_use_up_query_budget(self):
        with self.index([doc(0, "sourceword1 sourceword2"), doc(1, "partnerword"),
                         doc(2, "partnerword other")], budget=1) as index:
            ranking = index.rank("sourceword1 sourceword2 partnerword", excluded=[0], top_k=2)
            self.assertEqual(index.last_query_terms, ("partnerword",))
            self.assertTrue(all(score > 0 for _, score in ranking))
            index.rank("sourceword1 sourceword2 partnerword", excluded=[], top_k=1)
            self.assertEqual(index.last_query_terms, ("sourceword1",))

    def test_term_present_only_in_multiple_excluded_documents_is_omitted(self):
        with self.index([doc(0, "sharedexcluded"), doc(1, "sharedexcluded"),
                         doc(2, "allowedword")], budget=1) as index:
            ranking = index.rank("sharedexcluded allowedword", excluded=[0, 1], top_k=2)
            self.assertEqual(index.last_query_terms, ("allowedword",))
            self.assertEqual(ranking[0][0], 2)
            self.assertGreater(ranking[0][1], 0)

    def test_cache_signature_and_document_content_invalidate(self):
        documents = [doc(1, "alpha"), doc(2, "beta")]
        with self.index(documents) as index:
            self.assertFalse(index.cache_hit)
        with self.index(list(reversed(documents))) as index:
            self.assertTrue(index.cache_hit)
        with self.index(documents, fingerprint="two") as index:
            self.assertFalse(index.cache_hit)
        changed = [doc(1, "gamma"), doc(2, "beta")]
        with self.index(changed, fingerprint="two") as index:
            self.assertFalse(index.cache_hit)
            self.assertEqual(index.rank("alpha", top_k=2), [(1, 0.0), (2, 0.0)])

    def test_foreign_database_is_never_overwritten(self):
        path = self.directory / "index.sqlite3"
        connection = sqlite3.connect(str(path))
        with connection:
            connection.execute("CREATE TABLE important(value TEXT)")
            connection.execute("INSERT INTO important VALUES ('keep')")
        connection.close()
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "unrecognized|foreign"):
            self.index([doc(1, "alpha")])
        self.assertEqual(path.read_bytes(), before)

    def test_empty_allowed_pool_and_invalid_inputs(self):
        with self.index([doc(0, "alpha")]) as index:
            self.assertEqual(index.rank("alpha", excluded=[0]), [])
            self.assertEqual(index.rank("alpha", top_k=0), [])
            with self.assertRaises(ValueError):
                index.rank("alpha", top_k=-1)
            with self.assertRaises(ValueError):
                index.rank("alpha", excluded=["0"])
        index.close()
        with self.assertRaises(RuntimeError):
            index.rank("alpha")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.index([doc(1, "a"), doc(1, "b")])

    def test_multi_positive_metrics_and_zero_matches_are_zero(self):
        metrics = baseline.per_query_metrics([(9, 3.0), (2, 2.0), (7, 1.0), (4, 0.0)], [2, 4, 2])
        self.assertEqual(metrics["recall@5"], 0.5)
        self.assertEqual(metrics["Hit@5"], 1.0)
        self.assertEqual(metrics["MRR@100"], 0.5)
        self.assertEqual(metrics["P@1"], 0.0)
        self.assertTrue(all(value == 0 for value in baseline.per_query_metrics([(2, 0.0)], [2]).values()))
        long_ranking = [(i, 1.0) for i in range(101)]
        self.assertEqual(baseline.per_query_metrics(long_ranking, [100])["MRR@100"], 0.0)

    def test_evaluation_keeps_missed_gold_and_does_not_read_answers_for_rank(self):
        write_jsonl(self.directory / "documents.test.jsonl", [doc(0, "source"), doc(1, "rare"), doc(2, "other")])
        query = {"query_id": "Q", "source_doc_id": 0, "q_text": "", "q_n_text": ["rare"],
                 "positive_doc_ids": [1, 99], "excluded_doc_ids": []}
        write_jsonl(self.directory / "queries.test.jsonl", [query])
        result = baseline.evaluate(self.directory, output=self.directory / "first", top_k=1)
        self.assertEqual(result["recall@5"], 0.5)
        self.assertEqual(result["MRR@100"], 1.0)
        self.assertEqual(result["gold_outside_pool_occurrences"], 1)
        first = list(baseline.read_jsonl(self.directory / "first/recommendations.test.jsonl"))[0]
        query["positive_doc_ids"] = [2]
        write_jsonl(self.directory / "queries.test.jsonl", [query])
        second_metrics = baseline.evaluate(self.directory, output=self.directory / "second", top_k=1)
        second = list(baseline.read_jsonl(self.directory / "second/recommendations.test.jsonl"))[0]
        self.assertEqual(first["recommendations"], second["recommendations"])
        self.assertEqual(second_metrics["P@1"], 0.0)

    def test_evaluation_no_overlap_and_no_allowed_candidates_count_as_zero(self):
        write_jsonl(self.directory / "documents.test.jsonl", [doc(0, "source"), doc(1, "other")])
        queries = [
            {"query_id": "zero", "source_doc_id": 0, "q_text": "missing", "q_n_text": [],
             "positive_doc_ids": [1], "excluded_doc_ids": []},
            {"query_id": "none", "source_doc_id": 0, "q_text": "other", "q_n_text": [],
             "positive_doc_ids": [99], "excluded_doc_ids": [1]},
        ]
        write_jsonl(self.directory / "queries.test.jsonl", queries)
        result = baseline.evaluate(self.directory)
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["zero_token_overlap_query_count"], 2)
        self.assertEqual(result["P@1"], 0.0)
        self.assertEqual(result["recall@100"], 0.0)
        self.assertEqual(result["MRR@100"], 0.0)

    def test_outputs_cannot_replace_input_documents(self):
        documents = self.directory / "documents.test.jsonl"
        write_jsonl(documents, [doc(0, "source"), doc(1, "other")])
        write_jsonl(self.directory / "queries.test.jsonl", [])
        before = documents.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            baseline.evaluate(self.directory, output=documents)
        self.assertEqual(documents.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
