"""Deterministic train-only mining checks without tokens, BERT, or downloads."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
import partner_dense_baseline as dense
from partner_teacher_candidates import build_dense_training_candidates


def candidate_order(qid, ids):
    return sorted(ids, key=lambda doc_id: (hashlib.sha256(
        (qid + "\0" + str(doc_id)).encode()).digest(), doc_id))


class DenseTeacherCandidateTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="teacher_mining_", dir=str(temporary_root))
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        # Scores: excluded self 0=1, random 2=1, dense candidates 3/4 tie=1,
        # known positive 1=.8, old lexical hard 5=0, last candidate 6=-1.
        self.documents = torch.tensor([[1., 0.], [.8, .6], [1., 0.], [1., 0.],
                                       [1., 0.], [0., 1.], [-1., 0.]])
        self.row = {"query_id": "q", "positive_doc_ids": [1], "hard_negative_doc_ids": [5],
                    "random_negative_doc_ids": [2], "negative_doc_ids": [5, 2]}
        self.write_row()
        self.training = SimpleNamespace(
            split="train", doc_ids=list(range(7)), documents=self.documents.clone(),
            queries=torch.tensor([[1., 0.]]), query_ids=["q"], position={i: i for i in range(7)},
            pairs={"q": candidate_order("q", [1, 2, 5])}, targets={"q": {1}}, excluded={"q": {0}},
            fingerprints={"train.pairs.jsonl": dense.sha256_file(self.directory / "train.pairs.jsonl")})
        self.model = dense.Projection(2).requires_grad_(False)

    def write_row(self):
        (self.directory / "train.pairs.jsonl").write_text(json.dumps(self.row) + "\n", encoding="utf-8")

    def mine(self, split=None, batch=32):
        return build_dense_training_candidates(split or self.training, self.directory, self.model,
                                               torch.device("cpu"), query_batch_size=batch)

    def refresh(self):
        self.write_row()
        self.training.fingerprints["train.pairs.jsonl"] = dense.sha256_file(self.directory / "train.pairs.jsonl")

    def test_replaces_only_hard_keeps_random_positives_and_orders_ties_by_id(self):
        new_pairs, audit, records = self.mine()
        self.assertEqual(records[0]["dense_hard_doc_ids"], [3])
        self.assertEqual(records[0]["retained_random_doc_ids"], [2])
        self.assertEqual(records[0]["positive_doc_ids"], [1])
        self.assertEqual(records[0]["canonical_candidate_ids"], candidate_order("q", [1, 2, 5]))
        self.assertEqual(new_pairs, {"q": candidate_order("q", [1, 2, 3])})
        self.assertEqual(records[0]["candidate_ids"], new_pairs["q"])
        self.assertEqual(audit["canonical_negative_count"]["sum"], 2)
        self.assertEqual(audit["new_negative_count"]["sum"], 2)
        self.assertEqual(audit["replaced_hard_count"]["sum"], 1)
        self.assertFalse(audit["val_accessed"])
        self.assertFalse(audit["test_accessed"])
        self.assertFalse(audit["unobserved_items_are_verified_negatives"])

    def test_audit_coverage_and_full_pool_metrics_use_distinct_correct_denominators(self):
        _, audit, _ = self.mine()
        # Allowed unobserved pool is {2,3,4,5,6}, so top20 denominator is 5.
        self.assertAlmostEqual(audit["coverage"]["canonical"]["top20"], 2 / 5)
        self.assertAlmostEqual(audit["coverage"]["new"]["top100"], 2 / 5)
        self.assertEqual(audit["positive_full_pool_rank_summary"]["median"], 4.)
        self.assertAlmostEqual(audit["positive_minus_100th_allowed_score_summary"]["mean"], 1.8, places=6)
        metrics = audit["base_full_training_metrics"]
        self.assertEqual(metrics["recall@100"], 1.)
        self.assertEqual(metrics["P@1"], 0.)
        self.assertEqual(metrics["MRR@100"], .25)
        self.assertEqual(metrics["split"], "train")
        self.assertIn("min(k,eligible_unobserved_count)", audit["coverage_denominator"])
        json.dumps(audit, allow_nan=False)
        json.dumps(self.mine()[2], allow_nan=False)

    def test_nontrain_split_is_rejected_before_reading_missing_file(self):
        (self.directory / "train.pairs.jsonl").unlink()
        for split in ("val", "test"):
            with self.subTest(split=split):
                self.training.split = split
                with self.assertRaisesRegex(ValueError, "train only"):
                    self.mine()

    def test_inputs_model_weights_and_canonical_file_are_unchanged(self):
        before = copy.deepcopy(self.training)
        original_bytes = (self.directory / "train.pairs.jsonl").read_bytes()
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.model.train()
        self.mine()
        self.assertTrue(self.model.training)
        self.assertEqual((self.directory / "train.pairs.jsonl").read_bytes(), original_bytes)
        for name in ("doc_ids", "query_ids", "position", "pairs", "targets", "excluded", "fingerprints"):
            self.assertEqual(getattr(self.training, name), getattr(before, name))
        torch.testing.assert_close(self.training.documents, before.documents)
        torch.testing.assert_close(self.training.queries, before.queries)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)

    def test_pair_source_hash_and_loaded_candidate_order_are_checked(self):
        with (self.directory / "train.pairs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.mine()
        self.refresh()
        self.training.pairs["q"].reverse()
        with self.assertRaisesRegex(ValueError, "IDs/order differ"):
            self.mine()

    def test_overlap_and_invalid_budget_are_refused_but_old_hard_is_valid_backfill(self):
        self.row["hard_negative_doc_ids"] = [2]
        self.row["negative_doc_ids"] = [2]
        self.refresh()
        with self.assertRaisesRegex(ValueError, "hard/random overlap"):
            self.mine()
        self.row["hard_negative_doc_ids"] = [5]
        self.row["negative_doc_ids"] = [5, 2]
        self.refresh()
        # A loaded training split with no free dense alternative outside its
        # known positive, retained random item, and historical exclusions.
        self.training.excluded["q"] = {0, 3, 4, 5, 6}
        with self.assertRaisesRegex(ValueError, "excluded IDs"):
            self.mine()
        # Without a new company outside the old candidate list, two old hard
        # items can still satisfy two distinct hard slots.
        self.row["hard_negative_doc_ids"] = [5, 6]
        self.row["negative_doc_ids"] = [5, 6, 2]
        self.training.pairs["q"] = candidate_order("q", [1, 2, 5, 6])
        self.training.excluded["q"] = {0, 3, 4}
        self.refresh()
        # The pool has exactly the old hard items, so they are valid backfill;
        # there is no reason to fail merely because no *new* company is free.
        result, _, records = self.mine()
        self.assertEqual(records[0]["dense_hard_doc_ids"], [5, 6])
        self.assertEqual(len(result["q"]), 4)

    def test_two_query_batch_sizes_match_and_final_order_has_no_positive_prefix(self):
        self.training.query_ids.append("z")
        self.training.queries = torch.cat((self.training.queries, torch.tensor([[0., 1.]])))
        other = dict(self.row, query_id="z")
        path = self.directory / "train.pairs.jsonl"
        path.write_text(json.dumps(self.row) + "\n" + json.dumps(other) + "\n", encoding="utf-8")
        self.training.fingerprints[path.name] = dense.sha256_file(path)
        self.training.targets["z"] = {1}
        self.training.excluded["z"] = {0}
        self.training.pairs["z"] = candidate_order("z", [1, 2, 5])
        one = self.mine(batch=1)
        two = self.mine(batch=2)
        self.assertEqual(one[0], two[0])
        self.assertEqual(one[2], two[2])
        self.assertEqual(one[1], two[1])
        for record in one[2]:
            qid = record["query_id"]
            self.assertEqual(record["candidate_ids"], candidate_order(qid, record["candidate_ids"]))


if __name__ == "__main__":
    unittest.main()
