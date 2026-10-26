"""Independent scoring and integrity cases for the uncompressed max reference."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
import partner_dense_baseline as dense
import partner_evidence_max_baseline as maximum

SPEC = importlib.util.spec_from_file_location("max_dense_fixture", ROOT / "tests/test_partner_dense_baseline.py")
fixture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_module)


class EvidenceMaxScoringTests(unittest.TestCase):
    def test_padding_cannot_outscore_negative_real_matches(self):
        evidence = torch.tensor([[[-1., 0.], [100., 0.]], [[-0.5, 3. ** 0.5 / 2], [0., 0.]]])
        mask = torch.tensor([[True, False], [True, False]])
        scores = maximum.max_evidence_scores(torch.tensor([[1., 0.]]), F.normalize(evidence, dim=-1), mask)
        torch.testing.assert_close(scores, torch.tensor([[-1., -0.5]]))
        with self.assertRaisesRegex(ValueError, "real evidence"):
            maximum.max_evidence_scores(torch.ones(1, 2), evidence, torch.zeros_like(mask))

    def test_individual_max_and_normalized_mean_can_reverse_the_ranking(self):
        query = torch.tensor([[1., 0.]])
        evidence = torch.tensor([[[1., 0.], [-1., 0.]], [[0.8, 0.6], [0.8, 0.6]]])
        scores = maximum.max_evidence_scores(query, F.normalize(evidence, dim=-1), torch.ones(2, 2, dtype=torch.bool))
        mean_scores = query @ F.normalize(evidence.mean(1), dim=-1).T
        torch.testing.assert_close(scores, torch.tensor([[1., 0.8]]))
        torch.testing.assert_close(mean_scores, torch.tensor([[0., 0.8]]))
        self.assertEqual(scores.argmax().item(), 0)
        self.assertEqual(mean_scores.argmax().item(), 1)

    def test_full_pool_exclusions_ties_and_gold_independence(self):
        split = SimpleNamespace(doc_ids=[1, 2, 3, 4], evidence=torch.ones(4, 2, 2),
                                mask=torch.ones(4, 2, dtype=torch.bool), query_ids=["q", "miss"],
                                queries=torch.ones(2, 2), excluded={"q": {1, 2}, "miss": set()},
                                targets={"q": {4}, "miss": {999}})
        before = split.evidence.clone()
        metrics, rankings = maximum.evaluate_split(split, torch.device("cpu"), 1)
        self.assertEqual(rankings, {"q": [3, 4], "miss": [1, 2, 3, 4]})
        self.assertEqual(metrics["recall@100"], 0.5)
        self.assertEqual(metrics["P@1"], 0.)
        split.targets["q"] = {3}
        _, changed_gold_rankings = maximum.evaluate_split(split, torch.device("cpu"), 2)
        self.assertEqual(rankings, changed_gold_rankings)
        torch.testing.assert_close(split.evidence, before)


class EvidenceMaxPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.DenseBaselineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.directory = self.fixture.directory
        self.embedding_dir = self.fixture.encode()

    def args(self, name="fullmax.val.json"):
        return SimpleNamespace(data_dir=self.directory, embedding_dir=self.embedding_dir,
                               split="val", output=self.directory / name, device="cpu", batch_size=2)

    def test_complete_original_cache_evaluation_and_memory_accounting(self):
        before = {name: dense.fingerprint(self.embedding_dir / name)
                  for name in ("embedding_manifest.json", "index.jsonl", "vectors.npy")}
        store = dense.EmbeddingStore(self.embedding_dir, self.directory)
        split = maximum.CompleteEvidenceVectors(self.directory, "val", store)
        with dense.EvidenceDataset(self.directory, "val") as dataset:
            expected = np.stack([store.mean(dataset.histories[dataset.queries[qid]["source"]]["patents"][:6])
                                 for qid in split.query_ids])
        np.testing.assert_allclose(split.queries.numpy(), expected)
        self.assertEqual(split.evidence.shape, (4, 20, 4))
        self.assertEqual(split.mask.sum().item(), 25)
        with contextlib.redirect_stdout(io.StringIO()):
            maximum.evaluate(self.args())
        report = dense.read_json(self.args().output)
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["candidate_count"], 4)
        self.assertEqual(report["positive_count"], 4)
        self.assertEqual(report["evidence_vector_count"], 25)
        self.assertEqual(report["logical_evidence_vector_bytes"], 400)
        self.assertEqual(report["allocated_candidate_tensor_bytes"], 1280)
        self.assertEqual(report["allocated_mask_tensor_bytes"], 80)
        self.assertFalse(report["trained_projection"])
        self.assertFalse(report["fixed_index_budget_comparison"])
        self.assertFalse(report["performance_upper_bound_claimed"])
        self.assertFalse(report["gold_used_for_scoring"])
        self.assertEqual(before, {name: dense.fingerprint(self.embedding_dir / name) for name in before})

    def test_existing_report_and_tampered_cache_are_refused(self):
        args = self.args()
        args.output.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing evaluation report"):
            maximum.evaluate(args)
        self.assertEqual(args.output.read_text(encoding="utf-8"), "keep")
        with (self.embedding_dir / "vectors.npy").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
            maximum.evaluate(self.args("tampered.json"))

    def test_future_history_and_self_consistent_wrong_text_fingerprint_are_refused(self):
        history = self.directory / "histories.val.jsonl"
        original = history.read_bytes()
        rows = list(dense.rows(history))
        rows[1]["patents"][0]["publication_date"] = "2023-01-01"
        fixture_module.fixture_module.write_rows(history, rows)
        self.fixture.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "Future evidence"):
            maximum.evaluate(self.args())
        history.write_bytes(original)
        self.fixture.refresh_manifest()
        index = self.embedding_dir / "index.jsonl"
        rows = list(dense.rows(index))
        rows[0]["raw_text_sha256"] = "f" * 64
        fixture_module.fixture_module.write_rows(index, rows)
        manifest = dense.read_json(self.embedding_dir / "embedding_manifest.json")
        manifest["artifacts"]["index.jsonl"] = dense.fingerprint(index)
        dense.write_json(self.embedding_dir / "embedding_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "text SHA differs"):
            maximum.evaluate(self.args())


if __name__ == "__main__":
    unittest.main()
