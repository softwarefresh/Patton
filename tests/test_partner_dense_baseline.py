"""CPU boundary and miniature pipeline checks; no model downloads required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
import partner_dense_baseline as dense

SPEC = importlib.util.spec_from_file_location("dense_evidence_fixture", ROOT / "tests/test_partner_evidence_dataset.py")
fixture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_module)


class MiniEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0, hidden_size=4, vocab_size=100000)

    def forward(self, input_ids, attention_mask):
        values = input_ids.float()
        hidden = torch.stack((values % 7, values % 11, values % 13, torch.ones_like(values)), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


class DenseBaselineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.EvidenceDatasetTests()
        self.fixture.setUp()
        self.directory = self.fixture.directory
        self.model_dir = self.directory / "base_model"
        self.model_dir.mkdir()
        for filename, content in (("config.json", '{}'), ("tokenizer.json", '{}'),
                                  ("vocab.txt", '[PAD]\n[CLS]\n[SEP]\n'), ("pytorch_model.bin", 'fake test weights')):
            (self.model_dir / filename).write_text(content, encoding="utf-8")
        manifest_path = self.directory / "tokenization_manifest.json"
        manifest = dense.read_json(manifest_path)
        manifest["tokenizer_json_sha256"] = dense.sha256_file(self.model_dir / "tokenizer.json")
        manifest["vocab_sha256"] = dense.sha256_file(self.model_dir / "vocab.txt")
        dense.write_json(manifest_path, manifest)
        for split, year in (("val", 2023), ("test", 2024)):
            for stem in ("documents", "histories", "queries"):
                values = list(fixture_module.module.rows(self.directory / (stem + ".train.jsonl")))
                if stem in ("histories", "queries"):
                    for row in values:
                        row["forecast_year"] = year
                        row["cutoff_exclusive"] = str(year) + "-01-01"
                fixture_module.write_rows(self.directory / (stem + "." + split + ".jsonl"), values)
            fixture_module.write_rows(self.directory / (split + ".pairs.jsonl"), self.fixture.pairs)
        self.refresh_manifest()

    def refresh_manifest(self):
        names = ["evidence.jsonl.gz"]
        for split in ("train", "val", "test"):
            names += [stem + "." + split + ".jsonl" for stem in ("documents", "histories", "queries")]
            names.append(split + ".pairs.jsonl")
        dense.write_json(self.directory / "dataset_manifest.json",
                         {"artifacts": {name: dense.fingerprint(self.directory / name) for name in names}})

    def tearDown(self):
        self.fixture.tearDown()

    def encode(self, name="embeddings", maximum=0):
        output = self.directory / name
        args = SimpleNamespace(data_dir=self.directory, model_dir=self.model_dir,
                               output_dir=output, batch_size=3, max_patents=maximum, device="cpu")
        factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: MiniEncoder())
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoModel=factory)}), contextlib.redirect_stdout(io.StringIO()):
            dense.encode(args)
        return output

    def test_masked_mean_excludes_padding_and_keeps_existing_special_tokens(self):
        hidden = torch.tensor([[[1., 3.], [5., 7.], [999., 999.]], [[1., 2.], [3., 4.], [5., 6.]]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        result = dense.masked_token_mean(hidden, mask)
        torch.testing.assert_close(result, torch.tensor([[3., 5.], [3., 4.]]))

    def test_multi_positive_average_gives_every_positive_gradient(self):
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
        loss = dense.multi_positive_loss(scores, torch.tensor([True, False, True, False]))
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.tensor([-0.25, 0.25, -0.25, 0.25]))
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            dense.multi_positive_loss(scores, torch.zeros(4, dtype=torch.bool))

    def test_all_query_misses_and_multiple_positives_remain_in_denominator(self):
        result = dense.ranking_metrics({"a": [3, 1], "b": []}, {"a": {1, 2}, "b": {8}})
        self.assertEqual(result["recall@100"], 0.25)
        self.assertEqual(result["Hit@100"], 0.5)
        self.assertEqual(result["P@1"], 0.0)
        self.assertEqual(result["MRR@100"], 0.25)
        with self.assertRaisesRegex(ValueError, "Every query"):
            dense.ranking_metrics({"a": [1]}, {"a": {1}, "b": {8}})

    def test_full_pool_exclusions_and_stable_id_ties_do_not_depend_on_gold(self):
        split = SimpleNamespace(doc_ids=[1, 2, 3, 4], documents=torch.ones(4, 2),
                                query_ids=["q"], queries=torch.ones(1, 2),
                                excluded={"q": {1, 2}}, targets={"q": {4}})
        metrics, rankings = dense.evaluate_split(dense.Projection(2), split, torch.device("cpu"))
        self.assertEqual(rankings, {"q": [3, 4]})
        self.assertEqual(metrics["P@1"], 0.0)
        self.assertEqual(metrics["recall@100"], 1.0)
        split.targets = {"q": {3}}
        _, new_rankings = dense.evaluate_split(dense.Projection(2), split, torch.device("cpu"))
        self.assertEqual(rankings, new_rankings)

    def test_checkpoint_selection_rejects_test_metrics(self):
        self.assertTrue(dense.select_validation_checkpoint({"split": "val", "recall@100": 0.2}, 0.1))
        self.assertFalse(dense.select_validation_checkpoint({"split": "val", "recall@100": 0.2}, 0.2))
        with self.assertRaisesRegex(ValueError, "validation metrics only"):
            dense.select_validation_checkpoint({"split": "test", "recall@100": 1.0}, 0.1)

    def test_checkpoint_snapshot_keeps_optimizer_attached_under_overwrite_conversion(self):
        embedding_dir = self.encode()
        projection_class, optimizer_class = dense.Projection, torch.optim.AdamW
        captured = {}

        def make_projection(dimension):
            captured["model"] = projection_class(dimension)
            return captured["model"]

        def make_optimizer(parameters, *args, **kwargs):
            captured["optimizer"] = optimizer_class(parameters, *args, **kwargs)
            captured["initial_weight"] = captured["model"].linear.weight.detach().clone()
            return captured["optimizer"]

        arguments = SimpleNamespace(data_dir=self.directory, embedding_dir=embedding_dir,
                                    output_dir=self.directory / "run_conversion_guard",
                                    epochs=2, batch_size=1, lr=0.01, temperature=0.05, seed=7,
                                    weight_decay=0.0, device="cpu", eval_batch_size=2)
        previous_overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
        previous_swap = torch.__future__.get_swap_module_params_on_conversion()
        try:
            # This supported mode replaces Parameter objects even on CPU-only
            # conversions, exposing optimizer detachment without requiring CUDA.
            torch.__future__.set_overwrite_module_params_on_conversion(True)
            torch.__future__.set_swap_module_params_on_conversion(False)
            with patch.object(dense, "Projection", side_effect=make_projection), \
                    patch.object(torch.optim, "AdamW", side_effect=make_optimizer), \
                    contextlib.redirect_stdout(io.StringIO()):
                dense.train(arguments)
            live_parameters = tuple(captured["model"].parameters())
            optimized_parameters = tuple(parameter for group in captured["optimizer"].param_groups
                                         for parameter in group["params"])
            self.assertEqual(len(live_parameters), len(optimized_parameters))
            for live, optimized in zip(live_parameters, optimized_parameters):
                self.assertIs(live, optimized)
            self.assertFalse(torch.equal(captured["initial_weight"],
                                         captured["model"].linear.weight.detach()))
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(previous_overwrite)
            torch.__future__.set_swap_module_params_on_conversion(previous_swap)

    def test_complete_cpu_encode_train_evaluate_and_encoder_stays_frozen(self):
        embedding_dir = self.encode()
        store = dense.EmbeddingStore(embedding_dir, self.directory)
        self.assertEqual(store.vectors.shape, (len(self.fixture.tokens), 4))
        self.assertIsInstance(store.vectors, np.memmap)
        split = dense.SplitVectors(self.directory, "train", store)
        # Company 1 retains twenty separately encoded patent vectors.
        refs = self.fixture.histories[1]["patents"]
        expected = np.stack([store.vectors[store.lookup[ref["application_number"]][0]] for ref in refs]).mean(0)
        np.testing.assert_allclose(split.documents[split.position[1]].numpy(), expected)
        model_features = (split.documents.clone(), split.queries.clone())
        self.fixture.queries[0]["label_evidence"] = [{"date": "2999-01-01", "SECRET": "not a feature"}]
        self.fixture.write_metadata()
        self.refresh_manifest()
        altered = dense.SplitVectors(self.directory, "train", store)
        torch.testing.assert_close(model_features[0], altered.documents)
        torch.testing.assert_close(model_features[1], altered.queries)
        output = self.directory / "run"
        arguments = SimpleNamespace(data_dir=self.directory, embedding_dir=embedding_dir, output_dir=output,
                                    epochs=1, batch_size=2, lr=0.001, temperature=0.05, seed=7,
                                    weight_decay=0.01, device="cpu", eval_batch_size=2)
        with patch.object(dense.Projection, "cpu", side_effect=AssertionError("Checkpoint must not move the live training model")), \
                contextlib.redirect_stdout(io.StringIO()):
            dense.train(arguments)
        configuration = dense.read_json(output / "run_config.json")
        self.assertEqual(configuration["selection_split"], "val")
        self.assertNotIn("test", configuration["source_fingerprints"])
        self.assertFalse(configuration["test_accessed"])
        self.assertTrue(configuration["encoder_frozen"])
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        self.assertEqual(set(checkpoint["state_dict"]), {"linear.weight"})
        # Deliberately load the epoch-zero control to check truthful reporting.
        checkpoint["selected_epoch"] = 0
        checkpoint["state_dict"]["linear.weight"] = torch.eye(4)
        torch.save(checkpoint, output / "epoch_zero.pt")
        report = self.directory / "test_metrics.json"
        with contextlib.redirect_stdout(io.StringIO()):
            dense.evaluate(SimpleNamespace(data_dir=self.directory, embedding_dir=embedding_dir,
                           checkpoint=output / "epoch_zero.pt", split="test", output=report, device="cpu", batch_size=2))
        self.assertEqual(dense.read_json(report)["query_count"], 2)
        self.assertEqual(dense.read_json(report)["candidate_count"], 4)
        self.assertFalse(dense.read_json(report)["innovation_model"])
        self.assertEqual(dense.read_json(report)["selected_epoch"], 0)
        self.assertFalse(dense.read_json(report)["trained_projection"])
        self.assertTrue(dense.read_json(report)["checkpoint_loaded"])
        canonical_manifest = dense.read_json(self.directory / "dataset_manifest.json")
        canonical_manifest["test_revision"] = "different_cohort"
        dense.write_json(self.directory / "dataset_manifest.json", canonical_manifest)
        with self.assertRaisesRegex(ValueError, "canonical dataset manifest mismatch"):
            dense.evaluate(SimpleNamespace(data_dir=self.directory, embedding_dir=embedding_dir,
                           checkpoint=output / "best.pt", split="test", output=self.directory / "wrong_metrics.json",
                           device="cpu", batch_size=2))
        with self.assertRaisesRegex(ValueError, "existing training"):
            dense.train(arguments)

    def test_smoke_cache_and_stale_artifacts_are_refused(self):
        smoke = self.encode("smoke", 3)
        self.assertTrue(dense.read_json(smoke / "embedding_manifest.json")["smoke_only"])
        with self.assertRaisesRegex(ValueError, "Smoke embedding cache"):
            dense.EmbeddingStore(smoke, self.directory)
        full = self.encode()
        with (full / "index.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
            dense.EmbeddingStore(full, self.directory)

    def test_wrong_tokenizer_or_changed_canonical_source_refused(self):
        manifest, _ = dense.source_signature(self.directory)
        (self.model_dir / "tokenizer.json").write_text('{"wrong":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Tokenizer hash"):
            dense.model_signature(self.model_dir, manifest)
        full = self.encode_after_restoring_tokenizer()
        token_manifest = dense.read_json(self.directory / "tokenization_manifest.json")
        token_manifest["revision"] = "changed"
        dense.write_json(self.directory / "tokenization_manifest.json", token_manifest)
        with self.assertRaisesRegex(ValueError, "source fingerprint"):
            dense.EmbeddingStore(full, self.directory)

    def encode_after_restoring_tokenizer(self):
        (self.model_dir / "tokenizer.json").write_text('{}', encoding="utf-8")
        return self.encode()


if __name__ == "__main__":
    unittest.main()
