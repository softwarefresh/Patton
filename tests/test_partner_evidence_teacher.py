"""CPU integration and provenance checks for paired evidence-teacher training."""
import contextlib
import copy
import importlib.util
import io
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
import partner_evidence_teacher as teacher
from partner_evidence_teacher_model import EvidenceInteractionTeacher

SPEC = importlib.util.spec_from_file_location(
    "teacher_dense_fixture", ROOT / "tests/test_partner_dense_baseline.py")
fixture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_module)


class EvidenceTeacherRankingTests(unittest.TestCase):
    def split(self):
        # All candidates tie. Gold is deliberately absent from the model inputs.
        return SimpleNamespace(
            doc_ids=[1, 2, 3, 4], documents=torch.ones(4, 4),
            evidence=torch.ones(4, 3, 4), mask=torch.ones(4, 3, dtype=torch.bool),
            query_ids=["q", "miss"], queries=torch.ones(2, 4),
            excluded={"q": {1, 2}, "miss": set()},
            targets={"q": {4}, "miss": {999}})

    def test_full_pool_chunking_preserves_exclusions_ties_and_gold_independence(self):
        split = self.split()
        model = EvidenceInteractionTeacher(torch.eye(4), hidden_dim=4, mlp_hidden=4)
        inputs = (split.documents.clone(), split.queries.clone(), split.evidence.clone(), split.mask.clone())
        small_metrics, small_ranks = teacher.evaluate_split(
            model, split, torch.device("cpu"), query_batch_size=1, candidate_batch_size=1)
        large_metrics, large_ranks = teacher.evaluate_split(
            model, split, torch.device("cpu"), query_batch_size=5, candidate_batch_size=7)
        self.assertEqual(small_ranks, {"q": [3, 4], "miss": [1, 2, 3, 4]})
        self.assertEqual(small_ranks, large_ranks)
        for metric in ("recall@20", "recall@100", "Hit@100", "P@1", "MRR@100"):
            self.assertAlmostEqual(small_metrics[metric], large_metrics[metric])
        self.assertEqual(small_metrics["recall@100"], 0.5)
        self.assertEqual(small_metrics["P@1"], 0.)
        split.targets["q"] = {3}
        _, changed_gold_ranks = teacher.evaluate_split(
            model, split, torch.device("cpu"), query_batch_size=2, candidate_batch_size=2)
        self.assertEqual(small_ranks, changed_gold_ranks)
        for original, current in zip(inputs, (split.documents, split.queries, split.evidence, split.mask)):
            torch.testing.assert_close(original, current)

    def test_nontrivial_scores_and_top100_are_invariant_to_two_dimensional_chunking(self):
        generator = torch.Generator().manual_seed(71)
        evidence = torch.randn(109, 3, 4, generator=generator)
        split = SimpleNamespace(
            doc_ids=list(range(109)), evidence=evidence, documents=evidence.mean(1),
            mask=torch.ones(109, 3, dtype=torch.bool),
            query_ids=["a", "b", "c"], queries=torch.randn(3, 4, generator=generator),
            excluded={"a": {0, 2}, "b": {1}, "c": {3, 9}},
            targets={"a": {6, 8}, "b": {13}, "c": {90}})
        torch.manual_seed(73)
        model = EvidenceInteractionTeacher(torch.eye(4), hidden_dim=4, mlp_hidden=4)
        # A trained, nonzero interaction residual must also work in blocks.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.03)
        small, small_ranks = teacher.evaluate_split(
            model, split, torch.device("cpu"), query_batch_size=1, candidate_batch_size=13)
        full, full_ranks = teacher.evaluate_split(
            model, split, torch.device("cpu"), query_batch_size=3, candidate_batch_size=109)
        self.assertEqual(small_ranks, full_ranks)
        self.assertTrue(all(len(ranking) == 100 for ranking in small_ranks.values()))
        for qid, ranking in small_ranks.items():
            self.assertFalse(set(ranking) & split.excluded[qid])
            self.assertEqual(len(ranking), len(set(ranking)))
        for metric in ("recall@20", "recall@100", "Hit@100", "P@1", "MRR@100"):
            self.assertAlmostEqual(small[metric], full[metric])


class EvidenceTeacherPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.DenseBaselineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.directory = self.fixture.directory
        self.embedding_dir = self.fixture.encode()
        self.base_dir = self.directory / "dense_baseline"
        arguments = SimpleNamespace(
            data_dir=self.directory, embedding_dir=self.embedding_dir, output_dir=self.base_dir,
            epochs=1, batch_size=2, lr=0.001, temperature=0.05, seed=7,
            weight_decay=0.01, device="cpu", eval_batch_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            dense.train(arguments)
        self.base_checkpoint = self.base_dir / "best.pt"

    def args(self, name="teacher_pair"):
        return SimpleNamespace(
            data_dir=self.directory, embedding_dir=self.embedding_dir,
            base_checkpoint=self.base_checkpoint, output_dir=self.directory / name,
            epochs=1, batch_size=2, eval_batch_size=1, candidate_batch_size=2,
            lr=0.001, weight_decay=0.01, temperature=0.05, seed=7, device="cpu",
            hidden_dim=4, mlp_hidden=4, attention_temperature=0.2, residual_scale=0.25)

    def splits(self):
        store = dense.EmbeddingStore(self.embedding_dir, self.directory)
        return (store, teacher.TeacherSplit(self.directory, "train", store),
                teacher.TeacherSplit(self.directory, "val", store))

    def test_teacher_split_reuses_same_query_and_mean_features_and_real_evidence_slots(self):
        store, training, validation = self.splits()
        expected = dense.SplitVectors(self.directory, "train", store)
        torch.testing.assert_close(training.queries, expected.queries)
        torch.testing.assert_close(training.documents, expected.documents)
        self.assertEqual(training.doc_ids, expected.doc_ids)
        self.assertEqual(training.query_ids, expected.query_ids)
        self.assertEqual(training.pairs, expected.pairs)
        self.assertEqual(training.targets, expected.targets)
        self.assertEqual(training.excluded, expected.excluded)
        self.assertEqual(training.evidence.shape, (4, 20, 4))
        self.assertEqual(training.mask.sum().item(), 25)
        company_position = training.position[1]
        refs = self.fixture.fixture.histories[1]["patents"]
        expected_patents = np.stack([store.vectors[store.lookup[ref["application_number"]][0]] for ref in refs])
        np.testing.assert_array_equal(training.evidence[company_position].numpy(), expected_patents)
        self.assertTrue(training.mask[company_position].all())
        self.assertEqual(validation.fingerprints, dense.SplitVectors(self.directory, "val", store).fingerprints)

    def test_base_checkpoint_rejects_wrong_source_split_cache_format_and_nonfinite_weights(self):
        store, training, validation = self.splits()
        base_weight, signature, saved = teacher.load_base_checkpoint(
            self.base_checkpoint, store, training, validation)
        self.assertEqual(base_weight.shape, (4, 4))
        torch.testing.assert_close(base_weight, saved["state_dict"]["linear.weight"])
        self.assertEqual(signature, dense.fingerprint(self.base_checkpoint))
        changes = []
        for split, filename in (("train", "queries.train.jsonl"), ("val", "histories.val.jsonl")):
            modified = copy.deepcopy(saved)
            modified["configuration"]["source_fingerprints"][split][filename] = "f" * 64
            changes.append(("source_" + split, modified))
        modified = copy.deepcopy(saved)
        modified["configuration"]["embedding_manifest_sha256"] = "e" * 64
        changes.append(("cache", modified))
        modified = copy.deepcopy(saved)
        modified["format"] = "a_different_model"
        changes.append(("format", modified))
        modified = copy.deepcopy(saved)
        modified["selection_split"] = "test"
        changes.append(("test_selected", modified))
        modified = copy.deepcopy(saved)
        modified["state_dict"]["linear.weight"][0, 0] = float("nan")
        changes.append(("nonfinite", modified))
        modified = copy.deepcopy(saved)
        modified["state_dict"]["linear.weight"] = torch.eye(3)
        changes.append(("wrong_dimension", modified))
        for name, modified in changes:
            with self.subTest(name=name):
                checkpoint = self.directory / ("bad_" + name + ".pt")
                torch.save(modified, checkpoint)
                with self.assertRaises(ValueError):
                    teacher.load_base_checkpoint(checkpoint, store, training, validation)

    def test_smoke_embedding_cache_is_rejected_by_paired_training(self):
        arguments = self.args("reject_smoke")
        arguments.embedding_dir = self.fixture.encode("smoke", 3)
        with self.assertRaisesRegex(ValueError, "Smoke embedding cache"):
            with contextlib.redirect_stdout(io.StringIO()):
                teacher.train_pair(arguments)

    def test_cpu_paired_training_keeps_base_cache_frozen_and_does_not_open_test_split(self):
        args = self.args()
        before = {str(path): dense.fingerprint(path) for path in (
            self.base_checkpoint, self.embedding_dir / "embedding_manifest.json",
            self.embedding_dir / "index.jsonl", self.embedding_dir / "vectors.npy")}
        base_weight = torch.load(self.base_checkpoint, weights_only=True)["state_dict"]["linear.weight"].clone()
        # The manifest remains canonical; an attempt to load the test cohort
        # fails because its four split files no longer exist.
        for stem in ("documents", "histories", "queries"):
            (self.directory / (stem + ".test.jsonl")).unlink()
        (self.directory / "test.pairs.jsonl").unlink()
        models, optimizers, initial_weights = [], [], []
        model_class = teacher.EvidenceInteractionTeacher
        optimizer_class = torch.optim.AdamW

        def make_model(*arguments, **keywords):
            model = model_class(*arguments, **keywords)
            models.append(model)
            initial_weights.append({name: value.detach().clone() for name, value in model.state_dict().items()})
            return model

        def make_optimizer(parameters, *arguments, **keywords):
            optimizer = optimizer_class(parameters, *arguments, **keywords)
            optimizers.append(optimizer)
            return optimizer

        previous_overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
        previous_swap = torch.__future__.get_swap_module_params_on_conversion()
        try:
            torch.__future__.set_overwrite_module_params_on_conversion(True)
            torch.__future__.set_swap_module_params_on_conversion(False)
            with patch.object(teacher, "EvidenceInteractionTeacher", side_effect=make_model), \
                    patch.object(torch.optim, "AdamW", side_effect=make_optimizer), \
                    patch.object(model_class, "cpu", side_effect=AssertionError("Never move a live model to save")), \
                    contextlib.redirect_stdout(io.StringIO()):
                teacher.train_pair(args)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(previous_overwrite)
            torch.__future__.set_swap_module_params_on_conversion(previous_swap)
        # The disposable one-step preflight owns its own model and optimizer.
        self.assertEqual(len(models), 3)
        self.assertEqual(len(optimizers), 3)
        for model, optimizer in zip(models, optimizers):
            self.assertFalse(model.base_weight.requires_grad)
            torch.testing.assert_close(model.base_weight, base_weight)
            self.assertNotIn("base_weight", dict(model.named_parameters()))
            live = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            self.assertEqual({id(parameter) for parameter in live}, {id(parameter) for parameter in optimized})
        for name in initial_weights[1]:
            torch.testing.assert_close(initial_weights[1][name], initial_weights[2][name], rtol=0, atol=0)
        self.assertEqual(before, {name: dense.fingerprint(Path(name)) for name in before})
        configuration = dense.read_json(args.output_dir / "run_config.json")
        self.assertEqual(configuration["selection_split"], "val")
        self.assertFalse(configuration["test_accessed"])
        self.assertTrue(configuration["encoder_frozen"])
        self.assertNotIn("test", configuration["source_fingerprints"])
        base_metrics = dense.read_json(args.output_dir / "base.val.json")
        self.assertEqual(base_metrics["split"], "val")
        self.assertEqual(base_metrics["query_count"], 2)
        self.assertEqual(base_metrics["candidate_count"], 4)
        dense.read_json(args.output_dir / "comparison.json")
        for mode in ("mean", "attention"):
            checkpoint = torch.load(args.output_dir / mode / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["selection_split"], "val")
            self.assertEqual(checkpoint["validation"]["split"], "val")
            self.assertIn(checkpoint["selected_epoch"], (0, 1))
            self.assertTrue(all(value.device.type == "cpu" for value in checkpoint["state_dict"].values()))
            history = dense.read_json(args.output_dir / mode / "history.json")
            self.assertEqual([entry["epoch"] for entry in history], [0, 1])
            self.assertEqual(history[0]["validation"]["recall@100"], base_metrics["recall@100"])
            self.assertTrue(all(entry["validation"]["split"] == "val" for entry in history))

    def test_existing_training_output_is_refused_before_overwrite(self):
        args = self.args("existing")
        args.output_dir.mkdir()
        marker = args.output_dir / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(ValueError):
            with contextlib.redirect_stdout(io.StringIO()):
                teacher.train_pair(args)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_nonpositive_batches_and_nonfinite_temperature_are_rejected_before_output(self):
        for field, value in (("eval_batch_size", 0), ("candidate_batch_size", -1),
                             ("temperature", float("nan")), ("attention_temperature", float("inf"))):
            with self.subTest(field=field):
                args = self.args("invalid_" + field)
                setattr(args, field, value)
                with self.assertRaises(ValueError):
                    with contextlib.redirect_stdout(io.StringIO()):
                        teacher.train_pair(args)
                self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
