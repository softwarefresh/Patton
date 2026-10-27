"""Independent endpoint, provenance and no-training checks for residual scales."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
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
import partner_teacher_calibration as calibration

SPEC = importlib.util.spec_from_file_location(
    "calibration_dense_fixture", ROOT / "tests/test_partner_dense_baseline.py")
fixture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_module)

METRICS = ("recall@20", "recall@100", "Hit@100", "P@1", "MRR@100")


class ResidualScaleScoringTests(unittest.TestCase):
    def test_original_base_uses_whole_documents_and_default_32_query_batches(self):
        generator = torch.Generator().manual_seed(904)
        dimension, query_count, candidate_count = 4, 65, 109
        split = SimpleNamespace(
            split="val", doc_ids=list(range(candidate_count)),
            query_ids=["q{:03d}".format(index) for index in range(query_count)],
            documents=torch.randn(candidate_count, dimension, generator=generator),
            queries=torch.randn(query_count, dimension, generator=generator))
        weight = torch.eye(dimension) + torch.randn(dimension, dimension, generator=generator) * 0.05
        before = (split.documents.clone(), split.queries.clone(), weight.clone())
        calls = []
        original_forward = dense.Projection.forward

        def traced_forward(model, values):
            calls.append(tuple(values.shape))
            return original_forward(model, values)

        with patch.object(dense.Projection, "forward", traced_forward):
            reference = calibration.original_base_scores(split, weight, torch.device("cpu"))
        self.assertEqual(calls, [(109, 4), (32, 4), (32, 4), (1, 4)])
        self.assertIsInstance(reference, np.ndarray)
        self.assertEqual(reference.shape, (query_count, candidate_count))
        self.assertEqual(reference.dtype, np.float32)
        independent_model = dense.Projection(dimension)
        independent_model.load_state_dict({"linear.weight": weight})
        with torch.inference_mode():
            documents = independent_model(split.documents)
            independent = np.concatenate([
                (independent_model(split.queries[start:start + 32]) @ documents.T).numpy()
                for start in range(0, query_count, 32)], axis=0)
        np.testing.assert_array_equal(reference, independent)
        for original, current in zip(before, (split.documents, split.queries, weight)):
            torch.testing.assert_close(original, current, rtol=0, atol=0)

    def test_supplied_dense_reference_preserves_near_ties_and_is_not_recomputed(self):
        # This deliberately injected CPU perturbation demonstrates why the
        # projection/GEMM batch path matters. It does not reproduce or verify
        # the user's real CUDA rounding discrepancy.
        query_count, candidate_count = 65, 109
        documents = torch.tensor([[-1., 0.]]).repeat(candidate_count, 1)
        documents[0] = torch.tensor([0.8, 0.6])
        documents[1] = torch.tensor([0.8, -0.6])
        qids = ["q{:03d}".format(index) for index in range(query_count)]
        split = SimpleNamespace(
            split="val", doc_ids=list(range(candidate_count)), query_ids=qids,
            documents=documents, queries=torch.tensor([[1., 0.]]).repeat(query_count, 1),
            evidence=documents.unsqueeze(1).clone(), mask=torch.ones(candidate_count, 1, dtype=torch.bool),
            targets={qid: {1} for qid in qids}, excluded={qid: set() for qid in qids})
        weight = torch.eye(2)
        original_forward = dense.Projection.forward

        def batch_sensitive_forward(model, values):
            result = original_forward(model, values)
            if values.shape[0] <= query_count:
                result = result.clone()
                # Query chunks of 32/1 get one rounding direction; the former
                # all-queries-at-once path receives the opposite direction.
                result[:, 1] += 1.e-7 if values.shape[0] <= 32 else -1.e-7
                result = torch.nn.functional.normalize(result, dim=-1)
            return result

        with patch.object(dense.Projection, "forward", batch_sensitive_forward):
            reference = calibration.original_base_scores(split, weight, torch.device("cpu"))
            base_model = dense.Projection(2)
            base_model.load_state_dict({"linear.weight": weight})
            expected, _ = dense.evaluate_split(base_model, split, torch.device("cpu"))
            with torch.inference_mode():
                all_query_scores = (base_model(split.queries) @ base_model(split.documents).T).numpy()
        self.assertTrue((reference[:, 0] > reference[:, 1]).all())
        self.assertTrue((all_query_scores[:, 1] > all_query_scores[:, 0]).all())
        self.assertLess(float(np.max(np.abs(reference - all_query_scores))), 1.e-6)
        wrong_rankings = {
            qid: [split.doc_ids[index] for index in np.argsort(-all_query_scores[position], kind="stable")[:100]]
            for position, qid in enumerate(qids)}
        wrong = dense.ranking_metrics(wrong_rankings, split.targets)
        self.assertEqual(expected["MRR@100"], 0.5)
        self.assertEqual(wrong["MRR@100"], 1.)
        model = EvidenceInteractionTeacher(weight, hidden_dim=3, mlp_hidden=4)
        last_metrics, _ = teacher.evaluate_split(model, split, torch.device("cpu"), 8, 13)
        feature_before = tuple(value.clone() for value in
                               (split.documents, split.queries, split.evidence, split.mask, model.base_weight))
        reference_before = reference.copy()
        reference.flags.writeable = False
        with patch.object(calibration, "original_base_scores", side_effect=AssertionError("Reference recomputed")):
            reports, _ = calibration.evaluate_scales(
                model, split, torch.device("cpu"), query_batch_size=8, candidate_batch_size=13,
                base_scores_reference=reference)
        for metric in METRICS:
            self.assertEqual(reports["0.0"][metric], expected[metric])
            self.assertEqual(reports["1.0"][metric], last_metrics[metric])
        np.testing.assert_array_equal(reference, reference_before)
        for original, current in zip(feature_before,
                (split.documents, split.queries, split.evidence, split.mask, model.base_weight)):
            torch.testing.assert_close(original, current, rtol=0, atol=0)
        # The exact observed discrepancy must still fail the unchanged strict
        # endpoint control, rather than being hidden by a looser tolerance.
        recorded = dict(reports["0.0"], **{"MRR@100": 0.029681499497425247})
        changed = dict(recorded, **{"MRR@100": 0.029681688489332142})
        with self.assertRaisesRegex(ValueError, "control_mismatch"):
            calibration._check_controls(changed, recorded, "observed CUDA MRR discrepancy")

    def test_zero_and_one_endpoints_match_dense_and_teacher_with_chunking(self):
        generator = torch.Generator().manual_seed(481)
        evidence = torch.randn(109, 3, 4, generator=generator)
        mask = torch.ones(109, 3, dtype=torch.bool)
        mask[::3, 2] = False
        documents = evidence.masked_fill(~mask.unsqueeze(-1), 0.).sum(1) / mask.sum(1, keepdim=True)
        split = SimpleNamespace(
            doc_ids=list(range(109)), documents=documents, evidence=evidence, mask=mask,
            query_ids=["a", "b", "c"], queries=torch.randn(3, 4, generator=generator),
            excluded={"a": {0, 2}, "b": {1}, "c": {3, 9}},
            targets={"a": {6, 8}, "b": {13}, "c": {90}})
        weight = torch.eye(4) + torch.randn(4, 4, generator=generator) * 0.08
        base = dense.Projection(4)
        base.load_state_dict({"linear.weight": weight})
        torch.manual_seed(91)
        model = EvidenceInteractionTeacher(weight, hidden_dim=4, mlp_hidden=5)
        with torch.no_grad():
            model.mlp[-1].weight.copy_(torch.randn(1, 5, generator=generator) * 0.8)
            model.mlp[-1].bias.fill_(0.1)
        base_metrics, _ = dense.evaluate_split(base, split, torch.device("cpu"), 2)
        teacher_metrics, _ = teacher.evaluate_split(model, split, torch.device("cpu"), 2, 13)
        small, stats = calibration.evaluate_scales(model, split, torch.device("cpu"), 1, 13)
        whole, whole_stats = calibration.evaluate_scales(model, split, torch.device("cpu"), 3, 109)
        self.assertEqual(calibration.ALPHAS, (0., 0.01, 0.03, 0.1, 0.3, 1.))
        self.assertEqual(set(small), {str(alpha) for alpha in calibration.ALPHAS})
        self.assertTrue(stats)
        self.assertEqual(set(stats), set(whole_stats))
        with torch.inference_mode():
            base_scores = (base(split.queries) @ base(split.documents).T).numpy()
            prepared_q = model.prepare_queries(split.queries)
            prepared_c = model.prepare_candidates(split.evidence, split.mask, raw_means=split.documents)
            full_scores = model.score_prepared(prepared_q, prepared_c).numpy()
        # Check all interior scales against independent interpolation and
        # ranking, rather than only accepting the two familiar endpoints.
        for alpha in calibration.ALPHAS:
            scores = base_scores + alpha * (full_scores - base_scores)
            rankings = {}
            for index, qid in enumerate(split.query_ids):
                allowed = np.array([position for position, doc_id in enumerate(split.doc_ids)
                                    if doc_id not in split.excluded[qid]])
                order = allowed[np.argsort(-scores[index, allowed], kind="stable")[:100]]
                rankings[qid] = [split.doc_ids[position] for position in order]
            expected = dense.ranking_metrics(rankings, split.targets)
            for metric in METRICS:
                self.assertAlmostEqual(small[str(alpha)][metric], expected[metric])
        for metric in METRICS:
            self.assertAlmostEqual(small["0.0"][metric], base_metrics[metric])
            self.assertAlmostEqual(small["1.0"][metric], teacher_metrics[metric])
            for alpha in small:
                self.assertAlmostEqual(small[alpha][metric], whole[alpha][metric])
        for report in small.values():
            self.assertEqual(report["split"], "val")
            self.assertEqual(report["query_count"], 3)
            self.assertEqual(report["candidate_count"], 109)
            self.assertEqual(report["positive_count"], 4)

    def test_ties_exclusions_and_gold_do_not_change_residual_statistics(self):
        split = SimpleNamespace(
            doc_ids=[1, 2, 3, 4], documents=torch.ones(4, 4), evidence=torch.ones(4, 3, 4),
            mask=torch.ones(4, 3, dtype=torch.bool), query_ids=["miss", "q"], queries=torch.ones(2, 4),
            excluded={"q": {1, 2}, "miss": set()}, targets={"q": {4}, "miss": {999}})
        torch.manual_seed(11)
        model = EvidenceInteractionTeacher(torch.eye(4), hidden_dim=4, mlp_hidden=5)
        with torch.no_grad():
            model.mlp[-1].bias.fill_(0.4)
        before = tuple(value.clone() for value in (split.documents, split.evidence, split.queries, split.mask))
        initial, statistics = calibration.evaluate_scales(model, split, torch.device("cpu"), 1, 1)
        for report in initial.values():
            # Stable ascending IDs produce [3,4] for q and [1,2,3,4]
            # for miss; neither excluded q candidate may enter the ranking.
            self.assertEqual(report["recall@100"], 0.5)
            self.assertEqual(report["P@1"], 0.)
            self.assertEqual(report["MRR@100"], 0.25)
        split.targets["q"] = {3}
        changed, changed_statistics = calibration.evaluate_scales(model, split, torch.device("cpu"), 1, 1)
        self.assertEqual(statistics, changed_statistics)
        for report in changed.values():
            self.assertEqual(report["recall@100"], 0.5)
            self.assertEqual(report["P@1"], 0.5)
            self.assertEqual(report["MRR@100"], 0.5)
        for original, current in zip(before, (split.documents, split.evidence, split.queries, split.mask)):
            torch.testing.assert_close(original, current, rtol=0, atol=0)


class ResidualCalibrationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.DenseBaselineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.directory = self.fixture.directory
        self.embedding_dir = self.fixture.encode()
        baseline_args = SimpleNamespace(
            data_dir=self.directory, embedding_dir=self.embedding_dir,
            output_dir=self.directory / "baseline", epochs=1, batch_size=2,
            lr=0.001, temperature=0.05, seed=7, weight_decay=0.01, device="cpu", eval_batch_size=2)
        with contextlib.redirect_stdout(io.StringIO()):
            dense.train(baseline_args)
        self.base_checkpoint = baseline_args.output_dir / "best.pt"
        self.run_dir = self.directory / "teacher_canonical"
        arguments = SimpleNamespace(
            data_dir=self.directory, embedding_dir=self.embedding_dir,
            base_checkpoint=self.base_checkpoint, output_dir=self.run_dir,
            epochs=1, batch_size=2, eval_batch_size=1, candidate_batch_size=2,
            lr=0.001, weight_decay=0.01, temperature=0.05, seed=7, device="cpu",
            hidden_dim=4, mlp_hidden=4, attention_temperature=0.2, residual_scale=0.25,
            candidate_mode="canonical")
        with contextlib.redirect_stdout(io.StringIO()):
            teacher.train_pair(arguments)
        self.output_temp = tempfile.TemporaryDirectory(prefix="teacher_calibration_test_", dir=str(ROOT / "tmp"))
        self.addCleanup(self.output_temp.cleanup)
        self.output_dir = Path(self.output_temp.name)

    def args(self, name="calibration.json"):
        return SimpleNamespace(
            data_dir=self.directory, embedding_dir=self.embedding_dir, run_dir=self.run_dir,
            base_checkpoint=self.base_checkpoint, output=self.output_dir / name,
            device="cpu", batch_size=1, candidate_batch_size=2)

    def test_run_uses_fixed_grid_without_training_test_access_or_mutating_inputs(self):
        args = self.args()
        protected = [path for path in self.directory.rglob("*") if path.is_file()]
        before = {str(path): dense.fingerprint(path) for path in protected}
        prohibited = {stem + ".test.jsonl" for stem in ("documents", "histories", "queries")}
        prohibited.add("test.pairs.jsonl")
        original_open = Path.open

        def guarded_open(path, *arguments, **keywords):
            if path.name in prohibited:
                raise AssertionError("Calibration must not open the test cohort")
            return original_open(path, *arguments, **keywords)

        with patch.object(torch.optim, "AdamW", side_effect=AssertionError("No calibration optimizer")), \
                patch.object(dense, "train", side_effect=AssertionError("No baseline retraining")), \
                patch.object(teacher, "train_pair", side_effect=AssertionError("No teacher retraining")), \
                patch.object(calibration, "original_base_scores", wraps=calibration.original_base_scores) as reference_builder, \
                patch.object(Path, "open", guarded_open), contextlib.redirect_stdout(io.StringIO()):
            calibration.run(args)
        self.assertEqual(reference_builder.call_count, 1)
        self.assertEqual(before, {name: dense.fingerprint(Path(name)) for name in before})
        report = dense.read_json(args.output)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["format"], "partner_teacher_residual_calibration_v1")
        self.assertEqual(report["split"], "val")
        self.assertTrue(report["validation_exploratory"])
        self.assertTrue(report["no_training"])
        self.assertFalse(report["test_accessed"])
        self.assertEqual(report["fixed_grid"], list(calibration.ALPHAS))
        self.assertEqual(set(report["variants"]), {"mean", "attention"})
        store = dense.EmbeddingStore(self.embedding_dir, self.directory)
        split = teacher.TeacherSplit(self.directory, "val", store)
        for mode in ("mean", "attention"):
            saved = torch.load(self.run_dir / mode / "last.pt", weights_only=True)
            config = saved["configuration"]
            model = EvidenceInteractionTeacher(saved["state_dict"]["base_weight"],
                hidden_dim=config["hidden_dim"], mlp_hidden=config["mlp_hidden"], mode=mode,
                attention_temperature=config["attention_temperature"], residual_scale=config["residual_scale"])
            model.load_state_dict(saved["state_dict"])
            independent, _ = teacher.evaluate_split(model, split, torch.device("cpu"), 1, 2)
            scales = report["variants"][mode]["scales"]
            for metric in METRICS:
                self.assertAlmostEqual(scales["1.0"][metric], independent[metric])
            self.assertIn(report["variants"][mode]["best_validation_alpha"], calibration.ALPHAS)

    def refresh_last_signature(self, mode):
        signature = dense.fingerprint(self.run_dir / mode / "last.pt")
        summary_path = self.run_dir / mode / "training_summary.json"
        summary = dense.read_json(summary_path)
        summary["last_checkpoint"] = signature
        dense.write_json(summary_path, summary)
        comparison_path = self.run_dir / "comparison.json"
        comparison = dense.read_json(comparison_path)
        comparison["variants"][mode]["last_checkpoint"] = signature
        dense.write_json(comparison_path, comparison)

    def test_bound_candidate_configuration_last_role_and_base_weights_are_verified(self):
        mode = "mean"
        last_path = self.run_dir / mode / "last.pt"
        summary_path = self.run_dir / mode / "training_summary.json"
        comparison_path = self.run_dir / "comparison.json"
        original = torch.load(last_path, weights_only=True)
        originals = {path: path.read_bytes() for path in (last_path, summary_path, comparison_path)}
        mutations = []
        modified = copy.deepcopy(original)
        modified["configuration"]["derived_candidate_fingerprint"] = {"bytes": 1, "sha256": "f" * 64}
        mutations.append(("candidate_hash", modified))
        modified = copy.deepcopy(original)
        modified["checkpoint_role"] = "validation_best"
        mutations.append(("last_role", modified))
        modified = copy.deepcopy(original)
        modified["state_dict"]["base_weight"][0, 0] += 0.25
        mutations.append(("base_weights", modified))
        for name, modified in mutations:
            with self.subTest(name=name):
                try:
                    torch.save(modified, last_path)
                    # Keep recorded file fingerprints consistent to exercise
                    # semantic binding rather than just stale-checksum refusal.
                    self.refresh_last_signature(mode)
                    args = self.args("reject_" + name + ".json")
                    with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                        calibration.run(args)
                    self.assertFalse(args.output.exists())
                finally:
                    for path, content in originals.items():
                        path.write_bytes(content)

    def test_existing_report_and_output_inside_canonical_data_are_refused(self):
        args = self.args("keep.json")
        args.output.write_text("keep", encoding="utf-8")
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            calibration.run(args)
        self.assertEqual(args.output.read_text(encoding="utf-8"), "keep")
        args = self.args()
        args.output = self.directory / "calibration_must_not_be_written.json"
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            calibration.run(args)
        self.assertFalse(args.output.exists())

    def test_dense_derived_candidate_file_tampering_is_refused(self):
        # The public run integration uses a canonical miniature experiment.
        # Independently exercise the dense-hard file binding without fabricating
        # a second training run or weakening its recorded-source checks.
        configuration = dense.read_json(self.run_dir / "run_config.json")
        candidate_path = self.run_dir / "training_candidates.jsonl"
        candidate_path.write_text('{"query_id":"query_0","candidate_ids":[1,2,3]}\n', encoding="utf-8")
        signature = dense.fingerprint(candidate_path)
        configuration["candidate_mode"] = "dense-hard"
        configuration["derived_candidate_fingerprint"] = signature
        base_signature = dense.fingerprint(self.base_checkpoint)
        store = dense.EmbeddingStore(self.embedding_dir, self.directory)
        audit = {
            "split": "train", "val_accessed": False, "test_accessed": False,
            "canonical_pairs_sha256": configuration["source_fingerprints"]["train"]["train.pairs.jsonl"],
            "derived_candidate_fingerprint": signature, "base_checkpoint": base_signature,
            "embedding_manifest_sha256": store.manifest_sha256,
            "source_fingerprints": configuration["source_fingerprints"]["train"],
        }
        dense.write_json(self.run_dir / "candidate_audit.json", audit)
        comparison = {"candidate_audit": audit}
        self.assertEqual(calibration._validate_candidates(
            configuration, comparison, self.run_dir, base_signature, store), signature)
        with candidate_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "candidate fingerprint"):
            calibration._validate_candidates(configuration, comparison, self.run_dir, base_signature, store)

    def test_invalid_batch_sizes_are_rejected_without_creating_output(self):
        for field in ("batch_size", "candidate_batch_size"):
            with self.subTest(field=field):
                args = self.args("invalid_" + field + ".json")
                setattr(args, field, 0)
                with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                    calibration.run(args)
                self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
