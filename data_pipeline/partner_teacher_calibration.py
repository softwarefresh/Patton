"""Read-only full-validation residual calibration of existing diagnostic heads.

No optimization, test-cohort evaluation, or checkpoint writes are performed.
The six scales are fixed in advance. A nonzero scale selected here remains a
validation-exploratory result, not independent evidence of a strong teacher.
"""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

import partner_dense_baseline as dense
import partner_evidence_teacher as teacher
from partner_evidence_dataset import integer, require
from partner_evidence_teacher_model import EvidenceInteractionTeacher


FORMAT = "partner_teacher_residual_calibration_v1"
ALPHAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
METRICS = ("recall@20", "recall@100", "Hit@100", "P@1", "MRR@100")
COUNTS = ("query_count", "candidate_count", "positive_count")
SAMPLE_SEED = 0
SAMPLES_PER_QUERY = 128


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "min": None, "q25": None,
                "median": None, "q75": None, "max": None}
    require(bool(np.isfinite(values).all()), "Nonfinite diagnostic summary")
    quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
    return {"count": int(values.size), "mean": float(values.mean()),
            "min": float(quantiles[0]), "q25": float(quantiles[1]),
            "median": float(quantiles[2]), "q75": float(quantiles[3]), "max": float(quantiles[4])}


def evaluate_scales(model, split, device, query_batch_size=8, candidate_batch_size=512):
    """Score each pair once and compare the fixed scales on the complete pool.

    Only one query batch's full base/last score matrices live on the CPU.
    Residual means, standard deviation and maximum use every allowed pair;
    absolute quantiles use fixed-seed per-query uniform samples without gold.
    """
    require(getattr(split, "split", "val") == "val", "Calibration accepts validation only")
    require(integer(query_batch_size) and query_batch_size > 0 and
            integer(candidate_batch_size) and candidate_batch_size > 0, "Invalid scoring batch size")
    require(split.doc_ids and split.doc_ids == sorted(set(split.doc_ids)), "Candidate IDs must be sorted unique")
    require(split.query_ids and split.query_ids == sorted(set(split.query_ids)), "Query IDs must be sorted unique")
    require(set(split.query_ids) == set(split.targets) == set(split.excluded), "Query/target/exclusion IDs differ")
    require(split.queries.shape[0] == len(split.query_ids) and split.evidence.shape[0] == len(split.doc_ids) and
            split.documents.shape[0] == len(split.doc_ids), "Calibration feature counts differ")
    rankings = {str(alpha): {} for alpha in ALPHAS}
    sample_rng = np.random.default_rng(SAMPLE_SEED)
    sampled_absolute, base_boundary_scores = [], []
    count, residual_sum, square_sum, absolute_sum, absolute_max = 0, 0.0, 0.0, 0.0, 0.0
    last_log = time.monotonic()
    model.eval()
    with torch.inference_mode():
        chunks = []
        for start in range(0, len(split.doc_ids), candidate_batch_size):
            stop = start + candidate_batch_size
            chunks.append(model.prepare_candidates(split.evidence[start:stop].to(device),
                          split.mask[start:stop].to(device), split.documents[start:stop].to(device)))
        candidates = tuple(torch.cat([chunk[index] for chunk in chunks], dim=0) for index in range(3))
        del chunks
        all_queries = model.prepare_queries(split.queries.to(device))
        for start in range(0, len(split.query_ids), query_batch_size):
            qids = split.query_ids[start:start + query_batch_size]
            queries = tuple(value[start:start + len(qids)] for value in all_queries)
            base_scores = np.empty((len(qids), len(split.doc_ids)), dtype=np.float32)
            last_scores = np.empty_like(base_scores)
            for candidate_start in range(0, len(split.doc_ids), candidate_batch_size):
                stop = candidate_start + candidate_batch_size
                prepared = tuple(value[candidate_start:stop] for value in candidates)
                last = model.score_prepared(queries, prepared)
                base = queries[0] @ prepared[0].T
                base_scores[:, candidate_start:stop] = base.cpu().numpy()
                last_scores[:, candidate_start:stop] = last.cpu().numpy()
            require(bool(np.isfinite(base_scores).all()) and bool(np.isfinite(last_scores).all()),
                    "Nonfinite calibration scores")
            for offset, qid in enumerate(qids):
                allowed = np.array([position for position, doc_id in enumerate(split.doc_ids)
                                    if doc_id not in split.excluded[qid]], dtype=np.int64)
                require(len(allowed) > 0, "Calibration query has no allowed companies")
                base, last = base_scores[offset, allowed], last_scores[offset, allowed]
                residual = last - base
                doubles = residual.astype(np.float64)
                absolute = np.abs(doubles)
                count += len(residual)
                residual_sum += float(doubles.sum())
                square_sum += float(np.dot(doubles, doubles))
                absolute_sum += float(absolute.sum())
                absolute_max = max(absolute_max, float(absolute.max()))
                sample_indices = sample_rng.choice(len(allowed), size=min(SAMPLES_PER_QUERY, len(allowed)), replace=False)
                sampled_absolute.extend(float(value) for value in absolute[sample_indices])
                base_order = np.argsort(-base, kind="stable")
                boundary = float(base[base_order[min(100, len(base_order)) - 1]])
                base_boundary_scores.append(boundary)
                for alpha in ALPHAS:
                    # Preserve exact endpoint scoring paths, avoiding an extra
                    # subtract/add round trip for the recorded last checkpoint.
                    scores = base if alpha == 0 else last if alpha == 1 else base + alpha * residual
                    order = np.argsort(-scores, kind="stable")[:100]
                    rankings[str(alpha)][qid] = [split.doc_ids[allowed[index]] for index in order]
            if time.monotonic() - last_log >= 25:
                print(json.dumps({"stage": "calibration_progress", "mode": model.mode,
                      "completed_queries": start + len(qids), "total_queries": len(split.query_ids),
                      "candidate_count": len(split.doc_ids)}), flush=True)
                last_log = time.monotonic()
    reports = {}
    for alpha in ALPHAS:
        metrics = dense.ranking_metrics(rankings[str(alpha)], split.targets)
        metrics.update(split="val", query_count=len(split.query_ids), candidate_count=len(split.doc_ids),
                       positive_count=sum(len(gold) for gold in split.targets.values()),
                       evaluation_candidate_pool="all_allowed_documents", gold_used_for_scoring=False)
        reports[str(alpha)] = metrics
    require(count > 0, "No allowed residual observations")
    mean = residual_sum / count
    quantile_levels = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    quantiles = np.quantile(np.asarray(sampled_absolute), quantile_levels)
    stats = {"scope": "every_validation_query_and_every_historically_allowed_company_including_positives",
             "excluded_candidates_included": False, "count": count, "mean": mean,
             "std": math.sqrt(max(0.0, square_sum / count - mean * mean)),
             "mean_absolute": absolute_sum / count, "absolute_max": absolute_max,
             "exact_statistics": ["count", "mean", "std", "mean_absolute", "absolute_max"],
             "absolute_quantile_estimates": {str(level): float(value) for level, value in zip(quantile_levels, quantiles)},
             "quantile_sampling": {"method": "uniform_without_replacement_within_each_query",
                                   "seed": SAMPLE_SEED, "max_per_query": SAMPLES_PER_QUERY,
                                   "sample_count": len(sampled_absolute), "gold_used_for_sampling": False,
                                   "estimates_only_not_exact_full_pair_quantiles": True},
             "base_top100_boundary_score": _summary(base_boundary_scores),
             "boundary_definition": "min(100,allowed_count)th_base_score_for_each_query",
             "gold_used_for_residual_or_boundary_statistics": False}
    return reports, stats


def _check_controls(actual, expected, label):
    require(expected.get("split") == "val", "control_mismatch: " + label + " expected split")
    for name in COUNTS:
        require(name in expected and actual[name] == expected[name], "control_mismatch: " + label + " " + name)
    for name in METRICS:
        require(name in expected and math.isfinite(expected[name]) and
                math.isclose(actual[name], expected[name], rel_tol=0.0, abs_tol=1e-10),
                "control_mismatch: {} {} actual={} expected={}".format(
                    label, name, actual.get(name), expected.get(name)))


def _output_path(args):
    output = Path(args.output).resolve()
    require(not output.exists(), "Refusing existing calibration output")
    protected = (Path(args.data_dir).resolve(), Path(args.embedding_dir).resolve(),
                 Path(args.run_dir).resolve(), Path(args.base_checkpoint).resolve().parent)
    for directory in protected:
        try:
            output.relative_to(directory)
        except ValueError:
            continue
        raise ValueError("Calibration output must stay outside protected input directories")
    return output


def _validate_configuration(configuration, comparison, split, store, data_dir):
    require(configuration.get("format") == teacher.FORMAT and comparison.get("format") == teacher.FORMAT,
            "Unsupported teacher experiment format")
    require(configuration.get("selection_split") == "val" and configuration.get("test_accessed") is False and
            configuration.get("encoder_frozen") is True and configuration.get("base_projection_frozen") is True,
            "Unsupported teacher training protocol")
    require(comparison.get("status") == "completed" and comparison.get("selection_split") == "val" and
            comparison.get("test_accessed") is False, "Teacher run was not completed validation-only")
    require(configuration.get("embedding_manifest_sha256") == store.manifest_sha256,
            "Teacher/embedding fingerprint mismatch")
    require(configuration.get("dimension") == store.metadata["dimension"], "Teacher dimension mismatch")
    sources = configuration.get("source_fingerprints", {})
    require(set(sources) == {"train", "val"} and sources["val"] == split.fingerprints,
            "Teacher/validation source fingerprint mismatch")
    training = sources["train"]
    require(training.get("dataset_manifest.json") == split.fingerprints["dataset_manifest.json"],
            "Teacher/train canonical manifest mismatch")
    for name in ("tokenization_manifest.json", "evidence.tokens.jsonl.gz", "evidence.jsonl.gz"):
        require(training.get(name) == split.fingerprints[name], "Teacher/shared train source fingerprint mismatch")
    manifest = dense.read_json(Path(data_dir) / "dataset_manifest.json")
    for name in ("documents.train.jsonl", "histories.train.jsonl", "queries.train.jsonl", "train.pairs.jsonl"):
        require(manifest.get("artifacts", {}).get(name, {}).get("sha256") == training.get(name) and
                isinstance(training.get(name), str), "Teacher/train source is outside current canonical manifest")
    model_path = Path(__file__).with_name("partner_evidence_teacher_model.py")
    require(configuration.get("code_fingerprints", {}).get(model_path.name) == dense.fingerprint(model_path),
            "Teacher model code fingerprint mismatch")
    baseline = comparison.get("base_validation", {})
    require(baseline.get("embedding_manifest_sha256") == store.manifest_sha256 and
            baseline.get("input_fingerprints") == split.fingerprints, "Recorded base validation source mismatch")
    require(configuration.get("candidate_mode") == comparison.get("candidate_mode"), "Candidate mode mismatch")


def _validate_candidates(configuration, comparison, run_dir, base_signature, store):
    mode = configuration.get("candidate_mode")
    derived = configuration.get("derived_candidate_fingerprint")
    if mode == "canonical":
        require(derived is None and comparison.get("candidate_audit") is None,
                "Canonical run unexpectedly binds derived candidates")
        return None
    require(mode == "dense-hard", "Unsupported calibration candidate mode")
    actual = dense.fingerprint(run_dir / "training_candidates.jsonl")
    require(derived == actual, "Derived training candidate fingerprint mismatch")
    audit = dense.read_json(run_dir / "candidate_audit.json")
    require(audit == comparison.get("candidate_audit"), "Recorded candidate audit mismatch")
    require(audit.get("derived_candidate_fingerprint") == actual and
            audit.get("base_checkpoint") == base_signature and
            audit.get("embedding_manifest_sha256") == store.manifest_sha256 and
            audit.get("source_fingerprints") == configuration["source_fingerprints"]["train"],
            "Derived candidate source binding mismatch")
    require(audit.get("split") == "train" and audit.get("val_accessed") is False and audit.get("test_accessed") is False and
            audit.get("canonical_pairs_sha256") == configuration["source_fingerprints"]["train"]["train.pairs.jsonl"],
            "Derived candidates were not train-only canonical replacements")
    return actual


def _load_head(mode, configuration, comparison, run_dir, weight):
    variant = dense.read_json(run_dir / mode / "run_config.json")
    require(set(variant) == set(configuration) | {"mode", "trainable_parameters"} and variant.get("mode") == mode and
            all(variant.get(key) == value for key, value in configuration.items()), "Variant/root configuration mismatch")
    summary = dense.read_json(run_dir / mode / "training_summary.json")
    require(summary == comparison.get("variants", {}).get(mode), "Variant training summary/comparison mismatch")
    checkpoint = run_dir / mode / "last.pt"
    signature = dense.fingerprint(checkpoint)
    require(signature == summary.get("last_checkpoint"), "Diagnostic last checkpoint fingerprint mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved.get("format") == teacher.FORMAT and saved.get("mode") == mode and
            saved.get("checkpoint_role") == "diagnostic_last" and saved.get("selected_epoch") is None and
            saved.get("selection_split") is None, "Expected an unselected diagnostic_last checkpoint")
    require(saved.get("configuration") == variant, "Diagnostic checkpoint/variant configuration mismatch")
    require(saved.get("epoch") == configuration.get("epochs") == summary.get("final_epoch") and
            integer(saved.get("epoch")) and saved["epoch"] > 0, "Diagnostic final epoch mismatch")
    require(saved.get("validation") == summary.get("final_validation"), "Diagnostic final validation mismatch")
    model = EvidenceInteractionTeacher(weight, hidden_dim=configuration["hidden_dim"],
            mlp_hidden=configuration["mlp_hidden"], mode=mode,
            attention_temperature=configuration["attention_temperature"], residual_scale=configuration["residual_scale"])
    require(variant.get("trainable_parameters") == sum(parameter.numel() for parameter in model.parameters()),
            "Diagnostic parameter count mismatch")
    state, expected = saved.get("state_dict", {}), model.state_dict()
    require(set(state) == set(expected), "Diagnostic model state keys mismatch")
    for name, value in state.items():
        require(isinstance(value, torch.Tensor) and value.shape == expected[name].shape and
                value.dtype == expected[name].dtype and bool(torch.isfinite(value).all()),
                "Invalid diagnostic model weights: " + name)
    require(torch.equal(state["base_weight"], weight), "Diagnostic model changed the frozen base weight")
    model.load_state_dict(state, strict=True)
    return model, signature, summary["final_validation"]


def run(args):
    started = time.monotonic()
    output = _output_path(args)
    require(integer(args.batch_size) and args.batch_size > 0 and integer(args.candidate_batch_size) and
            args.candidate_batch_size > 0, "Invalid calibration batch sizes")
    run_dir, data_dir = Path(args.run_dir).resolve(), Path(args.data_dir).resolve()
    # EvidenceDataset otherwise builds a cache in data_dir. This diagnostic
    # requires the existing cache and never creates input-side artifacts.
    require((data_dir / "token_cache/evidence.tokens.sqlite3").is_file(), "Existing read-only token cache is required")
    configuration = dense.read_json(run_dir / "run_config.json")
    comparison = dense.read_json(run_dir / "comparison.json")
    require(integer(configuration.get("seed")), "Invalid recorded experiment seed")
    dense.set_seed(configuration["seed"])
    device = dense.choose_device(args.device)
    store = dense.EmbeddingStore(args.embedding_dir, data_dir)
    split = teacher.TeacherSplit(data_dir, "val", store)
    _validate_configuration(configuration, comparison, split, store, data_dir)
    # Validate the old base's complete source bindings without loading train
    # features: the train hashes were checked against the current manifest.
    train_signature = SimpleNamespace(split="train", fingerprints=configuration["source_fingerprints"]["train"])
    weight, base_signature, base_saved = teacher.load_base_checkpoint(args.base_checkpoint, store, train_signature, split)
    require(configuration.get("base_checkpoint") == base_signature and
            configuration.get("base_selected_epoch") == base_saved["selected_epoch"], "Base checkpoint fingerprint/epoch mismatch")
    derived = _validate_candidates(configuration, comparison, run_dir, base_signature, store)
    variants = {}
    for mode in ("mean", "attention"):
        print(json.dumps({"stage": "calibrating_existing_last_checkpoint", "mode": mode,
                          "split": "val", "fixed_grid": list(ALPHAS)}), flush=True)
        model, signature, expected_last = _load_head(mode, configuration, comparison, run_dir, weight)
        model.to(device).eval().requires_grad_(False)
        scales, residual_stats = evaluate_scales(model, split, device, args.batch_size, args.candidate_batch_size)
        _check_controls(scales["0.0"], comparison["base_validation"], mode + " alpha=0")
        _check_controls(scales["1.0"], expected_last, mode + " alpha=1")
        best_alpha = 0.0
        for alpha in ALPHAS[1:]:
            if scales[str(alpha)]["recall@100"] > scales[str(best_alpha)]["recall@100"]:
                best_alpha = alpha
        variants[mode] = {"scales": scales, "residual_stats": residual_stats,
                          "diagnostic_checkpoint": signature, "checkpoint_role": "diagnostic_last_not_validation_best",
                          "endpoint_controls_passed": {"alpha0_base": True, "alpha1_recorded_last": True},
                          "best_validation_alpha": best_alpha, "best_validation": scales[str(best_alpha)],
                          "selection_metric": "recall@100", "ties_keep_earlier_fixed_alpha": True}
        del model
    base_score = comparison["base_validation"]["recall@100"]
    mean_score = variants["mean"]["best_validation"]["recall@100"]
    attention_score = variants["attention"]["best_validation"]["recall@100"]
    signal = mean_score > base_score or attention_score > base_score
    report = {"status": "completed", "format": FORMAT, "split": "val", "validation_exploratory": True,
              "no_training": True, "test_accessed": False, "weights_modified": False, "checkpoints_written": False,
              "fixed_grid": list(ALPHAS), "grid_selected_before_evaluation": True, "variants": variants,
              "base_validation": comparison["base_validation"], "base_checkpoint": base_signature,
              "embedding_manifest_sha256": store.manifest_sha256, "input_fingerprints": split.fingerprints,
              "derived_candidate_fingerprint": derived, "candidate_mode": configuration["candidate_mode"],
              "run_config": dense.fingerprint(run_dir / "run_config.json"),
              "source_comparison": dense.fingerprint(run_dir / "comparison.json"),
              "attention_beats_base": attention_score > base_score,
              "attention_beats_mean_control": attention_score > mean_score,
              "attention_minus_base_recall100_pp": 100 * (attention_score - base_score),
              "attention_minus_mean_control_recall100_pp": 100 * (attention_score - mean_score),
              "statistical_significance_established": False, "multi_seed_stability_established": False,
              "independent_teacher_advantage_established": False, "compression_ready": False,
              "decision": "calibration_signal_requires_independent_validation_no_compression" if signal else
                          "close_current_teacher_implementation_no_compression",
              "elapsed_seconds": round(time.monotonic() - started, 3)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "embedding-dir", "run-dir", "base-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
