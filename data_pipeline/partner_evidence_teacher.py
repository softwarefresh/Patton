"""Paired evidence-head feasibility experiment on the existing frozen cache.

The trained dense projection is frozen. Mean and query-conditioned attention
heads add a bounded residual to that same base score. This is neither a trained
GraphFormers model nor the proposed compressed contribution-preserving model.
Only train/val are loaded; selection always uses the complete validation pool.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

import partner_dense_baseline as dense
from partner_evidence_dataset import EvidenceDataset, integer, require
from partner_evidence_max_baseline import CompleteEvidenceVectors
from partner_evidence_teacher_model import EvidenceInteractionTeacher


FORMAT = "partner_evidence_residual_teacher_v1"


def log(**values):
    print(json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


class TeacherSplit(CompleteEvidenceVectors):
    def __init__(self, data_dir, split, store):
        require(split in ("train", "val"), "Teacher experiment accepts train/val only")
        super().__init__(data_dir, split, store)
        self.split = split
        self.position = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        with EvidenceDataset(data_dir, split) as dataset:
            # Preserve the original NumPy float32 mean path for the frozen base.
            self.documents = torch.from_numpy(np.stack([
                store.mean(dataset.histories[doc_id]["patents"]) for doc_id in self.doc_ids]))
            self.pairs = {pair["query_id"]: list(pair["candidate_ids"]) for pair in dataset.pairs}
        require(bool(torch.isfinite(self.documents).all()) and bool(torch.isfinite(self.queries).all()),
                "Nonfinite mean features")


def load_base_checkpoint(path, store, training, validation):
    signature = dense.fingerprint(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved.get("format") == dense.FORMAT and saved.get("selection_split") == "val",
            "Unsupported base checkpoint; validation-selected dense projection required")
    config = saved.get("configuration", {})
    require(config.get("embedding_manifest_sha256") == store.manifest_sha256,
            "Base checkpoint/embedding fingerprint mismatch")
    require(config.get("selection_split") == "val" and config.get("test_accessed") is False and
            config.get("encoder_frozen") is True, "Unsupported base training protocol")
    for split in (training, validation):
        require(config.get("source_fingerprints", {}).get(split.split) == split.fingerprints,
                "Base checkpoint/source fingerprint mismatch: " + split.split)
    dimension = store.metadata["dimension"]
    require(config.get("dimension") == dimension and saved.get("dimension") == dimension,
            "Base checkpoint dimension mismatch")
    epoch = saved.get("selected_epoch")
    require(integer(epoch) and epoch >= 0, "Invalid base checkpoint epoch")
    state = saved.get("state_dict", {})
    require(set(state) == {"linear.weight"}, "Unsupported base projection state")
    weight = state["linear.weight"]
    require(isinstance(weight, torch.Tensor) and weight.shape == (dimension, dimension) and
            weight.dtype == torch.float32 and bool(torch.isfinite(weight).all()),
            "Invalid base projection weights")
    return weight.detach().clone(), signature, saved


def evaluate_split(model, split, device, query_batch_size=8, candidate_batch_size=512):
    """Exhaustive scoring with two-dimensional chunks and stable ID tie breaks."""
    require(query_batch_size > 0 and candidate_batch_size > 0, "Evaluation batch sizes must be positive")
    require(split.doc_ids == sorted(set(split.doc_ids)), "Candidate IDs must be sorted and unique")
    model.eval()
    rankings = {}
    totals = {"normalized_attention_entropy_sum": 0.0, "attention_max_weight_sum": 0.0,
              "attention_pair_count": 0}
    last_log = time.monotonic()
    with torch.inference_mode():
        # Each candidate is projected only once per validation pass.
        chunks = []
        for start in range(0, len(split.doc_ids), candidate_batch_size):
            stop = start + candidate_batch_size
            chunks.append(model.prepare_candidates(split.evidence[start:stop].to(device),
                          split.mask[start:stop].to(device), split.documents[start:stop].to(device)))
        candidates = tuple(torch.cat([chunk[index] for chunk in chunks], dim=0) for index in range(3))
        del chunks
        queries = model.prepare_queries(split.queries.to(device))
        for start in range(0, len(split.query_ids), query_batch_size):
            qids = split.query_ids[start:start + query_batch_size]
            query = tuple(value[start:start + len(qids)] for value in queries)
            scores = np.empty((len(qids), len(split.doc_ids)), dtype=np.float32)
            for candidate_start in range(0, len(split.doc_ids), candidate_batch_size):
                stop = candidate_start + candidate_batch_size
                candidate = tuple(value[candidate_start:stop] for value in candidates)
                values, diagnostics = model.score_prepared(query, candidate, return_diagnostics=True)
                scores[:, candidate_start:stop] = values.cpu().numpy()
                # Sum diagnostics on-device; synchronize once per query batch.
                for key in totals:
                    totals[key] = totals[key] + diagnostics[key]
            require(np.isfinite(scores).all(), "Nonfinite teacher scores")
            for offset, qid in enumerate(qids):
                allowed = np.array([index for index, doc_id in enumerate(split.doc_ids)
                                    if doc_id not in split.excluded[qid]], dtype=np.int64)
                ordered = allowed[np.argsort(-scores[offset, allowed], kind="stable")[:100]]
                rankings[qid] = [split.doc_ids[index] for index in ordered]
            if time.monotonic() - last_log >= 25:
                log(stage="validation_progress", completed_queries=start + len(qids),
                    total_queries=len(split.query_ids), full_candidate_count=len(split.doc_ids))
                last_log = time.monotonic()
        totals = {key: float(value) for key, value in totals.items()}
    metrics = dense.ranking_metrics(rankings, split.targets)
    pairs = totals["attention_pair_count"]
    metrics.update(split="val", query_count=len(split.query_ids), candidate_count=len(split.doc_ids),
                   positive_count=sum(len(value) for value in split.targets.values()),
                   evaluation_candidate_pool="all_allowed_documents", gold_used_for_scoring=False,
                   mean_normalized_attention_entropy=totals["normalized_attention_entropy_sum"] / pairs,
                   mean_max_attention_weight=totals["attention_max_weight_sum"] / pairs,
                   attention_diagnostics_scope="all_scored_pairs_including_excluded_candidates")
    return metrics, rankings


def make_model(weight, args, mode, device):
    return EvidenceInteractionTeacher(weight, hidden_dim=args.hidden_dim, mlp_hidden=args.mlp_hidden,
                                     mode=mode, attention_temperature=args.attention_temperature,
                                     residual_scale=args.residual_scale).to(device)


def snapshot(path, model, configuration, epoch, metrics):
    # Never migrate the live model after constructing its optimizer.
    saved_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch.save({"format": FORMAT, "configuration": configuration, "mode": configuration["mode"],
                "selected_epoch": epoch, "selection_split": "val", "validation": metrics,
                "state_dict": saved_weights}, path)


def preflight(weight, training, validation, args, device, base_model):
    """A disposable one-step GPU/CPU check; metrics here never select a model."""
    dense.set_seed(args.seed)
    model = make_model(weight, args, "attention", device)
    qid = training.query_ids[0]
    positions = [training.position[doc_id] for doc_id in training.pairs[qid]]
    query = training.queries[:1].to(device)
    evidence, mask = training.evidence[positions].to(device), training.mask[positions].to(device)
    prepared = model.prepare_candidates(evidence, mask, training.documents[positions].to(device))
    scores = model.score_prepared(model.prepare_queries(query), prepared).squeeze(0)
    expected = (base_model(query) @ base_model(training.documents[positions].to(device)).T).squeeze(0)
    require(torch.allclose(scores, expected, atol=1e-6, rtol=1e-6), "Epoch-zero base reproduction failed")
    positive_mask = torch.tensor([doc_id in training.targets[qid] for doc_id in training.pairs[qid]],
                                 dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss = dense.multi_positive_loss(scores / args.temperature, positive_mask)
    require(bool(torch.isfinite(loss)), "Nonfinite preflight loss")
    loss.backward()
    optimizer.step()
    changed = any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    require(changed and torch.equal(model.base_weight.cpu(), weight), "Preflight did not update only the head")
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        count = min(args.candidate_batch_size, len(validation.doc_ids))
        candidates = model.prepare_candidates(validation.evidence[:count].to(device),
                    validation.mask[:count].to(device), validation.documents[:count].to(device))
        queries = model.prepare_queries(validation.queries[:args.eval_batch_size].to(device))
        scores = model.score_prepared(queries, candidates)
        require(bool(torch.isfinite(scores).all()), "Nonfinite preflight validation scores")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    report = {"status": "passed", "smoke_only": True, "head_parameters_updated": changed,
              "base_projection_frozen": True, "epoch_zero_reproduces_base": True,
              "validation_query_sample": len(scores), "validation_candidate_sample": count,
              "sample_scoring_seconds": round(time.monotonic() - started, 3),
              "sample_metrics_used_for_selection": False, "test_accessed": False}
    log(stage="preflight", **report)
    return report


def train_variant(mode, weight, training, validation, training_device, args, config, baseline, output, device):
    dense.set_seed(args.seed)  # Equal initial parameters and query order for both controls.
    model = make_model(weight, args, mode, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = dict(config, mode=mode, trainable_parameters=sum(p.numel() for p in model.parameters()))
    destination = output / mode
    destination.mkdir()
    dense.write_json(destination / "run_config.json", config)
    best, best_epoch, history = baseline["recall@100"], 0, []
    base_record = {"epoch": 0, "training_loss": None, "validation": baseline,
                   "evaluation": "same_zero_residual_frozen_base_full_validation"}
    history.append(base_record)
    snapshot(destination / "best.pt", model, config, 0, baseline)
    dense.write_json(destination / "history.json", history)
    log(stage="epoch", mode=mode, **base_record)
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        order = list(range(len(training.query_ids)))
        random.Random(args.seed + epoch).shuffle(order)
        total_loss = 0.0
        last_log = time.monotonic()
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in batch:
                qid = training.query_ids[index]
                ids = training.pairs[qid]
                require(training.targets[qid] <= set(ids), "Training candidates drop a positive")
                positions = torch.tensor([training.position[doc_id] for doc_id in ids], device=device)
                candidates = model.prepare_candidates(training_device[0][positions], training_device[1][positions],
                                                      training_device[2][positions])
                query = model.prepare_queries(training_device[3][index:index + 1])
                scores = model.score_prepared(query, candidates).squeeze(0) / args.temperature
                targets = torch.tensor([doc_id in training.targets[qid] for doc_id in ids], dtype=torch.bool,
                                       device=device)
                losses.append(dense.multi_positive_loss(scores, targets))
            loss = torch.stack(losses).mean()
            require(bool(torch.isfinite(loss)), "Nonfinite teacher training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            require(bool(torch.isfinite(norm)), "Nonfinite teacher gradient")
            optimizer.step()
            total_loss += loss.item() * len(batch)
            if time.monotonic() - last_log >= 25:
                log(stage="training_progress", mode=mode, epoch=epoch,
                    completed_queries=start + len(batch), total_queries=len(order))
                last_log = time.monotonic()
        log(stage="complete_validation_pool", mode=mode, epoch=epoch)
        metrics, _ = evaluate_split(model, validation, device, args.eval_batch_size, args.candidate_batch_size)
        record = {"epoch": epoch, "training_loss": total_loss / len(order), "validation": metrics,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        history.append(record)
        if dense.select_validation_checkpoint(metrics, best):
            best, best_epoch = metrics["recall@100"], epoch
            snapshot(destination / "best.pt", model, config, epoch, metrics)
        dense.write_json(destination / "history.json", history)
        log(stage="epoch", mode=mode, **record)
    require(torch.equal(model.base_weight.cpu(), weight), "Frozen base projection changed")
    selected = history[best_epoch]["validation"]
    summary = {"mode": mode, "best_epoch": best_epoch, "best_validation": selected,
               "trained_residual_selected": best_epoch > 0, "base_projection_frozen": True,
               "trainable_parameters": config["trainable_parameters"],
               "best_checkpoint": dense.fingerprint(destination / "best.pt"), "test_accessed": False}
    dense.write_json(destination / "training_summary.json", summary)
    return summary


def train_pair(args):
    started = time.monotonic()
    for name in ("epochs", "batch_size", "eval_batch_size", "candidate_batch_size", "hidden_dim", "mlp_hidden"):
        require(integer(getattr(args, name)) and getattr(args, name) > 0, "Invalid positive integer: " + name)
    for name in ("lr", "temperature", "attention_temperature", "residual_scale"):
        require(math.isfinite(getattr(args, name)) and getattr(args, name) > 0, "Invalid positive value: " + name)
    require(math.isfinite(args.weight_decay) and args.weight_decay >= 0, "Invalid weight decay")
    output = Path(args.output_dir).resolve()
    require(not output.exists(), "Refusing existing teacher output; choose a fresh run directory")
    device = dense.choose_device(args.device)
    dense.set_seed(args.seed)
    log(stage="checking_original_embedding_cache")
    store = dense.EmbeddingStore(args.embedding_dir, args.data_dir)
    log(stage="loading_training_evidence")
    training = TeacherSplit(args.data_dir, "train", store)
    log(stage="loading_validation_evidence")
    validation = TeacherSplit(args.data_dir, "val", store)
    weight, base_signature, saved = load_base_checkpoint(args.base_checkpoint, store, training, validation)
    configuration = {
        "format": FORMAT, "encoder_frozen": True, "base_projection_frozen": True,
        "query_encoding": "existing_frozen_bert_raw_mean_first6_no_graphformer",
        "trained_components": "shared_low_dim_projection_and_query_evidence_set_residual_mlp",
        "base_checkpoint": base_signature, "base_selected_epoch": saved["selected_epoch"],
        "embedding_manifest_sha256": store.manifest_sha256,
        "source_fingerprints": {"train": training.fingerprints, "val": validation.fingerprints},
        "selection_split": "val", "selection_metric": "recall@100", "test_accessed": False,
        "innovation_model": False, "fixed_index_budget_comparison": False,
        "candidate_protocol": "same_canonical_all_known_positives_plus_hard_and_random_unobserved_items",
        "training_query_count": len(training.query_ids), "validation_query_count": len(validation.query_ids),
        "validation_candidate_count": len(validation.doc_ids), "dimension": len(weight),
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "weight_decay": args.weight_decay, "temperature": args.temperature,
        "hidden_dim": args.hidden_dim, "mlp_hidden": args.mlp_hidden,
        "attention_temperature": args.attention_temperature, "residual_scale": args.residual_scale,
        "gradient_clip_norm": 1.0, "eval_query_batch_size": args.eval_batch_size,
        "eval_candidate_batch_size": args.candidate_batch_size, "torch_version": str(torch.__version__),
        "code_fingerprints": {name: dense.fingerprint(Path(__file__).with_name(name)) for name in
                              ("partner_evidence_teacher.py", "partner_evidence_teacher_model.py")}}
    output.mkdir(parents=True)
    dense.write_json(output / "run_config.json", configuration)
    base_model = dense.Projection(len(weight))
    base_model.load_state_dict({"linear.weight": weight})
    base_model.to(device).eval().requires_grad_(False)
    log(stage="frozen_projection_full_validation")
    baseline, _ = dense.evaluate_split(base_model, validation, device, 32)
    baseline.update(split="val", encoder_frozen=True, base_projection_frozen=True,
                    embedding_manifest_sha256=store.manifest_sha256,
                    input_fingerprints=validation.fingerprints)
    dense.write_json(output / "base.val.json", baseline)
    log(stage="base_validation", **baseline)
    smoke = preflight(weight, training, validation, args, device, base_model)
    dense.write_json(output / "preflight.json", smoke)
    del base_model
    # Raw train features fit on the 4090; validation is projected in chunks.
    training_device = (training.evidence.to(device), training.mask.to(device),
                       training.documents.to(device), training.queries.to(device))
    summaries = {}
    for mode in ("mean", "attention"):
        log(stage="training_variant", mode=mode)
        summaries[mode] = train_variant(mode, weight, training, validation, training_device, args,
                                        configuration, baseline, output, device)
    attention = summaries["attention"]["best_validation"]["recall@100"]
    mean = summaries["mean"]["best_validation"]["recall@100"]
    beats_base = attention > baseline["recall@100"]
    beats_mean = attention > mean
    comparison = {"status": "completed", "format": FORMAT, "selection_split": "val",
                  "base_validation": baseline, "variants": summaries,
                  "attention_minus_base_recall100_pp": 100 * (attention - baseline["recall@100"]),
                  "attention_minus_mean_control_recall100_pp": 100 * (attention - mean),
                  "attention_beats_base": beats_base, "attention_beats_mean_control": beats_mean,
                  "decision": "replicate_teacher_before_compression" if beats_base and beats_mean else
                              "stop_compression_and_review_teacher",
                  "statistical_significance_established": False, "multi_seed_stability_established": False,
                  "test_accessed": False, "innovation_model_trained": False,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
    dense.write_json(output / "comparison.json", comparison)
    log(**comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train-pair", help="Train matched mean/attention residual controls; validate only")
    for name in ("data-dir", "embedding-dir", "base-checkpoint", "output-dir"):
        training.add_argument("--" + name, type=Path, required=True)
    for name, default in (("epochs", 10), ("batch-size", 32), ("eval-batch-size", 8),
                          ("candidate-batch-size", 512), ("hidden-dim", 128), ("mlp-hidden", 64), ("seed", 42)):
        training.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", 0.001), ("weight-decay", 0.01), ("temperature", 0.05),
                          ("attention-temperature", 0.2), ("residual-scale", 0.25)):
        training.add_argument("--" + name, type=float, default=default)
    training.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_pair(args)


if __name__ == "__main__":
    main()
