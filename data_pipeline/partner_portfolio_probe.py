"""Frozen-vector probe for multi-direction company representations.

This is a val-only diagnostic, not a trained model.  It compares a single mean
company vector against several fixed-budget portfolios built from the same
frozen patent vectors:

  mean                 one vector, all retained patents
  ipc_top4             up to four IPC-subclass groups (top three by size + rest)
  ipc_rare             top two by size + most distinctive group + rest
  random_top4_s*       random assignments matched to ipc_top4 group sizes
  random_rare_s*       random assignments matched to ipc_rare group sizes
  per_patent           every retained patent as its own vector (M<=20)

Scores are parameter-free max, top-2 mean, and normalized log-mean-exp
("softmax") with a fixed temperature grid.  The script never trains, never
opens the test split, and never modifies canonical data.  Its purpose is to
decide whether a trained multi-direction method is worth building.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from partner_dense_baseline import EmbeddingStore, choose_device, ranking_metrics
from partner_evidence_dataset import EvidenceDataset, require


FORMAT = "partner_portfolio_probe_v1"
QUERY_POOLING = "first6_raw_mean_l2"
COMPANY_POOLING = "group_mean_l2"
EVIDENCE_SLOTS = 20
POSITIVE_THRESHOLDS = (0.25, 1.0 / 3.0, 0.5)
PATENT_BUCKETS = ("2-5", "6-10", "11-20")
MAX_GROUPS = 4
DEFAULT_TEMPERATURES = "0.01,0.03,0.1,0.3,1.0,3.0"
IPC_PATTERN = re.compile(r"[A-HY][0-9]{2}[A-Z]")


def threshold_key(value):
    return "{:.4f}".format(value)


def patent_bucket(count):
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    return "11-20"


def parse_ipc_subclasses(value):
    """Return unique IPC subclasses (e.g. G06T) from a semicolon list."""
    result = []
    for code in str(value or "").replace(",", ";").split(";"):
        code = code.strip().upper()
        if not code:
            continue
        match = IPC_PATTERN.match(code)
        if match and match.group(0) not in result:
            result.append(match.group(0))
    return tuple(result)


def _labels_and_sizes(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return counts, ordered


def _assign_from_groups(labels, groups):
    mapping = {}
    for position, group in enumerate(groups):
        for label in group:
            require(label not in mapping, "Duplicate IPC group label")
            mapping[label] = position
    require(set(mapping) == set(labels), "IPC groups do not cover all patents")
    assign = [mapping[label] for label in labels]
    sizes = [0] * len(groups)
    for value in assign:
        sizes[value] += 1
    require(all(size > 0 for size in sizes), "IPC group is empty")
    return assign, sizes


def ipc_top4(labels):
    """Top three groups by size plus a merged rest group when needed."""
    _, ordered = _labels_and_sizes(labels)
    if len(ordered) <= MAX_GROUPS:
        groups = [[label] for label, _ in ordered]
    else:
        top = [label for label, _ in ordered[:3]]
        rest = [label for label, _ in ordered[3:]]
        groups = [[label] for label in top] + [rest]
    return _assign_from_groups(labels, groups)


def ipc_rare(labels, slot_vectors):
    """Top two groups, the most distinctive remaining group, and the rest."""
    _, ordered = _labels_and_sizes(labels)
    if len(ordered) <= MAX_GROUPS:
        groups = [[label] for label, _ in ordered]
        return _assign_from_groups(labels, groups)
    top = [label for label, _ in ordered[:2]]
    remaining = [label for label, _ in ordered[2:]]
    require(len(remaining) >= 2, "Rare-group selection needs at least two remaining groups")
    company = F.normalize(slot_vectors.mean(dim=0, dtype=torch.float32), dim=-1)
    best = None
    for label in remaining:
        positions = [position for position, value in enumerate(labels) if value == label]
        vector = F.normalize(slot_vectors[positions].mean(dim=0, dtype=torch.float32), dim=-1)
        similarity = float(torch.dot(vector, company))
        key = (similarity, label)
        if best is None or key < best[0]:
            best = (key, label)
    rare = best[1]
    other = [label for label in remaining if label != rare]
    require(other, "Merged rest group must not be empty")
    groups = [[top[0]], [top[1]], [rare], other]
    return _assign_from_groups(labels, groups)


def random_matched(labels, sizes, rng):
    """Random assignment with exactly the same group-size multiset."""
    slots = rng.permutation(len(labels)).tolist()
    assign = [0] * len(labels)
    start = 0
    for group, size in enumerate(sizes):
        require(size > 0, "Random group size must be positive")
        for slot in slots[start:start + size]:
            assign[slot] = group
        start += size
    require(start == len(labels), "Random group sizes do not cover all patents")
    return assign, list(sizes)


def build_assignments(data, representation, base_seed=42):
    """Build padded local group assignments [candidate, 20] and group sizes."""
    count = len(data.doc_ids)
    if representation == "mean":
        assign = np.zeros((count, 1), dtype=np.int64)
        sizes = [[int(data.n_patents[index])] for index in range(count)]
        return assign, sizes, 1
    if representation == "per_patent":
        assign = np.full((count, EVIDENCE_SLOTS), -1, dtype=np.int64)
        sizes = []
        for index in range(count):
            slots = int(data.n_patents[index])
            assign[index, :slots] = np.arange(slots, dtype=np.int64)
            sizes.append([1] * slots)
        return assign, sizes, EVIDENCE_SLOTS
    if representation in ("ipc_top4", "ipc_rare"):
        base = representation
        seed = base_seed
    elif representation.startswith("random_top4_"):
        base = "ipc_top4"
        seed = int(representation.rsplit("_", 1)[1])
    elif representation.startswith("random_rare_"):
        base = "ipc_rare"
        seed = int(representation.rsplit("_", 1)[1])
    else:
        raise ValueError("Unknown representation: " + representation)
    assign = np.full((count, EVIDENCE_SLOTS), -1, dtype=np.int64)
    sizes = []
    rng = np.random.default_rng(seed)
    for index in range(count):
        labels = data.technical_groups[index]
        slots = int(data.n_patents[index])
        require(len(labels) == slots, "IPC labels do not match evidence slots")
        if base == "ipc_top4":
            slot_assign, group_sizes = ipc_top4(labels)
        else:
            slot_assign, group_sizes = ipc_rare(labels, data.evidence[index, :slots])
        if representation.startswith("random_"):
            slot_assign, group_sizes = random_matched(labels, group_sizes, rng)
        require(len(slot_assign) == slots, "Assignment length mismatch")
        require(len(group_sizes) <= MAX_GROUPS, "Group budget exceeded")
        assign[index, :slots] = np.asarray(slot_assign, dtype=np.int64)
        sizes.append(group_sizes)
    return assign, sizes, MAX_GROUPS


def build_group_vectors(assign, r_max, evidence_device, device):
    """Mean and L2-normalize each group; return vectors [n, R, d], mask, counts."""
    require(assign.ndim == 2 and assign.shape[1] == EVIDENCE_SLOTS, "Invalid assignment shape")
    onehot = (assign[:, :, None] == np.arange(r_max, dtype=np.int64)[None, None, :]).astype(np.float32)
    onehot *= (assign >= 0)[:, :, None].astype(np.float32)
    onehot = torch.from_numpy(onehot).to(device)
    vectors = torch.einsum("nsr,nsd->nrd", onehot, evidence_device)
    counts = onehot.sum(dim=1)
    require(bool((counts > 0).any(dim=1).all()), "Every candidate needs at least one group")
    vectors = vectors / counts.clamp_min(1.0).unsqueeze(-1)
    vectors = F.normalize(vectors, dim=-1)
    mask = counts > 0
    return vectors, mask, mask.sum(dim=1)


def aggregate_scores(queries, vectors, mask, temperatures):
    """Return max, top-2 mean, and normalized log-mean-exp scores."""
    query = F.normalize(queries, dim=-1)
    flat = vectors.reshape(-1, vectors.shape[-1])
    scores = (query @ flat.T).reshape(len(queries), vectors.shape[0], vectors.shape[1])
    scores = scores.masked_fill(~mask.unsqueeze(0), -torch.inf)
    counts = mask.sum(dim=1).clamp_min(1)
    maximum = scores.max(dim=-1).values
    k = min(2, scores.shape[-1])
    top = torch.topk(scores, k=k, dim=-1).values
    if k == 2:
        top2 = torch.where(counts >= 2, top.sum(dim=-1) / 2.0, maximum)
    else:
        top2 = maximum
    result = {"max": maximum, "top2": top2}
    for temperature in temperatures:
        lse = torch.logsumexp(scores / temperature, dim=-1)
        result["softmax_{:g}".format(temperature)] = temperature * (lse - torch.log(counts.float()))
    return result


def rank_variants(data, vectors, mask, temperatures, device, batch_size):
    names = ["max", "top2"] + ["softmax_{:g}".format(value) for value in temperatures]
    rankings = {name: {} for name in names}
    vectors = vectors.to(device)
    mask = mask.to(device)
    with torch.inference_mode():
        for start in range(0, len(data.query_ids), batch_size):
            qids = data.query_ids[start:start + batch_size]
            queries = data.queries[start:start + len(qids)].to(device)
            scores = aggregate_scores(queries, vectors, mask, temperatures)
            for offset, qid in enumerate(qids):
                allowed = np.array(
                    [index for index, doc_id in enumerate(data.doc_ids) if doc_id not in data.excluded[qid]],
                    dtype=np.int64,
                )
                for name in names:
                    row = scores[name][offset].detach().cpu().numpy()
                    ordered = allowed[np.argsort(-row[allowed], kind="stable")[:100]]
                    rankings[name][qid] = [data.doc_ids[index] for index in ordered]
    return rankings


def per_query_recall(rankings, targets, query_ids):
    recall = np.zeros(len(query_ids), dtype=np.float64)
    hit = np.zeros(len(query_ids), dtype=np.float64)
    for index, qid in enumerate(query_ids):
        top = set(rankings[qid][:100])
        gold = targets[qid]
        recall[index] = len(top & gold) / len(gold)
        hit[index] = 1.0 if top & gold else 0.0
    return recall, hit


def build_positive_strata(data):
    """Method-independent strata based on query IPC versus full IPC groups."""
    index = {doc_id: position for position, doc_id in enumerate(data.doc_ids)}
    strata = {}
    for qid in data.query_ids:
        query_ipc = set(data.query_ipc[qid])
        strata[qid] = {}
        for doc_id in data.targets[qid]:
            position = index[doc_id]
            labels = data.technical_groups[position]
            slots = int(data.n_patents[position])
            groups = {}
            for slot, label in enumerate(labels):
                groups.setdefault(label, []).append(slot)
            best = None
            for label, group_slots in groups.items():
                group_ipc = set()
                for slot in group_slots:
                    group_ipc.update(data.patent_ipc_all[position][slot])
                overlap = len(query_ipc & group_ipc)
                union = len(query_ipc | group_ipc)
                jaccard = overlap / union if union else 0.0
                key = (overlap, jaccard, -len(group_slots), label)
                if best is None or key > best[0]:
                    best = (key, label, len(group_slots), overlap, jaccard)
            if best is None or best[3] == 0:
                share, minority, matched = None, None, None
            else:
                share = best[2] / slots
                minority = share < POSITIVE_THRESHOLDS[0]
                matched = best[1]
            strata[qid][doc_id] = {
                "share": share,
                "minority_0.25": minority,
                "matched_group": matched,
                "overlap": 0 if best is None else best[3],
                "n_patents": slots,
                "bucket": patent_bucket(slots),
            }
    return strata


def _mean_or_none(values):
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def summarize_positives(rankings, targets, query_ids, strata):
    ranks = {}
    for qid, gold in targets.items():
        rank_map = {doc_id: rank for rank, doc_id in enumerate(rankings[qid], 1)}
        for doc_id in gold:
            ranks[(qid, doc_id)] = rank_map.get(doc_id, 10 ** 9)
    overall = _mean_or_none([1.0 if rank <= 100 else 0.0 for rank in ranks.values()])
    thresholds = {}
    arrays = {}
    for threshold in POSITIVE_THRESHOLDS:
        key = threshold_key(threshold)
        minority, majority, unknown = [], [], []
        minority_query = np.full(len(query_ids), np.nan, dtype=np.float64)
        majority_query = np.full(len(query_ids), np.nan, dtype=np.float64)
        for index, qid in enumerate(query_ids):
            gold = targets[qid]
            minority_docs = []
            majority_docs = []
            for doc_id in gold:
                meta = strata[qid][doc_id]
                if meta["share"] is None:
                    unknown.append(ranks[(qid, doc_id)])
                elif meta["share"] < threshold:
                    minority.append(ranks[(qid, doc_id)])
                    minority_docs.append(doc_id)
                else:
                    majority.append(ranks[(qid, doc_id)])
                    majority_docs.append(doc_id)
            top = set(rankings[qid][:100])
            if minority_docs:
                minority_query[index] = len(top & set(minority_docs)) / len(minority_docs)
            if majority_docs:
                majority_query[index] = len(top & set(majority_docs)) / len(majority_docs)
        thresholds[key] = {
            "minority_positive_count": len(minority),
            "minority_positive_recall@100": _mean_or_none([1.0 if rank <= 100 else 0.0 for rank in minority]),
            "majority_positive_count": len(majority),
            "majority_positive_recall@100": _mean_or_none([1.0 if rank <= 100 else 0.0 for rank in majority]),
            "unknown_positive_count": len(unknown),
            "unknown_positive_recall@100": _mean_or_none([1.0 if rank <= 100 else 0.0 for rank in unknown]),
            "queries_with_minority_positive": int(np.isfinite(minority_query).sum()),
            "minority_query_recall@100": _mean_or_none(minority_query[np.isfinite(minority_query)]),
            "queries_with_majority_positive": int(np.isfinite(majority_query).sum()),
            "majority_query_recall@100": _mean_or_none(majority_query[np.isfinite(majority_query)]),
        }
        arrays["minority_" + key] = minority_query
        arrays["majority_" + key] = majority_query
    buckets = {}
    for bucket in PATENT_BUCKETS:
        selected = [
            ranks[(qid, doc_id)]
            for qid in targets
            for doc_id in targets[qid]
            if strata[qid][doc_id]["bucket"] == bucket
        ]
        buckets[bucket] = {
            "positive_count": len(selected),
            "positive_recall@100": _mean_or_none([1.0 if rank <= 100 else 0.0 for rank in selected]),
        }
    return {
        "overall_positive_recall@100": overall,
        "thresholds": thresholds,
        "buckets": buckets,
    }, arrays


def top100_stats(rankings, data):
    index = {doc_id: position for position, doc_id in enumerate(data.doc_ids)}
    means, large = [], []
    for ranking in rankings.values():
        sizes = np.array([int(data.n_patents[index[doc_id]]) for doc_id in ranking[:100]], dtype=np.float64)
        means.append(sizes.mean())
        large.append(float((sizes >= 11).mean()))
    return {
        "mean_candidate_patents_top100": float(np.mean(means)),
        "fraction_top100_with_11plus_patents": float(np.mean(large)),
    }


def fmt(value, digits=4):
    return "n/a" if value is None else "{:.{}f}".format(value, digits)


def bootstrap_ci(left, right, iterations, seed):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape, "Bootstrap arrays must align")
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) == 0:
        return {
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_fraction_positive": None,
            "query_count": 0,
        }
    difference = left - right
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(difference), len(difference))
        means[index] = difference[sample].mean()
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "bootstrap_fraction_positive": float((means > 0).mean()),
        "query_count": int(len(difference)),
    }


class PortfolioProbeData:
    """Frozen candidate evidence plus IPC metadata for the val split."""

    def __init__(self, data_dir, split, store):
        require(split == "val", "Portfolio probe is validation-only")
        require((Path(data_dir) / "dataset_manifest.json").is_file(), "Canonical dataset_manifest.json is required")
        with EvidenceDataset(data_dir, split) as dataset:
            self.fingerprints = dict(dataset.fingerprints)
            self.doc_ids = sorted(dataset.histories)
            self.dimension = store.metadata["dimension"]
            count = len(self.doc_ids)
            self.evidence = torch.zeros(count, EVIDENCE_SLOTS, self.dimension, dtype=torch.float32)
            self.mask = torch.zeros(count, EVIDENCE_SLOTS, dtype=torch.bool)
            self.technical_groups = []
            self.patent_ipc_all = []
            self.n_patents = np.zeros(count, dtype=np.int64)
            for position, doc_id in enumerate(self.doc_ids):
                references = dataset.histories[doc_id]["patents"]
                require(1 <= len(references) <= EVIDENCE_SLOTS, "Invalid historical evidence length")
                labels = []
                ipcs = []
                for slot, reference in enumerate(references):
                    entry = store.lookup.get(reference["application_number"])
                    require(entry is not None and entry[1] == reference["raw_text_sha256"],
                            "Evidence embedding missing or text SHA differs")
                    vector = store.vectors[entry[0]]
                    require(np.isfinite(vector).all(), "Nonfinite cached evidence vector")
                    self.evidence[position, slot] = torch.from_numpy(np.array(vector, copy=True))
                    self.mask[position, slot] = True
                    labels.append(reference.get("technical_group") or "UNKNOWN")
                    ipcs.append(parse_ipc_subclasses(reference.get("ipc_all", "")))
                self.technical_groups.append(labels)
                self.patent_ipc_all.append(ipcs)
                self.n_patents[position] = len(references)
            self.query_ids = sorted(dataset.queries)
            self.queries = torch.from_numpy(np.stack([
                store.mean(dataset.histories[dataset.queries[qid]["source"]]["patents"][:6])
                for qid in self.query_ids
            ]))
            self.targets = {qid: set(dataset.queries[qid]["positives"]) for qid in self.query_ids}
            self.excluded = {qid: set(dataset.queries[qid]["excluded"]) for qid in self.query_ids}
            self.query_ipc = {
                qid: parse_ipc_subclasses(
                    dataset.histories[dataset.queries[qid]["source"]]["patents"][0].get("ipc_all", "")
                )
                for qid in self.query_ids
            }
        require(int(self.mask.sum()) == int(self.n_patents.sum()), "Evidence mask count mismatch")
        self.evidence_count = int(self.mask.sum().item())
        self.manifest_sha256 = store.manifest_sha256


def evaluate_representation(data, representation, evidence_device, device, args, temperatures, strata):
    print(json.dumps({"stage": "representation", "name": representation}), flush=True)
    if representation == "mean":
        counts = data.mask.sum(dim=1, keepdim=True).to(device)
        vectors = evidence_device.sum(dim=1, keepdim=True) / counts.unsqueeze(-1).float()
        vectors = F.normalize(vectors, dim=-1)
        mask = torch.ones((len(data.doc_ids), 1), dtype=torch.bool, device=device)
        group_counts = torch.ones(len(data.doc_ids), dtype=torch.long, device=device)
    elif representation == "per_patent":
        vectors = F.normalize(evidence_device, dim=-1)
        mask = data.mask.to(device)
        group_counts = mask.sum(dim=1)
    else:
        assign, _, r_max = build_assignments(data, representation, base_seed=args.seed)
        vectors, mask, group_counts = build_group_vectors(assign, r_max, evidence_device, device)
    rankings = rank_variants(data, vectors, mask, temperatures, device, args.batch_size)
    variants = {}
    arrays = {}
    for name in ["max", "top2"] + ["softmax_{:g}".format(value) for value in temperatures]:
        metrics = ranking_metrics(rankings[name], data.targets)
        recall, hit = per_query_recall(rankings[name], data.targets, data.query_ids)
        positive, positive_arrays = summarize_positives(rankings[name], data.targets, data.query_ids, strata)
        variants[name] = {
            "metrics": metrics,
            "positive_strata": positive,
            "top100": top100_stats(rankings[name], data),
        }
        arrays[name] = {"query_recall": recall, "query_hit": hit}
        arrays[name].update(positive_arrays)
    softmax_names = ["softmax_{:g}".format(value) for value in temperatures]
    best_softmax = max(softmax_names, key=lambda name: variants[name]["metrics"]["recall@100"])
    counts_np = group_counts.detach().cpu().numpy().astype(np.int64)
    histogram = {str(value): int((counts_np == value).sum()) for value in sorted(set(counts_np.tolist()))}
    return variants, arrays, best_softmax, histogram


def build_comparisons(results, arrays, random_seeds, iterations, seed):
    comparisons = {}
    baseline = arrays["mean"]["max"]["query_recall"]
    for representation in results:
        for variant in results[representation]["variants"]:
            key = representation + "/" + variant
            current = arrays[representation][variant]
            comparison = {
                "vs_mean": bootstrap_ci(current["query_recall"], baseline, iterations, seed),
            }
            if representation.startswith("ipc_"):
                control_name = "random_top4" if representation == "ipc_top4" else "random_rare"
                controls = [
                    arrays[control_name + "_" + str(value)][variant]["query_recall"]
                    for value in range(random_seeds)
                ]
                control = np.nanmean(np.stack(controls), axis=0)
                comparison["vs_random_matched"] = bootstrap_ci(
                    current["query_recall"], control, iterations, seed
                )
                baseline_minority = arrays["mean"]["max"]["minority_0.2500"]
                baseline_majority = arrays["mean"]["max"]["majority_0.2500"]
                comparison["minority_vs_mean"] = bootstrap_ci(
                    current["minority_0.2500"], baseline_minority, iterations, seed
                )
                comparison["majority_vs_mean"] = bootstrap_ci(
                    current["majority_0.2500"], baseline_majority, iterations, seed
                )
            comparisons[key] = comparison
    return comparisons


def _gain(value, baseline):
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def build_decision(results, comparisons):
    baseline = results["mean"]["variants"]["max"]["positive_strata"]["thresholds"]["0.2500"]
    candidates = []
    for representation in ("ipc_top4", "ipc_rare"):
        if representation not in results:
            continue
        entry = results[representation]
        best_softmax = entry["best_softmax_variant"]
        for variant in ("max", "top2", best_softmax):
            key = representation + "/" + variant
            comparison = comparisons[key]
            positive = entry["variants"][variant]["positive_strata"]["thresholds"]["0.2500"]
            minority_gain = _gain(
                positive["minority_positive_recall@100"], baseline["minority_positive_recall@100"]
            )
            majority_gain = _gain(
                positive["majority_positive_recall@100"], baseline["majority_positive_recall@100"]
            )
            bucket_gains = {}
            for bucket in PATENT_BUCKETS:
                method_value = entry["variants"][variant]["positive_strata"]["buckets"][bucket]["positive_recall@100"]
                baseline_value = results["mean"]["variants"]["max"]["positive_strata"]["buckets"][bucket]["positive_recall@100"]
                bucket_gains[bucket] = _gain(method_value, baseline_value)
            finite_bucket_gains = [value for value in bucket_gains.values() if value is not None]
            mean_low = comparison["vs_mean"]["ci_low"]
            random_low = comparison["vs_random_matched"]["ci_low"]
            passes_mean = mean_low is not None and mean_low > 0
            passes_random = random_low is not None and random_low > 0
            mechanism = (
                minority_gain is not None
                and majority_gain is not None
                and minority_gain > majority_gain
            )
            buckets_ok = bool(finite_bucket_gains) and all(value >= -0.005 for value in finite_bucket_gains)
            candidates.append({
                "representation": representation,
                "variant": variant,
                "vs_mean": comparison["vs_mean"],
                "vs_random_matched": comparison["vs_random_matched"],
                "minority_gain": minority_gain,
                "majority_gain": majority_gain,
                "bucket_gains": bucket_gains,
                "passes_mean": passes_mean,
                "passes_random": passes_random,
                "mechanism_minority_gt_majority": mechanism,
                "buckets_ok": buckets_ok,
                "go": passes_mean and passes_random and mechanism and buckets_ok,
            })
    co_primary = [
        item for item in candidates
        if item["variant"] == "max" and item["representation"] in ("ipc_top4", "ipc_rare")
    ]
    primary_go = any(item["go"] for item in co_primary)
    secondary_go = any(item["go"] for item in candidates if item["variant"] != "max")
    if primary_go:
        decision = "candidate_signal"
    elif secondary_go:
        decision = "secondary_signal_only"
    else:
        decision = "no_reliable_signal"
    return {
        "decision": decision,
        "primary_go": primary_go,
        "secondary_go": secondary_go,
        "candidates": candidates,
    }


def baseline_check(label, actual, expected, tolerance):
    if expected is None:
        return {"label": label, "actual": float(actual), "expected": None, "passed": None}
    difference = float(actual) - float(expected)
    passed = abs(difference) <= tolerance
    require(passed, "{} reproduction failed: actual={:.8f} expected={:.8f} difference={:.8f}".format(
        label, actual, expected, difference
    ))
    return {
        "label": label,
        "actual": float(actual),
        "expected": float(expected),
        "difference": difference,
        "tolerance": float(tolerance),
        "passed": passed,
    }


def summary_markdown(payload):
    lines = [
        "# 企业多技术方向表示探针（val）",
        "",
        "本报告是只读、无训练的 go/no-go 探针；未访问 test、未修改规范数据。",
        "",
        "## 基础信息",
        "",
        "- 查询数：{}".format(payload["query_count"]),
        "- 候选数：{}".format(payload["candidate_count"]),
        "- 正例数：{}".format(payload["positive_count"]),
        "- 候选证据向量：{}（每企业至多 20）".format(payload["evidence_vector_count"]),
        "- 查询表示：{}".format(payload.get("query_pooling", "n/a")),
        "- embedding manifest：`{}`".format(payload["embedding_manifest_sha256"]),
        "",
        "## 基线复现",
        "",
    ]
    checks = payload.get("baseline_reproduction", [])
    if not checks:
        lines.append("- 未提供基线复现检查。")
    for check in checks:
        lines.append(
            "- {}：actual={} expected={} passed={}".format(
                check["label"], fmt(check["actual"], 8), fmt(check["expected"], 8), check["passed"],
            )
        )
    lines.extend([
        "",
        "## 主表",
        "",
        "| 表示 | 聚合 | R@20 | R@100 | Hit@100 | 少数方向正例召回 | 多数方向正例召回 | Top100 平均专利数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for representation, entry in payload["representations"].items():
        keep = ["max", "top2", entry["best_softmax_variant"]]
        for variant in keep:
            result = entry["variants"][variant]
            positive = result["positive_strata"]["thresholds"]["0.2500"]
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {} | {} | {:.2f} |".format(
                    representation,
                    variant,
                    result["metrics"]["recall@20"],
                    result["metrics"]["recall@100"],
                    result["metrics"]["Hit@100"],
                    fmt(positive["minority_positive_recall@100"]),
                    fmt(positive["majority_positive_recall@100"]),
                    result["top100"]["mean_candidate_patents_top100"],
                )
            )
    lines.extend(["", "主表只列 `max`、`top2` 和各表示按 val R@100 选出的最佳 softmax；完整温度网格在 JSON。"])
    base_positive = payload["representations"]["mean"]["variants"]["max"]["positive_strata"]["thresholds"]["0.2500"]
    lines.extend([
        "",
        "- 正例分层（25%）：少数方向 {}，多数方向 {}，未知 IPC {}；含少数方向正例的查询 {}。".format(
            base_positive["minority_positive_count"],
            base_positive["majority_positive_count"],
            base_positive["unknown_positive_count"],
            base_positive["queries_with_minority_positive"],
        ),
    ])
    for representation, entry in payload["representations"].items():
        lines.append("- {} 分组数分布：{}".format(representation, entry["group_count_histogram"]))
    lines.extend(["", "## 关键对照（R@100 配对 bootstrap）", "", "| 对照 | 差值 | 95% CI | bootstrap 正比例 |", "|---|---:|---|---:|"])
    best_rare = payload["representations"].get("ipc_rare", {}).get("best_softmax_variant")
    for key, comparison in payload["comparisons"].items():
        if key.endswith("/max") or key.endswith("/top2") or (best_rare and key == "ipc_rare/" + best_rare):
            for name, value in comparison.items():
                lines.append(
                    "| {} {} | {} | [{}, {}] | {} |".format(
                        key, name, fmt(value["mean_difference"]), fmt(value["ci_low"]),
                        fmt(value["ci_high"]), fmt(value["bootstrap_fraction_positive"], 3),
                    )
                )
    lines.extend(["", "## 判定", "", "**{}**".format(payload["decision"]["decision"]), ""])
    for item in payload["decision"]["candidates"]:
        lines.append(
            "- `{} / {}`：go={}；vs mean CI_low={}；vs random CI_low={}；少数增益={}；多数增益={}；桶={}".format(
                item["representation"], item["variant"], item["go"],
                fmt(item["vs_mean"]["ci_low"]), fmt(item["vs_random_matched"]["ci_low"]),
                fmt(item["minority_gain"]), fmt(item["majority_gain"]),
                ", ".join(
                    "{}: {}".format(bucket, fmt(value))
                    for bucket, value in item["bucket_gains"].items()
                ),
            )
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 探针只用原始冻结向量；若这里没有信号，训练版和压缩蒸馏都不应启动。",
        "- softmax 温度在 val 上选择，属于探索性结果；主判定使用无温度选择的 max。",
        "- 多个表示/聚合未做多重比较校正；bootstrap 区间仅用于判断方向，不作为最终显著性。",
        "- val 已用于多轮历史探索，本结果不是独立确认集；最终确认需要新的年份或划分。",
        "- 少数方向以查询 IPC 与候选完整 IPC 组的重叠定义，阈值为 25%；另报告 1/3 和 50% 作为敏感性。",
        "",
    ])
    return "\n".join(lines) + "\n"


def evaluate(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    require(not output.exists(), "Refusing existing evaluation report")
    require(args.split == "val", "Portfolio probe is validation-only")
    require(args.batch_size > 0, "Evaluation batch size must be positive")
    require(args.bootstrap > 0, "Bootstrap iterations must be positive")
    temperatures = [float(value) for value in args.temperatures.split(",") if value.strip()]
    require(temperatures and all(value > 0 for value in temperatures), "Temperatures must be positive")
    if args.representations:
        representations = [value.strip() for value in args.representations.split(",") if value.strip()]
    else:
        representations = ["mean", "per_patent", "ipc_top4", "ipc_rare"]
        for seed in range(args.random_seeds):
            representations.extend(["random_top4_" + str(seed), "random_rare_" + str(seed)])
    require("mean" in representations, "mean baseline is required")
    require("per_patent" in representations, "per_patent negative control is required")
    require(args.random_seeds > 0, "At least one random control seed is required")
    require(args.reproduction_tolerance > 0, "Reproduction tolerance must be positive")
    for representation in representations:
        require(
            representation in ("mean", "per_patent", "ipc_top4", "ipc_rare")
            or re.fullmatch(r"random_(top4|rare)_[0-9]+", representation),
            "Unknown representation: " + representation,
        )
    for base, control in (("ipc_top4", "random_top4_"), ("ipc_rare", "random_rare_")):
        if base in representations:
            for seed in range(args.random_seeds):
                require(control + str(seed) in representations,
                        "Missing matched random control: " + control + str(seed))
    device = choose_device(args.device)
    print(json.dumps({"stage": "checking_original_embedding_cache"}), flush=True)
    store = EmbeddingStore(args.embedding_dir, args.data_dir)
    print(json.dumps({"stage": "loading_validation_evidence"}), flush=True)
    data = PortfolioProbeData(args.data_dir, args.split, store)
    print(json.dumps({"stage": "building_method_independent_strata"}), flush=True)
    strata = build_positive_strata(data)
    evidence_device = data.evidence.to(device)
    results = {}
    arrays = {}
    checks = []
    for representation in representations:
        variants, variant_arrays, best_softmax, histogram = evaluate_representation(
            data, representation, evidence_device, device, args, temperatures, strata
        )
        results[representation] = {
            "group_count_histogram": histogram,
            "best_softmax_variant": best_softmax,
            "variants": variants,
        }
        arrays[representation] = variant_arrays
        if representation == "mean":
            checks.append(baseline_check(
                "mean/max R@100", variants["max"]["metrics"]["recall@100"],
                args.expected_mean_r100, args.reproduction_tolerance,
            ))
        if representation == "per_patent":
            checks.append(baseline_check(
                "per_patent/max R@100", variants["max"]["metrics"]["recall@100"],
                args.expected_fullmax_r100, args.reproduction_tolerance,
            ))
    print(json.dumps({"stage": "bootstrap_comparisons", "iterations": args.bootstrap}), flush=True)
    comparisons = build_comparisons(results, arrays, args.random_seeds, args.bootstrap, args.seed)
    decision = build_decision(results, comparisons)
    payload = {
        "format": FORMAT,
        "split": "val",
        "query_pooling": QUERY_POOLING,
        "company_pooling": COMPANY_POOLING,
        "query_count": len(data.query_ids),
        "candidate_count": len(data.doc_ids),
        "positive_count": sum(len(gold) for gold in data.targets.values()),
        "evidence_vector_count": data.evidence_count,
        "evidence_vectors_per_company_cap": EVIDENCE_SLOTS,
        "vector_dimension": data.dimension,
        "embedding_manifest_sha256": data.manifest_sha256,
        "input_fingerprints": data.fingerprints,
        "temperatures": temperatures,
        "random_seeds": args.random_seeds,
        "representations": results,
        "comparisons": comparisons,
        "baseline_reproduction": checks,
        "decision": decision,
        "caveats": [
            "val-only, no training, no test access",
            "softmax temperature selected on val; exploratory",
            "multiple representations and variants are not corrected for multiple comparisons",
            "val has been used in prior exploratory rounds; not an independent confirmation set",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    summary_path = Path(args.summary_output).resolve() if args.summary_output else output.parent / (output.stem + ".summary.md")
    require(not summary_path.exists(), "Refusing existing summary report")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    summary_path.write_text(summary_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "output": str(output),
        "summary": str(summary_path),
        "decision": decision["decision"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }), flush=True)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def self_test():
    labels = ["A", "A", "B", "C", "D", "E"]
    assign, sizes = ipc_top4(labels)
    check(len(assign) == 6 and sizes == [2, 1, 1, 2], "ipc_top4 sizes")
    check(sorted(set(assign)) == [0, 1, 2, 3], "ipc_top4 coverage")
    vectors = torch.tensor([
        [1.0, 0.0], [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
    ])
    assign, sizes = ipc_rare(labels, vectors)
    check(sizes == [2, 1, 1, 2], "ipc_rare sizes")
    check(assign[4] == 2, "ipc_rare selects the distinctive group")
    check(assign[5] == 3, "ipc_rare merges the rest")
    assign, sizes = random_matched(labels, [2, 1, 1, 2], np.random.default_rng(7))
    check(sorted(sizes) == [1, 1, 2, 2], "random matched sizes")
    check(all(value in (0, 1, 2, 3) for value in assign), "random matched coverage")
    check(parse_ipc_subclasses("G06T5/00; G06T3/40") == ("G06T",), "IPC parsing")
    check(patent_bucket(2) == "2-5" and patent_bucket(10) == "6-10" and patent_bucket(11) == "11-20",
          "patent buckets")
    queries = torch.tensor([[1.0, 0.0]])
    vectors = F.normalize(torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.8, 0.6], [0.6, 0.8]],
    ]), dim=-1)
    mask = torch.ones((2, 2), dtype=torch.bool)
    scores = aggregate_scores(queries, vectors, mask, [0.001, 1000.0])
    check(scores["max"][0, 0].item() > 0.999, "max aggregation")
    check(abs(scores["top2"][0, 0].item() - 0.5) < 1e-5, "top2 aggregation")
    check(scores["softmax_0.001"][0, 0].item() > 0.99, "softmax tends to max")
    check(abs(scores["softmax_1000"][0, 0].item() - 0.5) < 0.01, "softmax tends to mean")
    masked = torch.tensor([[True, False]])
    scores = aggregate_scores(queries, vectors[:1], masked, [0.1])
    check(int(torch.isfinite(scores["max"]).sum()) == 1, "masked group handling")
    ci = bootstrap_ci(np.array([1.0, 0.0]), np.array([1.0, 0.0]), 50, 1)
    check(ci["mean_difference"] == 0.0 and ci["ci_low"] == 0.0 and ci["ci_high"] == 0.0, "bootstrap null")
    print(json.dumps({"status": "passed", "checks": 15}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate", help="Run the val-only portfolio probe")
    evaluate_parser.add_argument("--data-dir", type=Path, required=True)
    evaluate_parser.add_argument("--embedding-dir", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("val",), default="val")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--summary-output", type=Path)
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--bootstrap", type=int, default=1000)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--random-seeds", type=int, default=3)
    evaluate_parser.add_argument("--temperatures", default=DEFAULT_TEMPERATURES)
    evaluate_parser.add_argument("--representations", default="")
    evaluate_parser.add_argument("--expected-mean-r100", type=float)
    evaluate_parser.add_argument("--expected-fullmax-r100", type=float)
    evaluate_parser.add_argument("--reproduction-tolerance", type=float, default=0.001)
    evaluate_parser.set_defaults(func=evaluate)
    self_parser = commands.add_parser("self-test", help="Run synthetic checks without data")
    self_parser.set_defaults(func=lambda args: self_test())
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
