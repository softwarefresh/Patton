"""Adaptive mean/max pooling probe for heterogeneous company groups.

This is a read-only feasibility experiment, not a trained model.  It uses the
existing frozen patent vectors and compares:

  mean            alpha=0 for every candidate (current baseline)
  max             alpha=1 for every candidate (negative control)
  global          one alpha fitted on train for all candidates
  size_group      one alpha per patent-count bucket (2-5, 6-10, 11-20)
  type_group      one alpha per company type (企业/高校/科研院所/未知)
  combined_group  one alpha per (size, type) cell with a global fallback
  random_*        random group assignment with the same group-size multiset

For a candidate c, score = (1 - alpha_g) * mean_score + alpha_g * max_score,
where g is c's group.  Alphas are fitted only on train (2022) by train
Recall@100, then evaluated on val (2023).  Test is never opened.

The point is to test the same small-innovation pattern as "one global
threshold -> several group-specific thresholds": does a meaningful group
definition beat one global alpha and a random grouping control?
"""
import argparse
import csv
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from partner_dense_baseline import EmbeddingStore, choose_device, ranking_metrics
from partner_evidence_dataset import EvidenceDataset, require
from partner_portfolio_probe import bootstrap_ci, fmt, patent_bucket, per_query_recall


FORMAT = "partner_adaptive_pooling_probe_v1"
QUERY_POOLING = "first6_raw_mean_l2"
DEFAULT_ALPHA_GRID = "0,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0"
SIZE_GROUPS = ("2-5", "6-10", "11-20")
EXPECTED_MEAN_R100 = 0.16352341425902947
EXPECTED_MAX_R100 = 0.0768640544991725


def load_type_map(path):
    """Read 信用代码 -> 类型 from the company pool CSV, if available."""
    path = Path(path) if path else None
    if path is None or not path.is_file():
        return {}, "heuristic"
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames and "信用代码" in reader.fieldnames and "类型" in reader.fieldnames,
            "Type map must contain 信用代码 and 类型 columns",
        )
        for row in reader:
            code = (row.get("信用代码") or "").strip()
            kind = (row.get("类型") or "").strip()
            if code and kind:
                result[code] = kind
    require(result, "Type map is empty")
    return result, str(path)


def company_type(code, name, type_map):
    """Use the registry type when available; otherwise a conservative name rule."""
    if code and code in type_map:
        return type_map[code]
    name = name or ""
    if "大学" in name or "学院" in name:
        return "高校"
    if "研究院" in name or "研究所" in name or "科学院" in name:
        return "科研院所"
    return "企业" if name else "未知"


class AdaptivePoolingData:
    """Frozen evidence, query vectors, targets, and candidate group labels."""

    def __init__(self, data_dir, split, store, type_map, type_source):
        require(split in ("train", "val"), "Adaptive pooling probe uses train and val only")
        require((Path(data_dir) / "dataset_manifest.json").is_file(), "Canonical dataset_manifest.json is required")
        with EvidenceDataset(data_dir, split) as dataset:
            self.split = split
            self.fingerprints = dict(dataset.fingerprints)
            self.doc_ids = sorted(dataset.histories)
            self.dimension = store.metadata["dimension"]
            count = len(self.doc_ids)
            self.evidence = torch.zeros(count, 20, self.dimension, dtype=torch.float32)
            self.mask = torch.zeros(count, 20, dtype=torch.bool)
            self.n_patents = np.zeros(count, dtype=np.int64)
            self.company_types = []
            for position, doc_id in enumerate(self.doc_ids):
                references = dataset.histories[doc_id]["patents"]
                require(1 <= len(references) <= 20, "Invalid historical evidence length")
                for slot, reference in enumerate(references):
                    entry = store.lookup.get(reference["application_number"])
                    require(entry is not None and entry[1] == reference["raw_text_sha256"],
                            "Evidence embedding missing or text SHA differs")
                    vector = store.vectors[entry[0]]
                    require(np.isfinite(vector).all(), "Nonfinite cached evidence vector")
                    self.evidence[position, slot] = torch.from_numpy(np.array(vector, copy=True))
                    self.mask[position, slot] = True
                self.n_patents[position] = len(references)
                document = dataset.documents[doc_id]
                self.company_types.append(company_type(
                    document.get("company_code"), document.get("company_name"), type_map
                ))
            self.query_ids = sorted(dataset.queries)
            self.queries = torch.from_numpy(np.stack([
                store.mean(dataset.histories[dataset.queries[qid]["source"]]["patents"][:6])
                for qid in self.query_ids
            ]))
            self.targets = {qid: set(dataset.queries[qid]["positives"]) for qid in self.query_ids}
            self.excluded = {qid: set(dataset.queries[qid]["excluded"]) for qid in self.query_ids}
        require(int(self.mask.sum()) == int(self.n_patents.sum()), "Evidence mask count mismatch")
        self.evidence_count = int(self.mask.sum().item())
        self.manifest_sha256 = store.manifest_sha256
        self.type_source = type_source
        self.size_groups = [patent_bucket(int(value)) for value in self.n_patents]
        self.type_groups = list(self.company_types)
        self.combined_groups = [
            size + "|" + kind for size, kind in zip(self.size_groups, self.type_groups)
        ]


def score_mean_max(data, device, batch_size):
    """Return [queries, candidates] cosine matrices for mean and per-patent max."""
    require(batch_size > 0, "Batch size must be positive")
    with torch.inference_mode():
        evidence = data.evidence.to(device)
        mask = data.mask.to(device)
        counts = mask.sum(dim=1, keepdim=True).float()
        means = evidence.sum(dim=1, keepdim=True) / counts.unsqueeze(-1)
        means = F.normalize(means.squeeze(1), dim=-1)
        patents = F.normalize(evidence, dim=-1)
        queries = F.normalize(data.queries.to(device), dim=-1)
        mean_scores = (queries @ means.T).cpu().numpy().astype(np.float32)
        max_scores = np.empty((len(data.query_ids), len(data.doc_ids)), dtype=np.float32)
        flat = patents.reshape(-1, patents.shape[-1])
        for start in range(0, len(data.query_ids), batch_size):
            end = min(start + batch_size, len(data.query_ids))
            scores = (queries[start:end] @ flat.T).reshape(end - start, len(data.doc_ids), 20)
            scores = scores.masked_fill(~mask.unsqueeze(0), -torch.inf)
            max_scores[start:end] = scores.max(dim=-1).values.cpu().numpy()
    require(np.isfinite(mean_scores).all() and np.isfinite(max_scores).all(), "Nonfinite score matrix")
    return mean_scores, max_scores


def build_allowed(data):
    return {
        qid: np.array(
            [index for index, doc_id in enumerate(data.doc_ids) if doc_id not in data.excluded[qid]],
            dtype=np.int64,
        )
        for qid in data.query_ids
    }


def alpha_vector(alpha_map, group_keys, fallback):
    return np.array([alpha_map.get(key, fallback) for key in group_keys], dtype=np.float32)


def rank_scores(mean_scores, max_scores, alpha, data, allowed, k=100):
    count = mean_scores.shape[1]
    if np.isscalar(alpha):
        alpha_array = np.full(count, float(alpha), dtype=np.float32)
    else:
        alpha_array = np.asarray(alpha, dtype=np.float32)
        require(alpha_array.shape == (count,), "Alpha vector length mismatch")
    scores = mean_scores * (1.0 - alpha_array)[None, :] + max_scores * alpha_array[None, :]
    rankings = {}
    for row, qid in enumerate(data.query_ids):
        indices = allowed[qid]
        order = indices[np.argsort(-scores[row, indices], kind="stable")[:k]]
        rankings[qid] = [data.doc_ids[index] for index in order]
    return rankings


def metrics_for_alpha(mean_scores, max_scores, alpha, data, allowed):
    rankings = rank_scores(mean_scores, max_scores, alpha, data, allowed)
    metrics = ranking_metrics(rankings, data.targets)
    recall, hit = per_query_recall(rankings, data.targets, data.query_ids)
    return metrics, recall, hit


def group_stats(data, group_keys):
    counts = Counter(group_keys)
    positives = Counter()
    index = {doc_id: position for position, doc_id in enumerate(data.doc_ids)}
    for qid, gold in data.targets.items():
        for doc_id in gold:
            positives[group_keys[index[doc_id]]] += 1
    return counts, positives


def fit_global(mean_scores, max_scores, data, allowed, grid):
    best = None
    for alpha in grid:
        metrics, recall, hit = metrics_for_alpha(mean_scores, max_scores, alpha, data, allowed)
        key = (metrics["recall@100"], metrics["MRR@100"], -alpha)
        if best is None or key > best[0]:
            best = (key, float(alpha), metrics, recall, hit)
    return best[1], best[2], best[3], best[4]


def fit_group(mean_scores, max_scores, data, allowed, group_keys, grid, global_alpha,
              min_candidates, min_positives, passes):
    counts, positives = group_stats(data, group_keys)
    groups = sorted(counts)
    alpha_map = {group: global_alpha for group in groups}
    active = [
        group for group in groups
        if counts[group] >= min_candidates and positives[group] >= min_positives
    ]
    for _ in range(passes):
        for group in active:
            best = None
            for alpha in grid:
                trial = dict(alpha_map)
                trial[group] = alpha
                alpha_array = alpha_vector(trial, group_keys, global_alpha)
                metrics, recall, hit = metrics_for_alpha(mean_scores, max_scores, alpha_array, data, allowed)
                key = (metrics["recall@100"], metrics["MRR@100"], -alpha)
                if best is None or key > best[0]:
                    best = (key, float(alpha), metrics, recall, hit)
            alpha_map[group] = best[1]
    alpha_array = alpha_vector(alpha_map, group_keys, global_alpha)
    metrics, recall, hit = metrics_for_alpha(mean_scores, max_scores, alpha_array, data, allowed)
    return alpha_map, active, counts, positives, metrics, recall, hit


def random_group_keys(group_keys, seed):
    labels = list(group_keys)
    rng = np.random.default_rng(seed)
    return [labels[index] for index in rng.permutation(len(labels))]


def evaluate_alpha_on_val(mean_scores, max_scores, data, allowed, alpha, group_keys, global_alpha):
    if np.isscalar(alpha):
        alpha_value = float(alpha)
    else:
        alpha_value = alpha_vector(alpha, group_keys, global_alpha)
    metrics, recall, hit = metrics_for_alpha(mean_scores, max_scores, alpha_value, data, allowed)
    return {"metrics": metrics, "recall": recall, "hit": hit}


def baseline_check(label, actual, expected, tolerance):
    if expected is None:
        return {"label": label, "actual": float(actual), "expected": None, "passed": None}
    difference = float(actual) - float(expected)
    passed = abs(difference) <= tolerance
    require(passed, "{} reproduction failed: actual={:.8f} expected={:.8f} difference={:.8f}".format(
        label, actual, expected, difference
    ))
    return {
        "label": label, "actual": float(actual), "expected": float(expected),
        "difference": difference, "tolerance": float(tolerance), "passed": passed,
    }


def bootstrap_comparison(left, right, iterations, seed):
    return bootstrap_ci(left, right, iterations, seed)


def summary_markdown(payload):
    lines = [
        "# 自适应池化探针：A（专利数分组）+ B（企业类型分组）",
        "",
        "只读、无神经网络训练；train 拟合 α，val 评估；test 未访问。",
        "",
        "## 基础信息",
        "",
        "- train 查询：{}；train 候选：{}".format(payload["train"]["query_count"], payload["train"]["candidate_count"]),
        "- val 查询：{}；val 候选：{}".format(payload["val"]["query_count"], payload["val"]["candidate_count"]),
        "- 查询表示：{}".format(payload["query_pooling"]),
        "- 企业类型来源：{}".format(payload["type_source"]),
        "- α 网格：{}".format(payload["alpha_grid"]),
        "- 最小组样本：{} 候选 / {} 正例".format(
            payload["min_group_candidates"], payload["min_group_positives"]),
        "",
        "## 基线复现",
        "",
    ]
    for check in payload["baseline_reproduction"]:
        lines.append(
            "- {}：actual={} expected={} passed={}".format(
                check["label"], fmt(check["actual"], 8), fmt(check["expected"], 8), check["passed"]
            )
        )
    lines.extend([
        "",
        "## val 主表",
        "",
        "| 方法 | R@20 | R@100 | Hit@100 | P@1 | MRR@100 | 说明 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for name, value in payload["methods"].items():
        metrics = value["metrics"]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
                name,
                metrics["recall@20"], metrics["recall@100"], metrics["Hit@100"],
                metrics["P@1"], metrics["MRR@100"],
                value.get("note", ""),
            )
        )
    lines.extend(["", "## 分组 α", "", "| 分组方式 | 组 | 候选数 | 正例数 | α |", "|---|---|---:|---:|---:|"])
    for kind in ("size_group", "type_group", "combined_group"):
        info = payload["groups"][kind]
        for group in sorted(info["alpha"]):
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    kind, group, info["candidate_counts"].get(group, 0),
                    info["positive_counts"].get(group, 0), fmt(info["alpha"][group], 3),
                )
            )
    lines.extend(["", "## 关键 bootstrap（R@100）", "", "| 对照 | 差值 | 95% CI | 正比例 |", "|---|---:|---|---:|"])
    for name, value in payload["comparisons"].items():
        lines.append(
            "| {} | {} | [{}, {}] | {} |".format(
                name, fmt(value["mean_difference"]), fmt(value["ci_low"]),
                fmt(value["ci_high"]), fmt(value["bootstrap_fraction_positive"], 3),
            )
        )
    lines.extend([
        "",
        "## 判定",
        "",
        "**{}**".format(payload["decision"]["decision"]),
        "",
    ])
    for item in payload["decision"]["reasons"]:
        lines.append("- " + item)
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- α 只在 train 上拟合，val 只用于比较；test 未访问。",
        "- 分组 α 只有同时赢“全局一个 α”和“随机同尺寸分组”才算分组本身有价值。",
        "- 小改动可能只有 0.5–2 pt，且可能是噪声；bootstrap 区间跨 0 时不要宣称创新。",
        "- 企业类型若来自名称启发式而非公司池 CSV，需要单独说明。",
        "- 这不是神经网络训练结果；不把它包装成“显著提升”。",
        "",
    ])
    return "\n".join(lines) + "\n"


def evaluate(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    require(not output.exists(), "Refusing existing evaluation report")
    require(args.batch_size > 0 and args.bootstrap > 0 and args.random_seeds > 0,
            "Invalid batch/bootstrap/random-seed setting")
    grid = [float(value) for value in args.alpha_grid.split(",") if value.strip()]
    require(grid and all(0.0 <= value <= 1.0 for value in grid), "Alpha grid must be in [0, 1]")
    require(args.min_group_candidates > 0 and args.min_group_positives > 0,
            "Group thresholds must be positive")
    device = choose_device(args.device)
    type_map, type_source = load_type_map(args.type_map)
    print(json.dumps({"stage": "loading_train", "type_source": type_source}), flush=True)
    store = EmbeddingStore(args.embedding_dir, args.data_dir)
    train = AdaptivePoolingData(args.data_dir, "train", store, type_map, type_source)
    train_allowed = build_allowed(train)
    print(json.dumps({"stage": "scoring_train"}), flush=True)
    train_mean, train_max = score_mean_max(train, device, args.batch_size)
    train.evidence = torch.empty(0)
    train.mask = torch.empty(0)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(json.dumps({"stage": "fitting_global"}), flush=True)
    global_alpha, global_metrics, global_recall, global_hit = fit_global(
        train_mean, train_max, train, train_allowed, grid
    )
    print(json.dumps({"stage": "fitting_size_groups"}), flush=True)
    size_alpha, size_active, size_counts, size_positives, size_metrics, size_recall, size_hit = fit_group(
        train_mean, train_max, train, train_allowed, train.size_groups, grid, global_alpha,
        args.min_group_candidates, args.min_group_positives, passes=2,
    )
    print(json.dumps({"stage": "fitting_type_groups"}), flush=True)
    type_alpha, type_active, type_counts, type_positives, type_metrics, type_recall, type_hit = fit_group(
        train_mean, train_max, train, train_allowed, train.type_groups, grid, global_alpha,
        args.min_group_candidates, args.min_group_positives, passes=2,
    )
    print(json.dumps({"stage": "fitting_combined_groups"}), flush=True)
    combined_alpha, combined_active, combined_counts, combined_positives, combined_metrics, combined_recall, combined_hit = fit_group(
        train_mean, train_max, train, train_allowed, train.combined_groups, grid, global_alpha,
        args.min_group_candidates, args.min_group_positives, passes=1,
    )
    random_size = []
    random_type = []
    for seed in range(args.random_seeds):
        print(json.dumps({"stage": "fitting_random_size", "seed": seed}), flush=True)
        keys = random_group_keys(train.size_groups, args.seed + seed)
        alpha_map, active, counts, positives, metrics, recall, hit = fit_group(
            train_mean, train_max, train, train_allowed, keys, grid, global_alpha,
            args.min_group_candidates, args.min_group_positives, passes=2,
        )
        random_size.append({"alpha": alpha_map, "active": active, "metrics": metrics, "recall": recall})
        print(json.dumps({"stage": "fitting_random_type", "seed": seed}), flush=True)
        keys = random_group_keys(train.type_groups, args.seed + 1000 + seed)
        alpha_map, active, counts, positives, metrics, recall, hit = fit_group(
            train_mean, train_max, train, train_allowed, keys, grid, global_alpha,
            args.min_group_candidates, args.min_group_positives, passes=2,
        )
        random_type.append({"alpha": alpha_map, "active": active, "metrics": metrics, "recall": recall})
    del train_mean, train_max, train_allowed
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(json.dumps({"stage": "loading_val"}), flush=True)
    val = AdaptivePoolingData(args.data_dir, "val", store, type_map, type_source)
    val_allowed = build_allowed(val)
    print(json.dumps({"stage": "scoring_val"}), flush=True)
    val_mean, val_max = score_mean_max(val, device, args.batch_size)
    methods = {}
    methods["mean"] = evaluate_alpha_on_val(val_mean, val_max, val, val_allowed, 0.0, None, 0.0)
    methods["max"] = evaluate_alpha_on_val(val_mean, val_max, val, val_allowed, 1.0, None, 0.0)
    methods["global"] = evaluate_alpha_on_val(
        val_mean, val_max, val, val_allowed, global_alpha, None, global_alpha
    )
    methods["size_group"] = evaluate_alpha_on_val(
        val_mean, val_max, val, val_allowed, size_alpha, val.size_groups, global_alpha
    )
    methods["type_group"] = evaluate_alpha_on_val(
        val_mean, val_max, val, val_allowed, type_alpha, val.type_groups, global_alpha
    )
    methods["combined_group"] = evaluate_alpha_on_val(
        val_mean, val_max, val, val_allowed, combined_alpha, val.combined_groups, global_alpha
    )
    random_size_val = []
    random_type_val = []
    for seed in range(args.random_seeds):
        keys = random_group_keys(val.size_groups, args.seed + seed)
        random_size_val.append(evaluate_alpha_on_val(
            val_mean, val_max, val, val_allowed, random_size[seed]["alpha"], keys, global_alpha
        ))
        keys = random_group_keys(val.type_groups, args.seed + 1000 + seed)
        random_type_val.append(evaluate_alpha_on_val(
            val_mean, val_max, val, val_allowed, random_type[seed]["alpha"], keys, global_alpha
        ))
    random_size_metrics = {
        key: float(np.mean([value["metrics"][key] for value in random_size_val]))
        for key in ("recall@20", "recall@100", "Hit@100", "P@1", "MRR@100")
    }
    random_type_metrics = {
        key: float(np.mean([value["metrics"][key] for value in random_type_val]))
        for key in ("recall@20", "recall@100", "Hit@100", "P@1", "MRR@100")
    }
    methods["random_size"] = {
        "metrics": random_size_metrics,
        "recall": np.mean(np.stack([value["recall"] for value in random_size_val]), axis=0),
        "note": "{} seeds 平均".format(args.random_seeds),
    }
    methods["random_type"] = {
        "metrics": random_type_metrics,
        "recall": np.mean(np.stack([value["recall"] for value in random_type_val]), axis=0),
        "note": "{} seeds 平均".format(args.random_seeds),
    }
    checks = [
        baseline_check("mean/max R@100", methods["mean"]["metrics"]["recall@100"],
                       args.expected_mean_r100, args.reproduction_tolerance),
        baseline_check("max/max R@100", methods["max"]["metrics"]["recall@100"],
                       args.expected_max_r100, args.reproduction_tolerance),
    ]
    comparisons = {}
    for name in ("size_group", "type_group"):
        comparisons[name + "_vs_mean"] = bootstrap_comparison(
            methods[name]["recall"], methods["mean"]["recall"], args.bootstrap, args.seed
        )
        comparisons[name + "_vs_global"] = bootstrap_comparison(
            methods[name]["recall"], methods["global"]["recall"], args.bootstrap, args.seed
        )
        comparison = methods["random_size"] if name == "size_group" else methods["random_type"]
        comparisons[name + "_vs_random"] = bootstrap_comparison(
            methods[name]["recall"], comparison["recall"], args.bootstrap, args.seed
        )
    comparisons["combined_vs_mean"] = bootstrap_comparison(
        methods["combined_group"]["recall"], methods["mean"]["recall"], args.bootstrap, args.seed
    )
    comparisons["combined_vs_global"] = bootstrap_comparison(
        methods["combined_group"]["recall"], methods["global"]["recall"], args.bootstrap, args.seed
    )

    def passes_group(name):
        return (
            methods[name]["metrics"]["recall@100"] > methods["mean"]["metrics"]["recall@100"]
            and comparisons[name + "_vs_mean"]["ci_low"] is not None
            and comparisons[name + "_vs_mean"]["ci_low"] > 0
            and methods[name]["metrics"]["recall@100"] > methods["global"]["metrics"]["recall@100"]
            and comparisons[name + "_vs_global"]["ci_low"] is not None
            and comparisons[name + "_vs_global"]["ci_low"] > 0
            and methods[name]["metrics"]["recall@100"] > methods[
                "random_size" if name == "size_group" else "random_type"
            ]["metrics"]["recall@100"]
            and comparisons[name + "_vs_random"]["ci_low"] is not None
            and comparisons[name + "_vs_random"]["ci_low"] > 0
        )

    size_ok = passes_group("size_group")
    type_ok = passes_group("type_group")
    global_vs_mean = bootstrap_comparison(
        methods["global"]["recall"], methods["mean"]["recall"], args.bootstrap, args.seed
    )
    global_ok = (
        methods["global"]["metrics"]["recall@100"] > methods["mean"]["metrics"]["recall@100"]
        and global_vs_mean["ci_low"] is not None and global_vs_mean["ci_low"] > 0
    )
    if size_ok or type_ok:
        decision = "candidate_signal"
    elif global_ok:
        decision = "global_only"
    else:
        decision = "no_signal"
    reasons = []
    if size_ok:
        reasons.append("size_group 同时赢 mean、global 和 random_size（bootstrap CI 下界 > 0）。")
    if type_ok:
        reasons.append("type_group 同时赢 mean、global 和 random_type（bootstrap CI 下界 > 0）。")
    if not size_ok and not type_ok:
        reasons.append("分组 α 没有同时满足 mean/global/random 三项条件。")
    if global_ok:
        reasons.append("全局一个 α 显著赢 mean；若分组不赢 global/random，只能算全局混合，不是多阈值创新。")
    if not global_ok:
        reasons.append("全局 α 也没有显著赢 mean。")
    if decision == "no_signal":
        reasons.append("当前信号不足，不建议把 A/B 包装成创新。")
    payload = {
        "format": FORMAT,
        "split": "train->val",
        "query_pooling": QUERY_POOLING,
        "type_source": type_source,
        "alpha_grid": grid,
        "min_group_candidates": args.min_group_candidates,
        "min_group_positives": args.min_group_positives,
        "random_seeds": args.random_seeds,
        "train": {
            "query_count": len(train.query_ids), "candidate_count": len(train.doc_ids),
            "positive_count": sum(len(gold) for gold in train.targets.values()),
        },
        "val": {
            "query_count": len(val.query_ids), "candidate_count": len(val.doc_ids),
            "positive_count": sum(len(gold) for gold in val.targets.values()),
        },
        "embedding_manifest_sha256": val.manifest_sha256,
        "input_fingerprints": val.fingerprints,
        "baseline_reproduction": checks,
        "methods": {
            name: {
                "metrics": value["metrics"],
                "note": value.get("note", ""),
            }
            for name, value in methods.items()
        },
        "groups": {
            "size_group": {
                "alpha": size_alpha, "active": size_active,
                "candidate_counts": dict(size_counts), "positive_counts": dict(size_positives),
            },
            "type_group": {
                "alpha": type_alpha, "active": type_active,
                "candidate_counts": dict(type_counts), "positive_counts": dict(type_positives),
            },
            "combined_group": {
                "alpha": combined_alpha, "active": combined_active,
                "candidate_counts": dict(combined_counts), "positive_counts": dict(combined_positives),
            },
        },
        "comparisons": comparisons,
        "global_alpha": global_alpha,
        "global_vs_mean": global_vs_mean,
        "random_size": [
            {"metrics": random_size_val[seed]["metrics"], "alpha": random_size[seed]["alpha"]}
            for seed in range(args.random_seeds)
        ],
        "random_type": [
            {"metrics": random_type_val[seed]["metrics"], "alpha": random_type[seed]["alpha"]}
            for seed in range(args.random_seeds)
        ],
        "decision": {"decision": decision, "size_ok": size_ok, "type_ok": type_ok,
                     "global_ok": global_ok, "reasons": reasons},
        "caveats": [
            "train-fitted alpha; val evaluation; test untouched",
            "small gains may be noise; group alpha must beat both global alpha and random grouping",
            "type groups may come from a name heuristic when the registry CSV is missing",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output).resolve() if args.summary_output else output.parent / (output.stem + ".summary.md")
    require(not summary_path.exists(), "Refusing existing summary report")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    summary_path.write_text(summary_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": "completed", "output": str(output), "summary": str(summary_path),
        "decision": decision, "elapsed_seconds": payload["elapsed_seconds"],
    }), flush=True)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def self_test():
    check(patent_bucket(2) == "2-5" and patent_bucket(10) == "6-10" and patent_bucket(11) == "11-20",
          "patent buckets")
    check(company_type("X", "某大学", {}) == "高校", "university heuristic")
    check(company_type("X", "某研究院", {}) == "科研院所", "institute heuristic")
    check(company_type("X", "某科技有限公司", {}) == "企业", "company heuristic")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "types.csv"
        path.write_text("信用代码,名称,专利数,类型\nX,某公司,1,企业\n", encoding="utf-8")
        type_map, source = load_type_map(path)
        check(type_map["X"] == "企业" and source.endswith("types.csv"), "type CSV")
    alpha = alpha_vector({"a": 0.1, "b": 0.2}, ["a", "b", "c"], 0.0)
    check(np.allclose(alpha, [0.1, 0.2, 0.0]), "alpha fallback")

    class FakeData:
        doc_ids = [10, 11, 12]
        query_ids = ["q1", "q2"]
        targets = {"q1": {10}, "q2": {12}}
        excluded = {"q1": set(), "q2": set()}

    data = FakeData()
    allowed = {qid: np.arange(3, dtype=np.int64) for qid in data.query_ids}
    mean = np.array([[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]], dtype=np.float32)
    maximum = np.array([[0.2, 0.8, 0.9], [0.9, 0.2, 0.1]], dtype=np.float32)
    ranked_mean = rank_scores(mean, maximum, 0.0, data, allowed)
    check(ranked_mean["q1"][0] == 10 and ranked_mean["q2"][0] == 12, "alpha=0 equals mean")
    ranked_max = rank_scores(mean, maximum, 1.0, data, allowed)
    check(ranked_max["q1"][0] == 12 and ranked_max["q2"][0] == 10, "alpha=1 equals max")
    mixed = rank_scores(mean, maximum, 0.5, data, allowed)
    check(mixed["q1"][0] == 11 and mixed["q2"][0] == 10, "alpha mix")
    print(json.dumps({"status": "passed", "checks": 9}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate", help="Fit alphas on train and evaluate on val")
    evaluate_parser.add_argument("--data-dir", type=Path, required=True)
    evaluate_parser.add_argument("--embedding-dir", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--summary-output", type=Path)
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--bootstrap", type=int, default=1000)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--random-seeds", type=int, default=3)
    evaluate_parser.add_argument("--type-map", type=Path, default=Path("data_pipeline/g_company_pool.csv"))
    evaluate_parser.add_argument("--alpha-grid", default=DEFAULT_ALPHA_GRID)
    evaluate_parser.add_argument("--min-group-candidates", type=int, default=100)
    evaluate_parser.add_argument("--min-group-positives", type=int, default=20)
    evaluate_parser.add_argument("--expected-mean-r100", type=float)
    evaluate_parser.add_argument("--expected-max-r100", type=float)
    evaluate_parser.add_argument("--reproduction-tolerance", type=float, default=0.001)
    evaluate_parser.set_defaults(func=evaluate)
    self_parser = commands.add_parser("self-test", help="Run synthetic checks without data")
    self_parser.set_defaults(func=lambda args: self_test())
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
