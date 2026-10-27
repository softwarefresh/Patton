"""Train-only dense candidate replacement over the frozen mean projection.

The original random items and all known positives are retained. Each canonical
lexical hard item is replaced by one distinct high-scoring unobserved item.
Unobserved items remain weak training negatives, not verified nonpartners.
Canonical data and the supplied split are never modified.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch

import partner_dense_baseline as dense
from partner_evidence_dataset import id_list, integer, require, rows, sha256_file


FORMAT = "partner_dense_train_candidates_v1"
POLICY = "replace_lexical_hard_with_same_count_frozen_base_dense_hard_keep_random_and_all_known_positives"


def _candidate_order(qid, doc_ids):
    return sorted(doc_ids, key=lambda doc_id: (
        hashlib.sha256((qid + "\0" + str(doc_id)).encode("utf-8")).digest(), doc_id))


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "min": None, "q25": None, "median": None,
                "q75": None, "max": None, "mean": None, "sum": 0.0}
    require(bool(np.isfinite(values).all()), "Nonfinite audit values")
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(count=int(values.size), min=float(quantiles[0]), q25=float(quantiles[1]),
                median=float(quantiles[2]), q75=float(quantiles[3]), max=float(quantiles[4]),
                mean=float(values.mean()), sum=float(values.sum()))


def build_dense_training_candidates(training, data_dir, base_model, device, query_batch_size=32):
    """Return replacement ID lists, a train-only audit, and serializable rows.

    Dense ranking excludes historical exclusions, every known positive, and
    retained random items. Equal scores break by ascending company ID. The
    final combined candidate order is the same label-independent SHA ordering
    used by EvidenceDataset. Score/rank audits use no validation/test cohort.
    """
    require(training.split == "train", "Dense candidate mining accepts train only")
    require(integer(query_batch_size) and query_batch_size > 0, "Invalid query batch size")
    doc_ids, query_ids = training.doc_ids, training.query_ids
    require(doc_ids and all(integer(doc_id) for doc_id in doc_ids) and
            doc_ids == sorted(set(doc_ids)), "Candidate IDs must be sorted unique integers")
    require(query_ids and all(isinstance(qid, str) and qid for qid in query_ids) and
            query_ids == sorted(set(query_ids)), "Query IDs must be sorted and unique")
    require(training.position == {doc_id: index for index, doc_id in enumerate(doc_ids)},
            "Training candidate position map differs")
    require(training.documents.ndim == 2 and training.queries.ndim == 2 and
            training.documents.shape[0] == len(doc_ids) and training.queries.shape[0] == len(query_ids) and
            training.documents.shape[1] == training.queries.shape[1], "Invalid training feature shapes")
    require(set(training.pairs) == set(query_ids) == set(training.targets) == set(training.excluded),
            "Training query, pair, and target ID sets differ")
    path = Path(data_dir) / "train.pairs.jsonl"
    source_sha = sha256_file(path)
    require(source_sha == training.fingerprints.get(path.name), "Canonical training pair fingerprint mismatch")
    canonical = {}
    candidate_ids = set(doc_ids)
    for row in rows(path):
        qid = row.get("query_id")
        require(qid in training.targets and qid not in canonical, "Unexpected or duplicate canonical query")
        positives = id_list(row.get("positive_doc_ids"), "canonical positives")
        hard = id_list(row.get("hard_negative_doc_ids"), "canonical hard items")
        random_ids = id_list(row.get("random_negative_doc_ids"), "canonical random items")
        negatives = id_list(row.get("negative_doc_ids"), "canonical unobserved items")
        require(positives and set(positives) == set(training.targets[qid]), "Canonical positives differ")
        require(not set(hard) & set(random_ids), "Canonical hard/random overlap prevents equal unique budgets")
        require(set(negatives) == set(hard + random_ids), "Canonical negative groups differ")
        combined = set(positives + hard + random_ids)
        require(combined <= candidate_ids and not combined & set(training.excluded[qid]),
                "Canonical candidates include missing or excluded IDs")
        require(not set(negatives) & set(positives), "Canonical unobserved item is a known positive")
        require(list(training.pairs[qid]) == _candidate_order(qid, combined),
                "Canonical candidate IDs/order differ from loaded training split")
        canonical[qid] = (positives, hard, random_ids)
    require(set(canonical) == set(query_ids), "Canonical training query coverage differs")
    require(sha256_file(path) == source_sha, "Canonical pair source changed while reading")

    new_pairs, records, rankings = {}, [], {}
    counts = {name: [] for name in ("canonical_candidate", "new_candidate", "canonical_negative", "new_negative",
                                    "replaced_hard", "retained_random", "positive", "eligible_unobserved")}
    coverage = {name: {20: [], 100: []} for name in ("canonical", "new")}
    score_values = {name: [] for name in ("canonical_unobserved", "new_unobserved", "positives")}
    unobserved_count, unobserved_sum = 0, 0.0
    unobserved_min, unobserved_max = None, None
    positive_ranks, boundary_gaps = [], []
    previous_mode = base_model.training
    try:
        base_model.eval()
        with torch.inference_mode():
            documents = base_model(training.documents.to(device))
            require(documents.ndim == 2 and documents.shape[0] == len(doc_ids) and
                    bool(torch.isfinite(documents).all()), "Invalid projected candidate vectors")
            for start in range(0, len(query_ids), query_batch_size):
                qids = query_ids[start:start + query_batch_size]
                queries = base_model(training.queries[start:start + len(qids)].to(device))
                scores = (queries @ documents.T).cpu().numpy()
                require(np.isfinite(scores).all(), "Nonfinite dense mining scores")
                for offset, qid in enumerate(qids):
                    values = scores[offset]
                    positives, hard, random_ids = canonical[qid]
                    positive_set, random_set = set(positives), set(random_ids)
                    excluded = set(training.excluded[qid])
                    # doc_ids are ascending; stable score sort resolves ties by ID.
                    ordered = np.argsort(-values, kind="stable")
                    allowed = [int(position) for position in ordered if doc_ids[position] not in excluded]
                    unobserved = [position for position in allowed if doc_ids[position] not in positive_set]
                    available = [position for position in unobserved if doc_ids[position] not in random_set]
                    require(len(available) >= len(hard), "Insufficient eligible dense items for equal candidate budget")
                    dense_ids = [doc_ids[position] for position in available[:len(hard)]]
                    old_negative_set, new_negative_set = set(hard + random_ids), set(dense_ids + random_ids)
                    new_ids = _candidate_order(qid, positive_set | new_negative_set)
                    require(len(old_negative_set) == len(new_negative_set) and
                            len(new_ids) == len(training.pairs[qid]), "Dense replacement changes candidate budget")
                    require(not new_negative_set & (positive_set | excluded), "Forbidden dense training item")
                    new_pairs[qid] = new_ids
                    records.append({"query_id": qid, "canonical_candidate_ids": list(training.pairs[qid]),
                                    "positive_doc_ids": positives, "dense_hard_doc_ids": dense_ids,
                                    "retained_random_doc_ids": random_ids, "candidate_ids": list(new_ids)})
                    rankings[qid] = [doc_ids[position] for position in allowed[:100]]
                    for name, value in (("canonical_candidate", len(training.pairs[qid])), ("new_candidate", len(new_ids)),
                                        ("canonical_negative", len(old_negative_set)), ("new_negative", len(new_negative_set)),
                                        ("replaced_hard", len(hard)), ("retained_random", len(random_ids)),
                                        ("positive", len(positive_set)), ("eligible_unobserved", len(unobserved))):
                        counts[name].append(value)
                    for cutoff in (20, 100):
                        top = {doc_ids[position] for position in unobserved[:cutoff]}
                        denominator = min(cutoff, len(unobserved))
                        for name, item_ids in (("canonical", old_negative_set), ("new", new_negative_set)):
                            coverage[name][cutoff].append(len(top & item_ids) / denominator if denominator else 0.0)
                    for name, item_ids in (("canonical_unobserved", old_negative_set),
                                           ("new_unobserved", new_negative_set), ("positives", positive_set)):
                        score_values[name].extend(float(values[training.position[doc_id]]) for doc_id in item_ids)
                    unobserved_values = values[unobserved]
                    if len(unobserved_values):
                        unobserved_count += len(unobserved_values)
                        unobserved_sum += float(unobserved_values.sum(dtype=np.float64))
                        low, high = float(unobserved_values.min()), float(unobserved_values.max())
                        unobserved_min = low if unobserved_min is None else min(low, unobserved_min)
                        unobserved_max = high if unobserved_max is None else max(high, unobserved_max)
                    ranks = {doc_ids[position]: rank for rank, position in enumerate(allowed, 1)}
                    boundary = float(values[allowed[min(100, len(allowed)) - 1]])
                    for doc_id in positives:
                        positive_ranks.append(ranks[doc_id])
                        boundary_gaps.append(float(values[training.position[doc_id]]) - boundary)
    finally:
        base_model.train(previous_mode)
    require(sha256_file(path) == source_sha, "Canonical pair source changed during mining")
    full_metrics = dense.ranking_metrics(rankings, training.targets)
    full_metrics.update(split="train", query_count=len(query_ids), candidate_count=len(doc_ids),
                        positive_count=sum(len(targets) for targets in training.targets.values()),
                        evaluation_candidate_pool="all_allowed_documents", gold_used_for_scoring=False)
    score_summary = {name: _summary(values) for name, values in score_values.items()}
    score_summary["all_allowed_unobserved"] = {
        "count": unobserved_count, "min": unobserved_min, "max": unobserved_max,
        "mean": unobserved_sum / unobserved_count if unobserved_count else None}
    audit = {"format": FORMAT, "split": "train", "candidate_policy": POLICY,
             "canonical_pairs_sha256": source_sha, "query_count": len(query_ids), "candidate_count": len(doc_ids),
             "candidate_ordering": "sha256(query_id+NUL+doc_id),doc_id_independent_of_labels",
             "mining_score": "frozen_mean_projection_cosine_before_training_temperature",
             "unobserved_items_are_verified_negatives": False, "all_known_positives_retained": True,
             "canonical_and_new_budgets_equal": True, "val_accessed": False, "test_accessed": False,
             "coverage": {name: {"top" + str(k): float(np.mean(values)) for k, values in cutoffs.items()}
                          for name, cutoffs in coverage.items()},
             "coverage_denominator": "macro mean of intersection_size/min(k,eligible_unobserved_count); zero if none",
             "coverage_rank_pool": "all_history_allowed_companies_excluding_all_known_positives_including_retained_random",
             "score_summary": score_summary, "positive_full_pool_rank_summary": _summary(positive_ranks),
             "positive_minus_100th_allowed_score_summary": _summary(boundary_gaps),
             "score_boundary_definition": "score at min(100,allowed_candidate_count); positives remain in allowed pool",
             "base_full_training_metrics": full_metrics}
    audit.update({name + "_count": _summary(values) for name, values in counts.items()})
    return new_pairs, audit, records
