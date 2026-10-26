"""Uncompressed M<=20 evidence-max reference using the original frozen cache.

Query = L2-normalized raw mean of the source's first six historical patents.
Company score = maximum cosine over its separately normalized real patents.
No learned projection, BERT loading, training, or FAISS index is involved.
This reference retains more vectors than the mean baseline; it is neither a
fixed-budget comparison nor a mathematical performance upper bound.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from partner_dense_baseline import EmbeddingStore, choose_device, ranking_metrics
from partner_evidence_dataset import EvidenceDataset, require


FORMAT = "partner_frozen_evidence_max_v1"
POOLING = "query_raw_mean_first6_l2;company_max_cosine_over_each_real_patent"


class CompleteEvidenceVectors:
    def __init__(self, data_dir, split, store):
        require((Path(data_dir) / "dataset_manifest.json").is_file(), "Canonical dataset_manifest.json is required")
        with EvidenceDataset(data_dir, split) as dataset:
            self.fingerprints = dict(dataset.fingerprints)
            self.doc_ids = sorted(dataset.histories)
            self.dimension = store.metadata["dimension"]
            self.evidence = torch.zeros(len(self.doc_ids), 20, self.dimension, dtype=torch.float32)
            self.mask = torch.zeros(len(self.doc_ids), 20, dtype=torch.bool)
            for position, doc_id in enumerate(self.doc_ids):
                for slot, ref in enumerate(dataset.histories[doc_id]["patents"]):
                    entry = store.lookup.get(ref["application_number"])
                    require(entry is not None and entry[1] == ref["raw_text_sha256"],
                            "Evidence embedding missing or text SHA differs")
                    vector = store.vectors[entry[0]]
                    require(np.isfinite(vector).all(), "Nonfinite cached evidence vector")
                    self.evidence[position, slot] = torch.from_numpy(np.array(vector, copy=True))
                    self.mask[position, slot] = True
            self.query_ids = sorted(dataset.queries)
            self.queries = torch.from_numpy(np.stack([
                store.mean(dataset.histories[dataset.queries[qid]["source"]]["patents"][:6])
                for qid in self.query_ids]))
            # Future relations remain separate evaluation targets/exclusion rules.
            self.targets = {qid: set(dataset.queries[qid]["positives"]) for qid in self.query_ids}
            self.excluded = {qid: set(dataset.queries[qid]["excluded"]) for qid in self.query_ids}
        self.evidence_count = int(self.mask.sum().item())


def max_evidence_scores(queries, normalized_evidence, evidence_mask):
    """Normalize queries and score pre-normalized evidence, masking empty slots."""
    require(queries.ndim == 2 and normalized_evidence.ndim == 3 and
            queries.shape[-1] == normalized_evidence.shape[-1], "Invalid vector shapes")
    require(evidence_mask.dtype == torch.bool and evidence_mask.shape == normalized_evidence.shape[:2] and
            bool(evidence_mask.any(dim=1).all()), "Each candidate requires real evidence")
    flat = normalized_evidence.reshape(-1, normalized_evidence.shape[-1])
    scores = (F.normalize(queries, dim=-1) @ flat.T).reshape(
        len(queries), normalized_evidence.shape[0], normalized_evidence.shape[1])
    scores.masked_fill_(~evidence_mask.unsqueeze(0), -torch.inf)
    return scores.max(dim=-1).values


def evaluate_split(split, device, batch_size=16):
    require(batch_size > 0 and split.doc_ids == sorted(set(split.doc_ids)), "Invalid batch size or candidate IDs")
    rankings = {}
    with torch.inference_mode():
        # One padded evidence payload on the scoring device; normalize in place.
        evidence = split.evidence.to(device, copy=True)
        require(bool(torch.isfinite(evidence).all()), "Nonfinite evidence vectors")
        evidence.div_(evidence.norm(dim=-1, keepdim=True).clamp_min_(1e-12))
        mask = split.mask.to(device)
        for start in range(0, len(split.query_ids), batch_size):
            qids = split.query_ids[start:start + batch_size]
            scores = max_evidence_scores(split.queries[start:start + len(qids)].to(device), evidence, mask).cpu().numpy()
            require(np.isfinite(scores).all(), "Nonfinite retrieval scores")
            for offset, qid in enumerate(qids):
                allowed = np.array([index for index, doc_id in enumerate(split.doc_ids)
                                    if doc_id not in split.excluded[qid]], dtype=np.int64)
                # Ascending IDs + stable sort fixes ties without consulting gold.
                order = allowed[np.argsort(-scores[offset, allowed], kind="stable")[:100]]
                rankings[qid] = [split.doc_ids[index] for index in order]
    metrics = ranking_metrics(rankings, split.targets)
    return metrics, rankings


def evaluate(args):
    started, output = time.monotonic(), Path(args.output).resolve()
    require(not output.exists(), "Refusing existing evaluation report")
    require(args.batch_size > 0, "Evaluation batch size must be positive")
    device = choose_device(args.device)
    print(json.dumps({"stage": "checking_original_embedding_cache"}), flush=True)
    store = EmbeddingStore(args.embedding_dir, args.data_dir)
    print(json.dumps({"stage": "gathering_all_historical_evidence", "split": args.split}), flush=True)
    split = CompleteEvidenceVectors(args.data_dir, args.split, store)
    print(json.dumps({"stage": "scoring_complete_candidate_pool", "evidence_vectors": split.evidence_count}), flush=True)
    metrics, _ = evaluate_split(split, device, args.batch_size)
    item_bytes, dimensions = split.evidence.element_size(), split.dimension
    metrics.update(
        format=FORMAT, split=args.split, encoder_frozen=True, trained_projection=False, innovation_model=False,
        pooling=POOLING, checkpoint_loaded=False, gold_used_for_scoring=False,
        query_count=len(split.query_ids), candidate_count=len(split.doc_ids),
        positive_count=sum(len(value) for value in split.targets.values()),
        evaluation_candidate_pool="all_allowed_documents", evidence_vector_count=split.evidence_count,
        evidence_vectors_per_company_cap=20, vector_dimension=dimensions, vector_dtype="float32",
        logical_evidence_vector_bytes=split.evidence_count * dimensions * item_bytes,
        allocated_candidate_tensor_bytes=split.evidence.numel() * item_bytes,
        allocated_mask_tensor_bytes=split.mask.numel() * split.mask.element_size(),
        allocated_query_tensor_bytes=split.queries.numel() * split.queries.element_size(),
        max_batch_score_tensor_bytes=min(args.batch_size, len(split.query_ids)) * len(split.doc_ids) * 20 * item_bytes,
        allocation_scope="tensor_payloads_including_padding_not_measured_process_or_cuda_peak",
        candidate_storage="uncompressed_up_to20_vectors_per_company_vs_one_mean_vector",
        retrieval_implementation="exhaustive_cached_vector_scoring_no_faiss_index",
        fixed_index_budget_comparison=False, performance_upper_bound_claimed=False,
        embedding_manifest_sha256=store.manifest_sha256, input_fingerprints=split.fingerprints,
        elapsed_seconds=round(time.monotonic() - started, 3))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("evaluate", help="Evaluate the original frozen full-evidence max reference")
    command.add_argument("--data-dir", type=Path, required=True)
    command.add_argument("--embedding-dir", type=Path, required=True)
    command.add_argument("--split", choices=("val", "test"), required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device", default="cuda")
    command.add_argument("--batch-size", type=int, default=16)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
