"""Frozen patent encoder + mean evidence pooling + trainable shared projection.

This is a v2 retrieval baseline, not GraphFormers or the proposed contribution
preservation model. Canonical inputs are read-only. Encoding includes existing
CLS/SEP in the masked token mean; padding never contributes. The encoder stays
frozen. Query pooling uses the center and at most five historical patents;
company pooling uses every retained patent (at most twenty).
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from partner_evidence_dataset import EvidenceDataset, integer, require, rows, sha256_file


FORMAT = "partner_dense_frozen_mean_v1"
POOLING = "masked_token_mean_including_existing_cls_sep;query_first6;company_all20"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def source_signature(data_dir):
    directory = Path(data_dir)
    manifest = read_json(directory / "tokenization_manifest.json")
    require(manifest.get("includes_cls_sep") is True and manifest.get("max_length") == 256,
            "Expected canonical tokens with CLS/SEP and max_length 256")
    require(integer(manifest.get("unique_patents")) and manifest["unique_patents"] > 0,
            "Invalid canonical patent count")
    result = {"tokenization_manifest.json": fingerprint(directory / "tokenization_manifest.json")}
    for name, key in (("evidence.jsonl.gz", "source_sha256"),
                      ("evidence.tokens.jsonl.gz", "output_sha256")):
        result[name] = fingerprint(directory / name)
        require(result[name]["sha256"] == manifest.get(key), "Canonical source hash mismatch: " + name)
    return manifest, result


def model_signature(directory, manifest):
    directory = Path(directory).resolve()
    result = {}
    for name, key in (("tokenizer.json", "tokenizer_json_sha256"), ("vocab.txt", "vocab_sha256")):
        result[name] = fingerprint(directory / name)
        require(result[name]["sha256"] == manifest.get(key), "Tokenizer hash mismatch: " + name)
    result["config.json"] = fingerprint(directory / "config.json")
    weights = sorted(set(directory.glob("*.safetensors")) | set(directory.glob("pytorch_model*.bin")))
    require(weights, "No local Hugging Face model weights found")
    for path in weights:
        require(path.is_file() and not path.is_symlink(), "Expected regular model weights")
        result[path.name] = fingerprint(path)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json",
                 "tokenizer_config.json", "special_tokens_map.json"):
        if (directory / name).is_file():
            result[name] = fingerprint(directory / name)
    return result


def choose_device(value):
    device = torch.device(value)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but torch.cuda.is_available() is false")
    return device


def masked_token_mean(hidden, mask):
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    require(bool((weights.sum(1) > 0).all()), "Empty encoder sequence")
    return (hidden * weights).sum(1) / weights.sum(1)


def encode(args):
    # Only this command needs transformers; train/evaluate use cached vectors.
    from transformers import AutoModel
    directory, destination = Path(args.data_dir).resolve(), Path(args.output_dir).resolve()
    require(args.batch_size > 0 and args.max_patents >= 0, "Invalid encoding batch/limit")
    require(not destination.exists(), "Refusing existing embedding output; choose a fresh output directory")
    manifest, sources = source_signature(directory)
    model_dir = Path(args.model_dir).resolve()
    model_sources = model_signature(model_dir, manifest)
    device = choose_device(args.device)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model.eval().requires_grad_(False).to(device)
    require(model.config.pad_token_id is not None, "Encoder must declare pad_token_id")
    target = min(args.max_patents or manifest["unique_patents"], manifest["unique_patents"])
    dimension = model.config.hidden_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="partner_embeddings_", dir=str(destination.parent)))
    start, count, seen = time.monotonic(), 0, set()
    try:
        vectors = np.lib.format.open_memmap(temporary / "vectors.npy", mode="w+", dtype="float32", shape=(target, dimension))
        with (temporary / "index.jsonl").open("w", encoding="utf-8", newline="\n") as index:
            batch = []

            def flush():
                nonlocal count
                if not batch:
                    return
                width = max(len(row["input_ids"]) for row in batch)
                ids = torch.full((len(batch), width), model.config.pad_token_id, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for position, row in enumerate(batch):
                    size = len(row["input_ids"])
                    ids[position, :size] = torch.tensor(row["input_ids"], device=device)
                    mask[position, :size] = 1
                with torch.inference_mode():
                    hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
                    pooled = masked_token_mean(hidden, mask).float().cpu().numpy()
                require(np.isfinite(pooled).all(), "Encoder produced nonfinite vectors")
                vectors[count:count + len(batch)] = pooled
                for row in batch:
                    index.write(json.dumps({"application_number": row["application_number"],
                                            "raw_text_sha256": row["raw_text_sha256"]}, separators=(",", ":")) + "\n")
                count += len(batch)
                if count % (args.batch_size * 100) == 0:
                    print("encoded {}/{} patents".format(count, target), flush=True)
                batch.clear()

            for row in rows(directory / "evidence.tokens.jsonl.gz"):
                require(count + len(batch) < target, "Shared token stream exceeds canonical patent count")
                app, digest, ids = row.get("application_number"), row.get("raw_text_sha256"), row.get("input_ids")
                require(isinstance(app, str) and app and app not in seen, "Invalid/duplicate token application")
                require(isinstance(digest, str) and len(digest) == 64, "Invalid text fingerprint")
                require(isinstance(ids, list) and 2 <= len(ids) <= 256 and
                        all(integer(value) and 0 <= value < model.config.vocab_size for value in ids), "Invalid token IDs")
                require(ids[0] == manifest["cls_token_id"] and ids[-1] == manifest["sep_token_id"], "Invalid CLS/SEP")
                seen.add(app)
                batch.append(row)
                if len(batch) == args.batch_size:
                    flush()
                if args.max_patents > 0 and count + len(batch) == target:
                    break
            flush()
        require(count == target, "Shared token stream ended early")
        vectors.flush()
        del vectors
        require(source_signature(directory)[1] == sources, "Canonical sources changed during encoding")
        require(model_signature(model_dir, manifest) == model_sources, "Encoder files changed during encoding")
        metadata = {"format": FORMAT, "pooling": POOLING, "encoder_frozen": True,
                    "smoke_only": args.max_patents > 0, "patents": count, "dimension": dimension,
                    "dtype": "float32", "source_fingerprints": sources, "model_fingerprints": model_sources,
                    "model_directory_at_encoding": str(model_dir), "torch_version": str(torch.__version__),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": round(time.monotonic() - start, 3),
                    "artifacts": {name: fingerprint(temporary / name) for name in ("vectors.npy", "index.jsonl")}}
        write_json(temporary / "embedding_manifest.json", metadata)
        require(not destination.exists(), "Embedding output appeared during encoding")
        temporary.rename(destination)
        print(json.dumps({"status": "passed", "patents": count, "smoke_only": metadata["smoke_only"],
                          "output_dir": str(destination), "elapsed_seconds": metadata["elapsed_seconds"]}))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class EmbeddingStore:
    def __init__(self, directory, data_dir, allow_smoke=False):
        self.directory = Path(directory).resolve()
        self.metadata = read_json(self.directory / "embedding_manifest.json")
        require(self.metadata.get("format") == FORMAT and self.metadata.get("pooling") == POOLING,
                "Unsupported embedding cache")
        require(allow_smoke or not self.metadata.get("smoke_only"), "Smoke embedding cache cannot train/evaluate formal cohorts")
        manifest, sources = source_signature(data_dir)
        require(self.metadata.get("source_fingerprints") == sources, "Embedding source fingerprint mismatch")
        require(self.metadata.get("encoder_frozen") is True, "Expected frozen encoder cache")
        for name in ("vectors.npy", "index.jsonl"):
            require(fingerprint(self.directory / name) == self.metadata.get("artifacts", {}).get(name),
                    "Embedding artifact fingerprint mismatch: " + name)
        self.vectors = np.load(self.directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        require(self.vectors.shape == (self.metadata["patents"], self.metadata["dimension"]) and
                self.vectors.dtype == np.float32, "Embedding shape/dtype mismatch")
        require(allow_smoke or self.metadata["patents"] == manifest["unique_patents"], "Incomplete formal embedding cache")
        self.lookup = {}
        for position, row in enumerate(rows(self.directory / "index.jsonl")):
            app = row.get("application_number")
            require(isinstance(app, str) and app and app not in self.lookup, "Duplicate/invalid embedding index")
            self.lookup[app] = (position, row.get("raw_text_sha256"))
        require(len(self.lookup) == len(self.vectors), "Embedding/index count mismatch")
        self.manifest_sha256 = sha256_file(self.directory / "embedding_manifest.json")

    def mean(self, references):
        require(references, "No evidence references")
        positions = []
        for ref in references:
            pair = self.lookup.get(ref["application_number"])
            require(pair is not None and pair[1] == ref["raw_text_sha256"], "Evidence embedding missing or text SHA differs")
            positions.append(pair[0])
        result = self.vectors[positions].mean(axis=0, dtype=np.float32)
        require(np.isfinite(result).all(), "Nonfinite pooled evidence")
        return result


class SplitVectors:
    """Historical feature vectors and separate outcome targets; no text/name labels."""
    def __init__(self, data_dir, split, store):
        self.split = split
        require((Path(data_dir) / "dataset_manifest.json").is_file(), "Canonical dataset_manifest.json is required")
        with EvidenceDataset(data_dir, split) as dataset:
            self.fingerprints = dict(dataset.fingerprints)
            self.doc_ids = sorted(dataset.histories)
            self.position = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
            self.documents = torch.from_numpy(np.stack([
                store.mean(dataset.histories[doc_id]["patents"]) for doc_id in self.doc_ids]))
            self.query_ids = sorted(dataset.queries)
            self.queries = torch.from_numpy(np.stack([
                store.mean(dataset.histories[dataset.queries[qid]["source"]]["patents"][:6])
                for qid in self.query_ids]))
            # Targets and exclusion rules never participate in vector construction.
            self.targets = {qid: set(dataset.queries[qid]["positives"]) for qid in self.query_ids}
            self.excluded = {qid: set(dataset.queries[qid]["excluded"]) for qid in self.query_ids}
            self.pairs = {pair["query_id"]: list(pair["candidate_ids"]) for pair in dataset.pairs}


class Projection(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dimension))

    def forward(self, vectors):
        return F.normalize(self.linear(vectors), dim=-1)


def multi_positive_loss(scores, positive_mask):
    require(scores.ndim == 1 and positive_mask.shape == scores.shape and bool(positive_mask.any()),
            "Each query needs at least one positive")
    return -F.log_softmax(scores, dim=0)[positive_mask].mean()


def ranking_metrics(rankings, targets):
    require(set(rankings) == set(targets) and targets, "Every query must be evaluated, including misses")
    totals = {"recall@20": 0.0, "recall@100": 0.0, "Hit@100": 0.0, "P@1": 0.0, "MRR@100": 0.0}
    for qid, gold in targets.items():
        require(gold, "Missing query gold")
        ranking = rankings[qid]
        require(len(ranking) == len(set(ranking)), "Ranking contains duplicate candidates")
        for cutoff in (20, 100):
            totals["recall@" + str(cutoff)] += len(set(ranking[:cutoff]) & gold) / len(gold)
        totals["Hit@100"] += bool(set(ranking[:100]) & gold)
        totals["P@1"] += bool(ranking and ranking[0] in gold)
        totals["MRR@100"] += next((1.0 / rank for rank, doc_id in enumerate(ranking[:100], 1) if doc_id in gold), 0.0)
    return {name: value / len(targets) for name, value in totals.items()}


def evaluate_split(model, split, device, query_batch_size=32):
    require(query_batch_size > 0, "Evaluation batch size must be positive")
    model.eval()
    rankings = {}
    with torch.inference_mode():
        documents = model(split.documents.to(device))
        require(bool(torch.isfinite(documents).all()), "Nonfinite candidate vectors")
        for start in range(0, len(split.query_ids), query_batch_size):
            query_ids = split.query_ids[start:start + query_batch_size]
            queries = model(split.queries[start:start + len(query_ids)].to(device))
            scores = (queries @ documents.T).cpu().numpy()
            require(np.isfinite(scores).all(), "Nonfinite retrieval scores")
            for offset, qid in enumerate(query_ids):
                allowed = np.array([position for position, doc_id in enumerate(split.doc_ids)
                                    if doc_id not in split.excluded[qid]], dtype=np.int64)
                # doc_ids are ascending; stable sorting makes exact ties ID-ordered.
                ordered = allowed[np.argsort(-scores[offset, allowed], kind="stable")[:100]]
                rankings[qid] = [split.doc_ids[position] for position in ordered]
    metrics = ranking_metrics(rankings, split.targets)
    metrics.update(query_count=len(split.query_ids), candidate_count=len(split.doc_ids),
                   evaluation_candidate_pool="all_allowed_documents", gold_used_for_scoring=False,
                   positive_count=sum(len(gold) for gold in split.targets.values()))
    return metrics, rankings


def select_validation_checkpoint(metrics, best_score):
    require(metrics.get("split") == "val", "Checkpoint selection requires validation metrics only")
    score = metrics["recall@100"]
    require(np.isfinite(score), "Invalid validation selection score")
    return score > best_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def train(args):
    require(args.epochs > 0 and args.batch_size > 0 and args.lr > 0 and args.temperature > 0,
            "Invalid training parameters")
    output = Path(args.output_dir).resolve()
    require(not output.exists(), "Refusing existing training output; choose a new run directory")
    device = choose_device(args.device)
    set_seed(args.seed)
    started = time.monotonic()
    print(json.dumps({"stage": "checking_embedding_cache"}), flush=True)
    store = EmbeddingStore(args.embedding_dir, args.data_dir)
    print(json.dumps({"stage": "pooling_training_evidence"}), flush=True)
    training = SplitVectors(args.data_dir, "train", store)
    print(json.dumps({"stage": "pooling_validation_evidence"}), flush=True)
    validation = SplitVectors(args.data_dir, "val", store)
    require(training.documents.shape[1] == validation.documents.shape[1], "Split vector dimension mismatch")
    model = Projection(store.metadata["dimension"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output.mkdir(parents=True)
    configuration = {"format": FORMAT, "pooling": POOLING, "encoder_frozen": True,
                     "trained_component": "shared_identity_initialized_linear_projection_only",
                     "embedding_manifest_sha256": store.manifest_sha256,
                     "source_fingerprints": {"train": training.fingerprints, "val": validation.fingerprints},
                     "selection_split": "val", "selection_metric": "recall@100", "test_accessed": False,
                     "dimension": store.metadata["dimension"], "seed": args.seed, "epochs": args.epochs,
                     "lr": args.lr, "weight_decay": args.weight_decay, "temperature": args.temperature,
                     "batch_size_queries": args.batch_size, "torch_version": str(torch.__version__)}
    write_json(output / "run_config.json", configuration)
    best_score, best_epoch, history = -1.0, -1, []
    # Epoch zero is the untrained frozen-encoder mean-pooling control.
    for epoch in range(args.epochs + 1):
        loss_total = 0.0
        if epoch:
            model.train()
            order = list(range(len(training.query_ids)))
            random.Random(args.seed + epoch).shuffle(order)
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for position in batch:
                    qid = training.query_ids[position]
                    candidate_ids = training.pairs[qid]
                    positives = training.targets[qid]
                    require(positives <= set(candidate_ids), "Training pair drops a positive")
                    candidates = training.documents[[training.position[doc_id] for doc_id in candidate_ids]].to(device)
                    query = training.queries[position:position + 1].to(device)
                    scores = (model(query) @ model(candidates).T).squeeze(0) / args.temperature
                    mask = torch.tensor([doc_id in positives for doc_id in candidate_ids], dtype=torch.bool, device=device)
                    losses.append(multi_positive_loss(scores, mask))
                loss = torch.stack(losses).mean()
                require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * len(batch)
        metrics, _ = evaluate_split(model, validation, device, args.eval_batch_size)
        metrics["split"] = "val"
        record = {"epoch": epoch, "training_loss": loss_total / len(training.query_ids) if epoch else None,
                  "validation": metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if select_validation_checkpoint(metrics, best_score):
            best_score, best_epoch = metrics["recall@100"], epoch
            # Copy weights for serialization; keep the live model and optimizer
            # on their training device throughout validation/checkpoint saves.
            saved_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save({"format": FORMAT, "dimension": store.metadata["dimension"],
                        "state_dict": saved_weights, "configuration": configuration,
                        "selected_epoch": epoch, "selection_split": "val", "validation": metrics}, output / "best.pt")
        write_json(output / "training_history.json", history)
    summary = {"status": "completed", "best_epoch": best_epoch,
               "best_validation_recall@100": best_score, "test_accessed": False,
               "encoder_frozen": True, "innovation_model_trained": False,
               "elapsed_seconds": round(time.monotonic() - started, 3)}
    write_json(output / "training_summary.json", summary)
    print(json.dumps(summary), flush=True)


def evaluate(args):
    output = Path(args.output).resolve()
    require(not output.exists(), "Refusing existing evaluation report")
    device = choose_device(args.device)
    store = EmbeddingStore(args.embedding_dir, args.data_dir)
    split = SplitVectors(args.data_dir, args.split, store)
    model = Projection(store.metadata["dimension"])
    checkpoint_signature = None
    selected_epoch = None
    if args.checkpoint:
        checkpoint_signature = fingerprint(args.checkpoint)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        require(saved.get("format") == FORMAT and saved.get("selection_split") == "val", "Unsupported checkpoint")
        require(saved["configuration"]["embedding_manifest_sha256"] == store.manifest_sha256,
                "Checkpoint/embedding fingerprint mismatch")
        canonical_manifest = split.fingerprints["dataset_manifest.json"]
        require(all(saved["configuration"].get("source_fingerprints", {}).get(name, {}).get("dataset_manifest.json") ==
                    canonical_manifest for name in ("train", "val")), "Checkpoint/canonical dataset manifest mismatch")
        require(saved["configuration"].get("source_fingerprints", {}).get(args.split, split.fingerprints) == split.fingerprints,
                "Checkpoint/split fingerprint mismatch")
        selected_epoch = saved.get("selected_epoch")
        require(integer(selected_epoch) and selected_epoch >= 0, "Invalid checkpoint epoch")
        model.load_state_dict(saved["state_dict"], strict=True)
    metrics, _ = evaluate_split(model.to(device), split, device, args.batch_size)
    metrics.update(format=FORMAT, split=args.split, encoder_frozen=True, pooling=POOLING,
                   trained_projection=bool(args.checkpoint and selected_epoch > 0), selected_epoch=selected_epoch,
                   checkpoint_loaded=bool(args.checkpoint), checkpoint=checkpoint_signature,
                   embedding_manifest_sha256=store.manifest_sha256, input_fingerprints=split.fingerprints,
                   innovation_model=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encoding = commands.add_parser("encode", help="Encode shared patents once with a frozen local base model")
    encoding.add_argument("--data-dir", type=Path, required=True)
    encoding.add_argument("--model-dir", type=Path, required=True)
    encoding.add_argument("--output-dir", type=Path, required=True)
    encoding.add_argument("--batch-size", type=int, default=16)
    encoding.add_argument("--max-patents", type=int, default=0, help="Nonzero creates a smoke-only cache, forbidden for train/evaluate")
    training = commands.add_parser("train", help="Train only the shared projection; select best using full validation pool")
    training.add_argument("--data-dir", type=Path, required=True)
    training.add_argument("--embedding-dir", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--eval-batch-size", type=int, default=32)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--temperature", type=float, default=0.05)
    training.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate", help="Evaluate every query against the complete eligible candidate pool")
    evaluation.add_argument("--data-dir", type=Path, required=True)
    evaluation.add_argument("--embedding-dir", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--split", choices=("val", "test"), required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--batch-size", type=int, default=32)
    for command in (encoding, training, evaluation):
        command.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"encode": encode, "train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
