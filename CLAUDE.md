# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Patton is a research codebase for pretraining language models on **text-rich networks** (ACL 2023). It implements two pretraining strategies: network-contextualized masked language modeling and masked node prediction. The core architecture is **GraphFormers** — BERT extended with graph attention layers inserted between transformer layers to aggregate neighbor node representations.

Downstream tasks: node classification, dense retrieval, reranking, and link prediction.

## Current research: Patent-based enterprise collaboration recommendation

**Goal:** Recommend companies that can collaborate based on patent information.

**Chosen downstream tasks:** Retrieval → Reranking (two-stage pipeline). Link prediction as auxiliary.

**Pipeline:** Patent query → FAISS coarse retrieval (Top-K candidates) → GraphFormers fine reranking → ranked company list.

### Patent → Patton data mapping

| Patent domain | Patton field | Content |
|---|---|---|
| Patent node | `q_text` | Patent title + abstract + claims |
| Company node | `k_text` | Company name + business scope + description |
| Patent neighbors | `q_n_text` (×5) | Same IPC class patents, similar tech patents |
| Company neighbors | `k_n_text` (×5) | Company's patent portfolio, known partners |
| Patent-Company link | positive pair | Known patent assignee/application relationships |
| Non-matching company | negative pair | BM25 hard negatives or random sample |

### Required raw data sources

- Patent DB (title/abstract/claims/IPC) + Company DB (name/business scope) + Assignee relationships
- For neighbors: same-IPC patents, company's patent portfolio, known partnerships

### Three training phases

1. **Pretraining** (`train.jsonl`, `corpus.txt`): `{q_text, k_text, q_n_text, k_n_text}` linked pairs + all node texts for MLM
2. **Retrieval** (`train.text.jsonl`, `documents.json`): `{q_text, q_n_text, positives: [{k_text, k_n_text}], negatives: [...]}` — hard negatives via BM25
3. **Reranking** (`train.rerank.32.jsonl`): same format, more candidates per query (train ~32, test up to 10000)

### Data construction flow

Raw tables → build node texts (node_text.tsv) → build neighbor graph → create link pairs with BM25 negatives → tokenize via `build_train_neg.py` → train

## Environment & dependencies

- Python 3.8, install with `pip3 install -r requirements.txt`
- Key pinned deps: `torch==1.8.0`, `transformers==4.21.1`, `faiss-gpu==1.7.2`
- Run all commands from `src/` directory
- Create `ckpt/` and `logs/` directories at repo root before training

## Running commands

All entry points are shell scripts in `src/`. Edit the `PROJ_DIR` variable in each script to point to your local repo path, then run from `src/`:

**Pretraining:**
```bash
bash run_pretrain.sh          # Patton from bert-base-uncased
bash run_pretrain_sci.sh      # SciPatton from allenai/scibert_scivocab_uncased
```

**Downstream finetuning (edit `$STEP` in test scripts to the best checkpoint step):**

| Task | Train | Test |
|------|-------|------|
| Node classification | `bash nc_class_train.sh` | `bash nc_class_test.sh` |
| Retrieval | `bash nc_retrieve_train.sh` | `bash nc_retrieve_infer.sh` then `bash nc_retrieve_retrieval.sh` |
| Reranking | `bash nc_rerank_train.sh` | `bash nc_rerank_test.sh` |
| Link prediction | `bash lp_train.sh` | `bash lp_test.sh` |

**Direct Python invocation** (used inside scripts):
```bash
# Single-GPU training
CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.train_class --output_dir ... --model_name_or_path ...

# Multi-GPU distributed training
python -m torch.distributed.launch --nproc_per_node=4 --master_port 19298 -m OpenLP.driver.patton_pretrain ...
```

Available driver modules: `patton_pretrain`, `train`, `train_neg`, `train_class`, `test`, `test_class`, `test_rerank`, `infer`, `search`.

## Architecture

### Model hierarchy

```
GraphFormersForLinkPredict  ← entry point, concatenates center + neighbor inputs
  └── GraphFormers (BertPreTrainedModel)
        ├── BertEmbeddings
        └── GraphBertEncoder
              ├── BertLayer × N (standard BERT layers)
              └── GraphAggregation  ← cross-attention: center node [CLS] attends to all neighbor [CLS] tokens
```

The key innovation is in `GraphBertEncoder.forward()`: for every layer after the first, it reshapes hidden states into `(batch, subgraph_nodes, seq_len, hidden)`, runs `GraphAggregation` (a cross-attention where the center node's `[CLS]` queries all neighbor `[CLS]` tokens), then writes the aggregated embedding back into the center node's station position before feeding to the next BERT layer.

### Wrapper models (`modeling.py`)

- **`DenseModel`** — contrastive learning with in-batch negatives; base class for retrieval/link-prediction
- **`DenseLMModel`** — extends `DenseModel` with an MLM head; jointly trains contrastive + MLM loss for pretraining
- **`DenseModelforNCC`** — adds a `LinearClassifier` on top for node classification
- **`DenseRerankModel`** — pairwise scoring for reranking (scores each query against multiple candidates)
- **`DenseModelForInference`** — no-gradient encoding for index building and search

### Model registry

`src/OpenLP/models/__init__.py` maps model type string to class:
```python
AutoModels = {'graphformer': GraphFormersForLinkPredict}
```
All models are loaded via `AutoModels[model_args.model_type].from_pretrained(...)`.

### Loss functions (`loss.py`)

- `SimpleContrastiveLoss` — standard in-batch softmax cross-entropy
- `DistributedContrastiveLoss` — gathers representations across all GPUs before computing loss (used with `--negatives_x_device`)

### Data flow

1. Raw data: JSONL files where each row is a linked node pair with fields `q_text`, `k_text`, `q_n_text` (neighbor texts), `k_n_text` (neighbor texts)
2. Pre-tokenization scripts in `src/scripts/build_train*.py` convert raw JSONL to tokenized format for faster training
3. Dataset classes in `dataset/` load tokenized data and apply task-specific collation (MLM masking in `TrainLMCollator`, hard negative sampling in `TrainHnCollator`, etc.)
4. Custom `DenseTrainer` (extends Hugging Face `Trainer`) handles the training loop with optional gradient caching (`GCDenseTrainer`)

### Key configuration flags

- `--model_type graphformer` — selects the GraphFormers architecture
- `--mlm_loss True` — enables joint contrastive + MLM pretraining
- `--neighbor_mask_ratio` — probability of masking/corrupting neighbors during pretraining
- `--negatives_x_device` — share negative representations across GPUs for larger effective batch
- `--grad_cache` — enable gradient checkpointing to reduce GPU memory
- `--hn_num` — number of hard negatives per query (for retrieval training)

## Data

Processed datasets are downloaded separately from Google Drive (see README). The `data/` directory contains domain-specific subdirectories (Economics, Geology, Mathematics, sports, cloth), each with task-specific splits under `nc/`, `nc-coarse/`, `link_prediction/`, and `sci-pretrain/`.

## No tests or linting

This repository has no unit tests, no linting configuration, and no CI/CD. Evaluation is done via the driver test scripts (`test.py`, `test_class.py`, `test_rerank.py`) which run inference and compute metrics (P@1, MRR, NDCG, accuracy, recall, F1 via `utils.py`).
