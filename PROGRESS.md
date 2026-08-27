# 实验进度（专利→企业合作推荐）

> 本文件记录实验流水线的当前状态，跨会话维护。新开会话先读这里。
> 最近更新：2026-08-27

## 当前状态：② 检索训练已跑完，但因训练数据标签泄漏**作废**，需改数据重跑

| 阶段 | 状态 | 产物 | 指标 |
|---|---|---|---|
| ① 预训练（MLM+对比，中文底座→专利语料） | ✅ 2026-08-25 完成（30h） | `ckpt/patent/pretrain/graphformer/1e-5/` | train_loss 1.29；val MRR **0.83** / NDCG@10 **0.87**。**不受泄漏影响**（q/k 两侧都有邻居，纯 in-batch 负例） |
| ② 检索训练（第 1 版） | ⚠️ 2026-08-27 正常跑完 1231 步（9h54m），但**结果作废** | `ckpt/patent/nc_retrieval/graphformer/1e-5/` | train_loss 0.136（末 200 步 <5e-5）；val P@1 0.39 / MRR 0.63 —— **均为泄漏产物，不可引用** |
| ②' 修数据 → 重跑检索训练 | 🔄 待执行（~10h + tokenize 时间） | 同上（会覆盖，先 scp 存档） | |
| ③ 检索推理 → FAISS 检索 + recall | ⏳ | logs + trec 结果 | `nc_retrieve_infer_patent.sh` → `nc_retrieve_retrieval_patent.sh` |
| ④ 重排训练（100k 样本，~6h） | ⏳ 数据同样需先修 | `ckpt/patent/nc_rerank/` | 底座用 ②' 的 checkpoint |
| ⑤ 重排测试 | ⏳ | P@1/MRR/NDCG | 需把 `nc_rerank_test_patent.sh` 的 STEP 改成最佳存档步数 |
| ⑥ 结果回传本地 | ⏳ 部分 | 本地 `ckpt/patent/` | 本地目前只有 `pretrain`；②的泄漏版 checkpoint 建议先拉回当对照组 |

## 事故：正例/负例邻居不对称 → 确定性标签泄漏（2026-08-27 发现）

**现象**：检索训练末段 `loss: 0.0`（HF 是 round 到 4 位，即 <5e-5），而验证 P@1 只有 0.39。

**根因**：`*.text.jsonl` 里**正例的 `k_n_text` 有真实内容，负例的五条全是空串**。而邻居 mask 是模型的显式输入通道
（`train_dataset.py:128` / `inference_dataset.py:55` 都是 `1 if 非空 else 0`），正例 mask=`[1,1,1,1,1]`、负例 mask=`[0,0,0,0,0]`，
模型不需要任何语义理解，读 mask 就能 100% 分类。

**数值佐证**（假设「有邻居的排前面」，验证时 20 候选里恰好 4 个带邻居）：预测 P@1≈1/4、MRR≈0.62、
NDCG@10 必然**等于** NDCG@100；实测 0.392 / 0.632 / 两者小数点后 16 位完全相同。三项吻合。

**第二重问题**：FAISS 建索引用 `documents.txt`（只有 `id,text` 两列），`inference_dataset.py:53` 的
`if 'n_text' in example` 走不进去 → 25,598 家企业**全部零邻居入库**。即模型最依赖的特征在推理时恒等于「负例」取值，
训练/推理口径根本对不上。

**上游原版是怎么做的**：`src/scripts/bm25_neg.py:122-123` 中正例和负例**都**写 `k_n_text: [""]`——
原版 Patton 检索阶段 k 侧两边都不给邻居，与 `documents.txt` 无邻居一致；图结构只走**查询侧** `q_n_text` + 预训练权重。
专利管线单方面给正例填了邻居，一次性制造了泄漏和不一致。

**修法（对齐上游）**：把正例的 `k_n_text` 也置空，k 侧全员无邻居。企业侧图谱知识由**预训练**承载（那边不受影响）。
脚本：`data_pipeline/fix_kn_leak.py`（默认 dry-run 只统计，`--apply` 才改写；负例若意外带邻居会自动中止不改文件）。

> **注意：空邻居仍占一个子图序列槽**（`train_dataset.py:127` 用 `create_one_example([0])` 占位），
> 所以置空必须保留原有的 5 个槽位，不能把列表清成 `[]`，否则子图张量形状对不上。

**影响面**：`nc/` 下 5 个文件——`train.text.jsonl`、`val.text.jsonl`、`train.rerank.32.text.jsonl`、
`val.rerank.32.text.jsonl`、`test.rerank.10000.text.jsonl`。**重排的测试集也泄漏**，若不修，最终报的
P@1/MRR/NDCG 是虚高的。查询侧 `test.node.text.jsonl`（`id,text,n_text`）**不动**——它与推理端一致，是对的。

## 生效配置（12G 3080 Ti 硬约束，勿改回）

- 数据：检索训练用 `train.half.jsonl`（315k 抽 50%）；验证用 `val.small.jsonl` / `val.rerank.small.jsonl`（323k 抽 1/32 ≈ 1 万条——全量验证一次要 3-4h）
- **k 侧（企业侧）不给邻居是刻意的**，别再往回填：既为消除泄漏，也为对齐 `documents.txt` 的推理口径
- `max_len 256` + **batch 1 × 累积 128** = 有效 batch 128（每样本 36 条子图序列 × ~175MB/条，batch 2 也 OOM）
- 验证 batch 4（16 会 OOM）；`save_steps 500` < `eval_steps 1000`（HF 顺序 log→eval→save，验证 OOM 会连存档一起丢）
- **不用 grad_cache**（max_len 256 下前向缓存一样爆，且拖慢）；脚本统一 `python -u`（stdout 无缓冲，loss 实时落盘）
- 预训练保持 batch 4 × 累积 32；检索/重排训练一律 batch 1 × 累积 128

> **batch 1 的副作用**：梯度累积**不产生 in-batch 负例**（每个 micro-batch 各算各的 loss），
> 所以对比难度始终只有 1 正 + `hn_num` 负 = 5 路，远低于原版设计。这也是训练 loss 容易压到很低的结构性原因之一。

## 踩坑记录（均已修复在脚本里）

1. 原版 batch 按 max_len 32 设计；256 下注意力矩阵按长度平方增长，显存 8 倍
2. grad_cache 只省反向重算，不省前向缓存
3. nohup 下 loss 行（tqdm.write→stdout）块缓冲，攒到进程结束才落盘；tensorboard 里其实一直有 → `python -u`
4. 存档在验证之后执行：eval OOM 时 9.4h 训练零存档 → save_steps 500
5. 全量 323k 验证集一次 3-4h → 抽 1/32
6. **正例填邻居、负例留空 = 标签泄漏**（详见上一节）。教训：正负例除标签外必须**构造方式完全一致**；
   训练时喂的字段也必须和推理时能拿到的字段一致。训练 loss 掉到 0 不是好消息，先怀疑泄漏。

## 服务器操作速查（/workspace/Patton）

```bash
conda activate patton
git pull                                  # 拿最新脚本（每次重启训练前）
tail -f retrieve.log                      # 盯训练（loss 每 100 步一条）
grep eval_loss retrieve.log               # 查验证指标
ps aux | grep OpenLP | grep -v grep       # 确认在跑
nohup bash src/nc_retrieve_train_patent.sh > retrieve.log 2>&1 &   # ② 检索训练
nohup bash src/nc_rerank_train_patent.sh > rerank.log 2>&1 &       # ④ 重排训练
```

### 修泄漏 → 重跑 ② 的完整流程

```bash
# 0) 先把泄漏版 checkpoint 拉回本地当对照组（本地 Git Bash 执行），重跑会覆盖它
scp -rP 23 root@<外网IP>:/workspace/Patton/ckpt/patent/nc_retrieval ckpt/patent/nc_retrieval_leaked

# 1) 改数据（服务器）：先 dry-run 看统计，确认「正例 100% 有邻居 / 负例 0%」再执行
cd /workspace/Patton && git pull
python data_pipeline/fix_kn_leak.py --dir data/patent/nc
python data_pipeline/fix_kn_leak.py --dir data/patent/nc --apply

# 2) 重新 tokenize（只跑 nc 部分，预训练数据不受影响、别浪费时间）
cd /workspace/Patton/src/scripts
TOK=/workspace/Patton/ckpt/chinese-roberta-wwm-ext
NC=/workspace/Patton/data/patent/nc
python build_train_neg.py --input_dir $NC --output $NC --tokenizer $TOK --max_length 256 --mp_workers 8
python build_train_neg.py --input_dir $NC --output $NC --tokenizer $TOK --max_length 256 --mp_workers 8 --prefix rerank.32
python build_train_neg.py --input_dir $NC --output $NC --tokenizer $TOK --max_length 256 --mp_workers 8 --prefix rerank.10000

# 3) 重建抽样文件（tokenize 会覆盖 train.jsonl/val.jsonl，抽样必须重做）
cd $NC
awk 'NR % 2 == 0'  train.jsonl           > train.half.jsonl
awk 'NR % 32 == 0' val.jsonl             > val.small.jsonl
awk 'NR % 32 == 0' val.rerank.32.jsonl   > val.rerank.small.jsonl

# 4) 重跑检索训练
cd /workspace/Patton
nohup bash src/nc_retrieve_train_patent.sh > retrieve.log 2>&1 &
```

> 重跑后的**健康信号**：训练 loss 不应再掉到 0.0；验证 `ndcg_10` 与 `ndcg_100` 应当**不再完全相等**。
> 若这两条仍出现，说明还有别的泄漏路径，别急着往下走。

## 结果回传（本地 Git Bash）

```bash
scp -rP 23 root@<外网IP>:/workspace/Patton/ckpt/patent/pretrain ckpt/patent/      # 预训练
scp -rP 23 root@<外网IP>:/workspace/Patton/ckpt/patent/nc_retrieval ckpt/patent/  # 检索
scp -rP 23 root@<外网IP>:/workspace/Patton/ckpt/patent/nc_rerank ckpt/patent/     # 重排
scp -rP 23 root@<外网IP>:/workspace/Patton/logs/patent logs/patent/
```

## 重要提醒

- 优云智算按量计费：**关机后 7 天实例释放、系统盘清空**——每阶段结束立刻 scp 结果回本地
- 服务器系统盘 178G；每阶段训练前 `rm -rf ~/.cache/huggingface/datasets` 可释放 ~58G
- tokenize 步骤（`build_train_neg.py`）是纯文件 IO、缺文件会自动跳过、不走 HF 缓存；但**训练**侧
  `TrainHnDataset` 用 `load_dataset("json",...)` 有缓存，重跑训练前清 `~/.cache/huggingface/datasets` 更稳妥（顺带释放 ~58G）
- 检索训练 ~4h 存一次档（500 步）；中途崩溃后从零重跑（`train_neg.py:106` 是裸的 `trainer.train()`，
  命令行传 `--resume_from_checkpoint` 会被解析但**静默忽略**，要 resume 得改代码）
- 本地 `data/patent/nc/*.text.jsonl` 是服务器数据的干净副本（未修改），需要回滚时从本地重传
