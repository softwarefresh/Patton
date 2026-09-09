# 实验进度（专利→企业合作推荐）

> 本文件记录实验流水线的当前状态，跨会话维护。新开会话先读这里。
> 最近更新：2026-09-05

## 当前状态：⑧ 检索 v2 负结果（recall 反降 6-7 点）→ 保留 ②'；待办：scp 收尾 + 关机 + 写论文

| 阶段 | 状态 | 产物 | 指标 |
|---|---|---|---|
| ① 预训练（MLM+对比，中文底座→专利语料） | ✅ 2026-08-25 完成（30h） | `ckpt/patent/pretrain/graphformer/1e-5/` | train_loss 1.29；val MRR **0.83** / NDCG@10 **0.87**。**不受泄漏影响** |
| ② 检索训练（泄漏版） | ❌ 作废，已存档 | 本地 `ckpt/patent/nc_retrieval_leaked/`（scp 于 08-28） | P@1 0.39 / MRR 0.63 —— 泄漏灌水，仅作对照组 |
| ②' 检索训练（修复版） | ✅ 2026-08-28 完成（10h02m，1231 步） | `ckpt/patent/nc_retrieval/graphformer/1e-5/` | eval_loss 1.30；**P@1 0.60 / MRR 0.75 / NDCG@10 0.81**（真实值，且高于泄漏版） |
| ③ 检索推理 → FAISS 检索 + recall | ✅ 2026-09-07 重跑完成（`retrieve2.log`） | `node_label_embed/` + trec 结果 | **稠密 recall@50 0.343 / @100 0.432**（n=318,596）；BM25 本地补算（`data_pipeline/bm25_recall.py`，与训练负例同口径）recall@20 0.013 / @50 0.022 / @100 0.034 / @200 0.051 —— **粗筛必须用稠密**；BM25 仅 12 倍弱于稠密，只配当负例来源 |
| ④ 重排训练（100k 样本） | ✅ 完成（08-29，6h16m，781 步） | `ckpt/patent/nc_rerank/graphformer/1e-5/checkpoint-500` | 底座用 ②' 的 checkpoint；train_loss 0.2087；**无训练中 eval**（781 步 < eval_steps 1000） |
| ⑤ 重排测试 | ✅ 2026-09-05 完成（38h41m，159,298 步） | `rerank_test.log` | 全量 **P@1 0.817 / MRR 0.876 / NDCG@5 0.891 / NDCG@10 0.901**（冒烟 0.842/0.893/0.914，全量略降正常——test 每查询最多 10000 候选） |
| ⑥ 结果回传本地 | ✅ 完成（2026-09-06） | 本地 `ckpt/patent/` + `logs/patent/` | pretrain/检索/重排 ckpt 与日志全部到齐；服务器 tokenize 数据（33GB）未回传（本地有文本版可重建） |
| ⑦ 消融：去预训练 | ✅ 2026-09-07 完成（781 步，train_loss 1.442 vs 主 0.209） | `ckpt/patent/nc_rerank_nopretrain/` | val.rerank.small 同文件对比：主模型 P@1 0.830 / MRR 0.886 / NDCG@10 0.910 vs 消融 P@1 0.766 / MRR 0.838 / NDCG@10 0.870 → **+6.4 P@1 点**（消融同时去掉 ① 预训练 + ②' 检索训练两段；单独归因需再补「预训练→重排」中间版） |
| ⑧ 检索 v2（in-batch 负例） | ❌ 负结果（2026-09-08，batch 4@128+hn3 实测） | `ckpt/patent/nc_retrieval_v2/`（可删） | recall@50 0.277 / @100 0.363 vs ②' 0.343 / 0.432（**−6.9**）——max_len 128 截断损失盖过 in-batch 负例增益；**教训：查询文本长度主导检索质量**。保留 ②'，级联重训不启动 |

## ③ 的重要注意（pkl 缓存陷阱）

`search.py:77`：若 `data/patent/nc/patent_patent_retrieval_dict.pkl` 已存在，search 会**直接读旧结果、跳过新检索**。
所以每次换模型/换索引重跑 ③，必须确认该 pkl 不存在（脚本正常结束时末尾会 rm；中途失败的不会）。

## 两阶段叙事要点（2026-09-07，写论文前先读）

- **端到端天花板 = 稠密 recall@100 0.432**：重排再好，粗筛漏掉的就救不回来。
- 重排 P@1 0.817 的设定是「1 正例 + 20 BM25 干扰项」的排序任务，**与 FAISS 无关**，论文里必须写清。
- 检索弱的可解释根因之一：batch 1×累积 128 训练**没有 in-batch 负例**（5 路对比），见「生效配置」。
  **已尝试修复（⑧，2026-09-08）**：max_len 128 换 batch 4 恢复 in-batch 负例 + hn 3（batch 4@128+hn4 实测 OOM 10.9G），
  结果 recall@50/100 反降 6-7 点 → **文本长度主导检索质量，in-batch 负例救不回来**。后续如需再试：双卡（`--negatives_x_device`）
  或更大显存保持 256 长度；否则检索瓶颈留作 future work。
- 论文建议结构：检索阶段如实报 recall（稠密 vs BM25 对比表本身就是一个有信息量的结果）；
  核心贡献 = 重排精度（P@1 0.817）+ 预训练增益（+6.4 P@1 点）；检索改进列为 future work 或后续实验。

## 事故一：正例/负例邻居不对称 → 确定性标签泄漏（2026-08-27 发现，已修复）

**现象**：检索训练末段 `loss: 0.0`（HF round 4 位，即 <5e-5），而验证 P@1 只有 0.39。

**根因**：`*.text.jsonl` 里**正例的 `k_n_text` 有真实内容，负例的五条全是空串**。而邻居 mask 是模型的显式输入通道
（`train_dataset.py:128` / `inference_dataset.py:55` 都是 `1 if 非空 else 0`），正例 mask=`[1,1,1,1,1]`、负例 mask=`[0,0,0,0,0]`，
模型读 mask 即可 100% 分类。

**数值佐证**（假设「有邻居的排前面」，验证时 20 候选里恰好 4 个带邻居）：预测 P@1≈1/4、MRR≈0.62、
NDCG@10 必然**等于** NDCG@100；实测 0.392 / 0.632 / 两者小数点后 16 位完全相同。三项吻合。

**修法（对齐上游）**：k 侧全员无邻居。企业侧图谱知识由**预训练**承载（那边 q/k 两侧都有邻居、纯 in-batch 负例，不受影响）。
三处修改：
1. `build_patent_data.py`（根）：positives 的 `k_n_text` 不再 `kn.get(...)`，与 negatives 同为 `[''] * N_NB`
2. `fix_kn_leak.py`：对已生成文件（文本版 `.text.jsonl` 或 tokenize 后 `.jsonl` 均可）原地清空正例邻居；
   默认 dry-run，`--apply` 才写；负例若意外带邻居自动中止
3. tokenize 后格式：空串编码成 `[]`（`build_train_neg.py` 用 add_special_tokens=False），与 `k_n != []` 判定一致

> **空邻居仍占一个子图序列槽**（`train_dataset.py:127` 用 `create_one_example([0])` 占位），置空必须保留 5 个槽位。

**影响面**：检索+重排的 train/val/test 全部 8 个 `.jsonl`；预训练不受影响。
**修复版结果（对比泄漏版）**：eval P@1 0.39→**0.60**、MRR 0.63→**0.75**、NDCG@10 0.72→**0.81**；
`ndcg_10 ≠ ndcg_100`（0.8116 vs 0.8133）= 泄漏根除的判据。认真学语义比作弊还强。

## 事故二：经营范围含 \t/\n → TSV 记录劈裂（2026-08-28 发现，已修复）

**现象**：③ 建索引报 `ParserError: Expected 2 fields in line 4251, saw 3`。

**根因**：爱企查经营范围字段本身混有制表符和换行符（25,598 家里 **1,831 家**带 `\n`、3 家带 `\t`），
`stage_documents` 直接 `f.write(f"{id}\t{contents}\n")`，一条记录被劈成多行：
`documents.txt` 27,429 行（应为 25,598）；`node_text.tsv`、`test.node.text.tsv` 同病（本地已统计确认）。

**修法**：
- 服务器（已执行）：从 `documents.json` 重建 `documents.txt`——JSON 里换行是转义符不劈行，
  重建时文本内 `\t`/`\n` 替换为空格，得到 25,598 行、每行恰好 1 个制表符
- 本地（已提交 e23e4da）：`stage_documents` 写 documents.json/txt 时清洗；`test.node.text.tsv` 同样处理。
  清洗只在写文件层，doc_map 的 k_text 保持原样（训练端已 tokenize，不引入差异）
- `node_text.tsv` 未动（③ 不读它；legacy 用途，日后用到再清洗）

## 生效配置（12G 3080 Ti 硬约束，勿改回）

- 数据：检索训练用 `train.half.jsonl`（315k 抽 50%）；验证用 `val.small.jsonl` / `val.rerank.small.jsonl`（323k 抽 1/32 ≈ 1 万条——全量验证一次要 3-4h）
- **k 侧（企业侧）不给邻居是刻意的**，别再往回填：既为消除泄漏，也为对齐 `documents.txt` 的推理口径
- `max_len 256` + **batch 1 × 累积 128** = 有效 batch 128（每样本 36 条子图序列 × ~175MB/条，batch 2 也 OOM）
- 验证 batch 4（16 会 OOM）；`save_steps 500` < `eval_steps 1000`（HF 顺序 log→eval→save，验证 OOM 会连存档一起丢）
- **不用 grad_cache**（max_len 256 下前向缓存一样爆，且拖慢）；脚本统一 `python -u`（stdout 无缓冲，loss 实时落盘）
- 预训练保持 batch 4 × 累积 32；检索/重排训练一律 batch 1 × 累积 128

> **batch 1 的副作用**：梯度累积**不产生 in-batch 负例**（每个 micro-batch 各算各的 loss），
> 对比难度始终只有 1 正 + `hn_num` 负 = 5 路，远低于原版设计。训练 loss 容易压低的另一结构性原因。

## 踩坑记录（均已修复在脚本里）

1. 原版 batch 按 max_len 32 设计；256 下注意力矩阵按长度平方增长，显存 8 倍
2. grad_cache 只省反向重算，不省前向缓存
3. nohup 下 loss 行（tqdm.write→stdout）块缓冲，攒到进程结束才落盘；tensorboard 里其实一直有 → `python -u`
4. 存档在验证之后执行：eval OOM 时 9.4h 训练零存档 → save_steps 500
5. 全量 323k 验证集一次 3-4h → 抽 1/32
6. **正例填邻居、负例留空 = 标签泄漏**。教训：正负例除标签外必须**构造方式完全一致**；
   训练时喂的字段必须和推理时能拿到的字段一致。训练 loss 掉到 0 不是好消息，先怀疑泄漏
7. **写 TSV 前必须清洗字段内的 \t/\n**。教训：CSV 列内嵌分隔符/换行是经典坑，
   任何 `f.write(f"{a}\t{b}")` 式输出都要先 sanitize；JSON 反而安全（转义）
8. search 有 pkl 结果缓存（search.py:77），换模型重跑要先删

## 服务器操作速查（/workspace/Patton）

```bash
conda activate patton
git pull                                  # 拿最新脚本（每次重启训练前）
tail -f retrieve.log                      # 盯训练（loss 每 100 步一条）
grep eval_loss retrieve.log               # 查验证指标
ps aux | grep OpenLP | grep -v grep       # 确认在跑
ls data/patent/nc/*retrieval_dict.pkl     # ③ 重跑前确认无旧 pkl
nohup bash src/nc_retrieve_train_patent.sh > retrieve.log 2>&1 &   # ② 检索训练
nohup bash src/nc_rerank_train_patent.sh > rerank.log 2>&1 &       # ④ 重排训练
bash src/nc_retrieve_infer_patent.sh      # ③ 建索引（documents.txt 25,598 家）
bash src/nc_retrieve_retrieval_patent.sh  # ③ 检索 + recall@50/100
```

### ⑤ 重排全量评测：✅ 完成（2026-09-05，38h41m）

最终指标（`grep eval_prc rerank_test.log`）：P@1 **0.817** / MRR **0.876** / NDCG@5 **0.891** / NDCG@10 **0.901**。
对比 ②' 检索（P@1 0.60 / MRR 0.75 / NDCG@10 0.81）提升明显——两阶段管线（FAISS 粗筛 → 重排精排）生效。
回传已完成（2026-09-06）：`ckpt/patent/nc_rerank`（1.6G，checkpoint-500 完整含 optimizer/scheduler）+ `logs/patent/` 三个日志。
注意：`retrieve.log` 实为 ②' 检索**训练**日志（末次 eval：prc 0.597 / mrr 0.752 / ndcg@10 0.812，与表一致）；
③ FAISS recall 前台运行未落日志，数值只能从服务器 trec 结果文件补算（`eval_trec.py`，纯 CPU）。

### 修泄漏（若日后重建数据后仍需修 tokenize 文件）

```bash
python data_pipeline/fix_kn_leak.py --dir data/patent/nc            # dry-run 看统计
python data_pipeline/fix_kn_leak.py --dir data/patent/nc --apply    # 原地改写
```

## 结果回传（本地 Git Bash）

```bash
scp -rP 23 root@<外网IP>:/workspace/Patton/ckpt/patent/pretrain ckpt/patent/      # 预训练
scp -rP 23 root@<外网IP>:/workspace/Patton/ckpt/patent/nc_retrieval ckpt/patent/  # 检索(修复版)
scp -rP 23 root@<外网IP>:/workspace/Patton/ckpt/patent/nc_rerank ckpt/patent/     # 重排
scp -rP 23 root@<外网IP>:/workspace/Patton/logs/patent logs/patent/
```

## 重要提醒

- 优云智算按量计费：**关机后 7 天实例释放、系统盘清空**——每阶段结束立刻 scp 结果回本地
- 服务器系统盘 178G；训练前 `rm -rf ~/.cache/huggingface/datasets` 可释放 ~29G（实测）
- tokenize 步骤（`build_train_neg.py`）是纯文件 IO、缺文件会自动跳过、不走 HF 缓存；训练侧
  `TrainHnDataset` 用 `load_dataset("json",...)` 有缓存，重跑前清更稳妥
- 检索训练 ~4h 存一次档（500 步）；中途崩溃后从零重跑（`train_neg.py:106` 是裸的 `trainer.train()`，
  命令行传 `--resume_from_checkpoint` 会被解析但**静默忽略**，要 resume 得改代码）
- 本地 `data/patent/nc/*.text.jsonl` 是服务器数据的干净副本（未修改），需要回滚时从本地重传
