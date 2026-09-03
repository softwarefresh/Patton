# 实验进度（专利→企业合作推荐）

> 本文件记录实验流水线的当前状态，跨会话维护。新开会话先读这里。
> 最近更新：2026-08-28

## 当前状态：③ 检索推理 + FAISS 检索 + recall 进行中

| 阶段 | 状态 | 产物 | 指标 |
|---|---|---|---|
| ① 预训练（MLM+对比，中文底座→专利语料） | ✅ 2026-08-25 完成（30h） | `ckpt/patent/pretrain/graphformer/1e-5/` | train_loss 1.29；val MRR **0.83** / NDCG@10 **0.87**。**不受泄漏影响** |
| ② 检索训练（泄漏版） | ❌ 作废，已存档 | 本地 `ckpt/patent/nc_retrieval_leaked/`（scp 于 08-28） | P@1 0.39 / MRR 0.63 —— 泄漏灌水，仅作对照组 |
| ②' 检索训练（修复版） | ✅ 2026-08-28 完成（10h02m，1231 步） | `ckpt/patent/nc_retrieval/graphformer/1e-5/` | eval_loss 1.30；**P@1 0.60 / MRR 0.75 / NDCG@10 0.81**（真实值，且高于泄漏版） |
| ③ 检索推理 → FAISS 检索 + recall | 🔄 进行中 | `node_label_embed/` + trec 结果 | 建索引时 documents.txt 被换行符劈裂，已从 documents.json 重建（25,598 家、0 异常行），infer/search 重跑中 |
| ④ 重排训练（100k 样本，~6h） | ⏳ 数据已修（正例 k_n 清空） | `ckpt/patent/nc_rerank/` | 底座用 ②' 的 checkpoint；**开跑前先清旧 pkl** |
| ⑤ 重排测试 | 🔄 全量运行中（09/03 09:12 启动，≈38.5h，预计 09/04 23:45 前后完成） | `rerank_test.log` | 验证集冒烟已通过：P@1 0.842 / MRR 0.893 / NDCG@10 0.914 |
| ⑥ 结果回传本地 | ⏳ 部分 | 本地 `ckpt/patent/` | ②' 的 ckpt 跑完后需 scp 拉回（关机 7 天清盘） |

## ③ 的重要注意（pkl 缓存陷阱）

`search.py:77`：若 `data/patent/nc/patent_patent_retrieval_dict.pkl` 已存在，search 会**直接读旧结果、跳过新检索**。
所以每次换模型/换索引重跑 ③，必须确认该 pkl 不存在（脚本正常结束时末尾会 rm；中途失败的不会）。

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

### ⑤ 重排全量评测进度查询（新开会话先跑这些）

```bash
cd /workspace/Patton
ps aux | grep test_rerank | grep -v grep        # 在跑 = 有输出
tail -1 rerank_test.log                         # 进度条 %（159298 步，1.15 it/s ≈ 38.5h）
grep eval_prc rerank_test.log                   # 完成后取最终指标
```

⚠️ 中途**不要**在新窗口重复启动评测脚本；脚本被 `git checkout` 重置过可执行位，
重跑前先 `chmod +x src/nc_rerank_test_patent.sh`。环境必须 `conda activate patton`（不是 py312）。

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
