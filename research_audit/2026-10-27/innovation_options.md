# 下一步创新方向建议

日期：2026-10-27  
背景：企业多技术兴趣表示（技术名片）在当前任务上已判定无可靠增益，详见同目录 `multi_interest_negative_report.md`。本文记录可以考虑的新方向、可复用代码/论文、预期收益和风险。  
原则：别人做过的组件可以用；但论文主张要有一个能被对照实验检验的 delta，不能只把标准模块接起来。

## 1. 当前判断

当前基线不是“再换一个池化技巧”能大幅提升的：

- 编码器是中文 BERT 类模型，输入被截断到 256 token；
- “重排”实际上只是点积，没有 token 级交互；
- 训练只用 1,341 条 query，负例以 BM25/随机为主；
- 企业历史专利只做平均，没有真正的 query-conditioned 证据选择。

所以最可能涨点的顺序是：

1. **强预训练编码器 + 混合检索**：先把第一段召回打上去；
2. **真正的 cross-encoder 重排**：把点积换成 token 交互；
3. **企业技术组合感知的 cross-encoder**：作为论文主创新；
4. **图 + 文本混合**：作为备选/补充，不作为第一优先。

## 2. 推荐顺序总表

| 优先级 | 方向 | 主要收益 | 论文角色 | 风险 |
|---|---|---|---|---|
| P0 | Qwen3-Embedding / BGE-M3 + BM25/dense RRF | 第一段 Recall@100 可能大幅提升 | 强基线/工程贡献 | 编码器预训练时间泄漏需披露 |
| P0 | Qwen3-Reranker / BGE-reranker-v2-m3 | P@1、NDCG、MRR 提升 | 强基线/重排组件 | 长文本、显存、延迟 |
| P1 | 企业组合感知的 cross-encoder | 论文主创新；可能同时提升 P@1/NDCG | 方法贡献 | 1,341 条 query 仍偏少，需 LoRA/早停 |
| P2 | 图 + 文本混合（NCN/BUDDY） | 暖启动子集提升；冷启动回退文本 | 备选方法贡献 | 先例多，delta 必须清楚 |
| 暂不做 | 生成式检索/SIDs、未来 IPC 组合、TabPFN、需求匹配 | 长周期/高风险/数据不足 | — | 不建议现在投入 |

## 3. 方向 A：强编码器 + 混合检索 + 真 cross-encoder

### 3.1 为什么值得先做

你现在的编码器把专利截断到 256 token，而专利的摘要和权利要求通常远长于这个长度；企业历史专利也常常只看到第一篇。换成一个长上下文、中文强的预训练编码器，是最可能带来“一个数量级以内最大单点增益”的改动。

### 3.2 具体组件

**A1. 编码器**

- 首选：**Qwen3-Embedding-0.6B**（1024 维，32K 上下文，官方报告中文 CMTEB-R 71.02；Apache 2.0）。
- 备选：**BGE-M3**（多语言，dense + sparse + multi-vector，支持长文本；FlagEmbedding 官方实现）。
- 用法：冻结编码器；query 用焦点专利/前 6 篇均值；候选企业用 2–20 篇历史专利；先做 mean pooling，再试 top-k/attention。
- 重要：当前 43.4 万篇专利中 **99.39% 都达到 256 token 上限**（这是截断后的长度，不是原文长度；原文中位数 600 字、99.46% 超过 256 字），模型只看到标题/摘要/早期权利要求；`evidence.jsonl.gz` 还有 `text[:1600]` 的存储截断（2.47% 专利受影响）。需要先从 `g_domain.db` 恢复完整清洗文本，再用新模型 tokenizer 统计长度、选择上限并编码；**不能复用旧的 256-token 缓存**。

**A2. 混合检索**

- BM25（已有 SQLite 索引）与 dense 各取 top-100，用 **RRF** 融合；
- 调 RRF 的 `k`（常用 60）和两路权重；只看 val；
- 预期：BM25 12.08% 与 dense 15–18% 的错误不完全重合，融合通常有 1–3 pt 的空间。

**A3. 真 cross-encoder 重排**

- 首选：**Qwen3-Reranker-0.6B**（32K 上下文，中文强；官方报告 CMTEB-R 71.31）。
- 备选：**BGE-reranker-v2-m3**（更轻、生态成熟、FlagEmbedding 直接可用）。
- 先 zero-shot 重排 top-50，确认有没有增益；再考虑 LoRA/小学习率微调。
- 微调数据：全部正例 + dense hard negatives（用当前 best retriever 从 train 全池挖，排除已知正例/历史关系/自身）；不要只用 BM25/随机负例。
- 预期：P@1、MRR、NDCG 提升；Recall@100 由第一段决定，重排只影响进入候选后的排序。

### 3.3 成本和顺序

- Qwen3-Embedding-0.6B 编码 43.4 万篇专利：4090 上预计数小时（取决于 max length 和 batch）。
- zero-shot reranker：top-50 × 2,112 test queries ≈ 10.6 万对；0.6B cross-encoder 可分批跑。
- 先做 A1/A2，再做 A3；不要一开始就全参微调。

### 3.4 论文定位

这套是“强基线/工程提升”，不是论文新意。它的作用是：

- 把第一段 Recall@100 打到一个新水平；
- 给后面的 portfolio-aware cross-encoder 提供更好的候选和 hard negatives；
- 让审稿人看到你不是只会堆多原型。

## 4. 方向 B：企业组合感知的 cross-encoder（论文主创新）

### 4.1 核心想法

保留你原来的“企业有多个技术方向”的思想，但**不再手工做 max/top-k**：

> 把 query 专利与企业的一组历史专利/技术方向一起交给 cross-encoder，让模型自己学习“query 命中了哪个方向、哪些专利证据重要”。

这正好针对之前失败的原因：

- max 会放大单篇噪声；cross-encoder 可以做 token 级证据比较；
- IPC 分组没有独立价值；模型可以从文本和 IPC 中自己学；
- 1,341 条 query 不足以从零训练，但可以微调一个预训练 reranker（LoRA/冻结大部分层）。

### 4.2 输入结构

不要简单把企业所有专利拼成一段长文本。建议结构化输入：

```text
[QUERY]
专利标题 + 摘要 + 权利要求 + IPC

[CANDIDATE COMPANY]
企业名称 + 经营范围

[EVIDENCE 1]
历史专利标题 + 摘要 + IPC
[EVIDENCE 2]
...
[EVIDENCE K]
...
```

- K 取 3–5，按 first-stage 分数或 IPC 重叠选，避免 20 篇全塞进去；
- 在输入中保留 `[QUERY]`、`[COMPANY]`、`[EVIDENCE]` 标记，让模型知道边界；
- 可用企业 IPC 分布作为额外结构化特征（例如 “IPC: G06T 40%, H04L 30%, ...”）。

### 4.3 训练

- 底座：Qwen3-Reranker-0.6B 或 BGE-reranker-v2-m3；
- 先 zero-shot；再用 train 微调；
- 损失：listwise softmax / pairwise / binary cross-entropy 均可；关键是**全部正例**都参与；
- 负例：dense hard negatives（从 best retriever 挖）+ 少量随机负例；
- 1,341 条 query 偏少：LoRA、冻结底座、1–2 epoch、早停、3 seeds；不要全参训练；
- 可选：用 RankGPT/Qwen3 对 top-20 做 listwise 重排，再蒸馏到小 cross-encoder，增加监督信号。

### 4.4 必备对照

如果这些对照不做，审稿人会说“这只是普通 cross-encoder”：

1. 同样输入、同样参数的**纯文本拼接 cross-encoder**（不分 `[EVIDENCE]`、不做结构化选择）；
2. 现有点积重排；
3. mean / top-k / IPC 分组的企业组合池化；
4. 如果有图特征，加 RLTHeG / From patents to partners 类图方法；
5. 冷启动（无历史合作边）vs 暖启动分层。

### 4.5 论文主张

可以写成：

> 面向“未来首次合作”的企业伙伴检索，提出一种企业技术组合感知的 cross-encoder：将企业历史专利集合、IPC 分布与查询专利结构化输入，学习 query-conditioned 的证据选择；在严格时间隔离的评测上，比较其相对同参数文本拼接 cross-encoder、点积重排和企业组合池化的增量。

不要写成“首次使用多兴趣/多向量/cross-encoder”。这些都不是新的；新的是**组合结构 + 任务协议 + 冷启动分析**。

### 4.6 预期与风险

- 预期：P@1/NDCG 有明显提升；Recall@100 取决于第一段，可能小幅提升；
- 风险：1,341 条 query 仍可能过拟合；如果结构化输入不胜过文本拼接，就撤掉“组合感知”的贡献；
- 停止条件：zero-shot reranker 和微调都不胜同参数文本拼接 → 这条线不成立。

## 5. 方向 C：图 + 文本混合（备选）

### 5.1 数据信号

- train 正例中约 **34.7%** 有历史共同邻居；
- 但 390/455 个查询源没有历史边；冷启动是常态；
- 因此不能做“纯图”方法，只能做“文本为主、图为辅、冷启动回退”。

### 5.2 可复用代码/论文

- **NCN / NCNC**：`GraphPKU/NeuralCommonNeighbor`，Neural Common Neighbor 特征；
- **BUDDY / subgraph sketching**：`melifluos/subgraph-sketching`；
- 先例：RLTHeG / From patents to partners（IP&M 2026）、R&D partner recommendation（IP&M 2026）；
- 这些先例已经覆盖“时序异构图 + 合作推荐”，所以纯“图+文本”很难算新；delta 必须落在**未来首次合作 + 冷启动门控 + 严格时间隔离**。

### 5.3 最小实验

- 特征：共同邻居数、RA、NCN 特征、候选企业历史合作边、文本分数、企业专利数/IPC 重叠；
- 模型：LightGBM/LambdaMART 或小 cross-encoder；
- 分层：暖启动（有历史边）vs 冷启动（无历史边）；冷启动只用文本；
- 判定：整体是否超过纯文本 cross-encoder；暖启动子集是否显著提升。

## 6. 现在不建议做

- **生成式检索 / semantic IDs（GRID）**：GRID、CapsID、GRDR、冷物品可达性都已占位；候选池 2 万多家、冷启动多，生成式 ID 可能记不住/不可达；不是快速涨点路径。
- **未来 IPC 组合预测**：MC-GCN、技术融合预测已有先例；你本地审计只有 153 个相关事件，数据太薄，不适合做主模型；可作为长期辅助任务。
- **TabPFN/TabICL**：适合小样本强基线，但很难打赢微调 cross-encoder，也不构成论文新意；可作为诊断工具。
- **需求—机理—条件匹配**：要新增需求级人工核验数据，周期长，不是当前涨点选项。
- **多兴趣/多原型**：已经收口，不再训练。

## 7. 两周执行计划

| 时间 | 任务 | 产出/判定 |
|---|---|---|
| Day 1–2 | Qwen3-Embedding-0.6B 编码 43.4 万篇专利原文；企业 mean/portfolio；val 评测 | R@100 是否明显 >16.35% |
| Day 3–4 | BM25 + dense RRF；调 k/权重 | 是否再涨 1–3 pt |
| Day 5–7 | Qwen3-Reranker/BGE zero-shot top-50；不行再 LoRA | P@1/NDCG 是否提升 |
| Day 8–12 | 结构化 portfolio-aware cross-encoder；全部正例 + dense hard negatives | 是否胜过同参数文本拼接 |
| Day 13–14 | 冷/暖分层、3 seeds、bootstrap、时间泄漏披露 | 决定是否写成论文 |

## 8. 评测纪律

- **候选池固定**：先报告第一段 Recall@100，再评测重排；漏召回计 0，不补答案。
- **test 不参与调参**：所有选择在 val 上做；最终 test 只跑一次。
- **时间隔离**：输入严格截止；编码器预训练时间要披露；Qwen3 作为“现代编码器上界”，老编码器作为严格对照。
- **同预算对照**：同编码器、同候选数、同参数；不要把“换更大模型”的收益算给新机制。
- **多种子/区间**：至少 3 seeds；query 级 bootstrap；同一企业/事件的查询不要当独立样本。
- **冷/暖分层**：分别报告有历史合作边和无历史边的查询；否则图方法的收益会被平均掉。
- **负结果也记录**：如果 portfolio-aware cross-encoder 不胜文本拼接，要明确写出来，不要包装。

## 9. 资源链接

**编码器/重排器**

- Qwen3-Embedding 官方博客：<https://qwenlm.github.io/blog/qwen3-embedding/>
- Qwen3-Embedding GitHub：<https://github.com/QwenLM/Qwen3-Embedding>
- Qwen3-Embedding-0.6B：<https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- Qwen3-Reranker-0.6B：<https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>
- BGE-M3：<https://huggingface.co/BAAI/bge-m3>
- BGE-reranker-v2-m3：<https://huggingface.co/BAAI/bge-reranker-v2-m3>
- FlagEmbedding：<https://github.com/FlagOpen/FlagEmbedding>

**重排/蒸馏**

- RankGPT：<https://github.com/sunnweiwei/RankGPT>，论文：<https://arxiv.org/abs/2304.09542>
- ColBERT：<https://github.com/stanford-futuredata/ColBERT>
- Jina-ColBERT-v2：<https://huggingface.co/jinaai/jina-colbert-v2>

**图方法**

- NCN：<https://github.com/GraphPKU/NeuralCommonNeighbor>，论文：<https://arxiv.org/abs/2302.00890>
- BUDDY / subgraph sketching：<https://github.com/melifluos/subgraph-sketching>，论文：<https://arxiv.org/abs/2209.15486>
- From patents to partners / RLTHeG：<https://www.sciencedirect.com/science/article/abs/pii/S0306457325003991>
- R&D partner recommendation：<https://www.sciencedirect.com/science/article/abs/pii/S0306457326002001>

**专利表示**

- PatentSBERTa：<https://huggingface.co/AI-Growth-Lab/PatentSBERTa>
- PaECTER：<https://arxiv.org/abs/2402.19411>

**生成式检索（暂不作为首选）**

- GRID：<https://github.com/snap-research/GRID>，论文：<https://arxiv.org/abs/2507.22224>

**技术融合（暂不作为首选）**

- MC-GCN 技术融合预测：<https://www.sciencedirect.com/science/article/abs/pii/S1751157725000987>

**小样本表格模型（诊断基线）**

- TabPFN：<https://github.com/PriorLabs/TabPFN>
- TabICL：<https://github.com/soda-inria/tabicl>
