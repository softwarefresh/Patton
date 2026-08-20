# 数据适配审查记录：专利→企业推荐 接入 Patton

> 记录时间：2026-08-04。基于对 repo 实际代码（脚本 / dataset / collator / Graphformer / 真实 cloth 数据）逐行核查。
> 用途：数据收集完成、开始改造管线前，按此清单执行。配合 `research-summary.md`（方案）与 `data-example.md`（数据示例）阅读。

---

## 一、总评

路线大方向合理，专利→企业 的 q/k 映射、三阶段流程（预训练→检索→重排）与 Patton 完全吻合。但存在 **2 处硬伤**（不解决直接失效）和 **2 处对原管线的认识偏差**。

## 二、硬伤 1：max_len = 32 写死

专利文本（标题+摘要+权利要求）可达上千 token，但代码里 32 是默认/写死的：

| 位置 | 现状 | 改动 |
|---|---|---|
| `src/scripts/build_train.py:15` | `--max_length 32`（可配） | 预训练时传更大的值 |
| `src/scripts/build_train_neg.py:35,37,41-49` | `max_length=32` **硬编码** | 参数化，建议 128~256 |
| `src/OpenLP/arguments.py:101` | 默认 `max_len=32` | 改默认 |
| `run_pretrain.sh:30`、`nc_retrieve_train.sh:33`、`nc_rerank_train.sh:35`、`infer`/`search` | `--max_len 32` | 统一调大 |

注意：`config.json` 中 `max_position_embeddings=512`，空间够，是脚本没放开。
代价：`per_device_train_batch_size 128 @ len32` 需按比例下调（长度×4 → batch≈32），否则显存不够。

## 三、硬伤 2：bert-base-uncased 是纯英文词表

- 词表 30522（`pretrained_ckpt/*/patton/config.json`），中文专利几乎全 `[UNK]`。
- repo 附带的全部预训练 checkpoint（cloth/sports/Economics/Geology/Mathematics）都是英文，**一个都用不上**。
- 必须换 `bert-base-chinese`（或中文预训练 BERT）作底座；而 `GraphFormers` 图结构层无现成中文权重 → **预训练阶段从"可选"变"必做"**。
- `run_pretrain_sci.sh` 已示范换底座（scibert→scipatton），照此模式换中文底座即可。
- **语言决策（全中文 / 中英混合）是开工前第一个要拍板的决定。**

## 四、认识偏差 1：检索/重排阶段 k_n_text（企业邻居）实际为空

`src/scripts/bm25_neg.py:122-123` 生成检索数据时写死 `k_n_text: [""]`；真实 `cloth/nc/train.16.jsonl` 中所有正负例的 `k_n_text` 都是 `[[]]`。

→ **原版 Patton 在检索微调/重排时只用查询侧邻居 `q_n_text`，键侧企业邻居根本不参与。**

若想在企业侧也喂邻居（企业专利组合/合作伙伴，对本任务很有判别价值），需要**自行改造数据构建**——这是对原管线的增强，不是照搬。要做就在 `bm25_neg.py` 之后、`build_train_neg.py` 之前加一步"为企业正负例填充 k_n_text"。

## 五、认识偏差 2：数据量级差距

cloth：95 万商品 / 2771 类 / 788 万链接 / 每样本 5 邻居。
建议至少：5000 专利 / 500 企业 / 1 万链接（`data-example.md` §7）——**差两个数量级**。110M 参数的图模型拿万级样本预训练学不出东西。量级不足时预训练应降级为可选项（见第八节路线）。

## 六、数据格式硬性约束（适配检查清单）

1. **邻居必须是固定 5 个槽位，空邻居用 `[]` 占位，不能少给。**
   原因：`src/OpenLP/models/Graphformer.py:495-496` 用 `neighbor_input.view(B, -1, L)` reshape；collator（`data_collator.py:31` 等）把 batch 内所有邻居 `sum(q_n, [])` 摊平后 pad。batch 内每样本邻居数必须一致（=5），否则**静默错位不报错**。
   正确写法见 `cloth/nc/test.rerank.10000.jsonl`：`[真实文本], [], [], [], []`。

2. **企业文本（documents）必须全局唯一。** `bm25_neg.py:38-39` 对重复文本 `assert` 崩溃。

3. **三份候选文件、两种格式都要有，ID 一致：**
   - `documents.txt`（TSV：`id\t文本`）—— `bm25_neg.py:29`、`infer`/`search` 读
   - `documents.json`（`[{id, contents}, ...]`）—— `bm25.py:31` 读

4. **检索数据起点是 `node_classification.jsonl`**（`{q_text, q_n_text, labels:[企业id], label_names}`），流程：
   `bm25.py`（产 `bm25_all_trec`）→ `bm25_neg.py`（产 `node_retrieval.jsonl` + 按行序 8:1:1 切分 + `test.truth.trec`）。
   `test.truth.trec` 格式：`qid 0 docid 1`（`bm25_neg.py:147`）。

5. **预训练 `train.jsonl` 是 tokenize 后的整数列表**（见真实 `cloth/train.jsonl`），由 `build_train.py` 从 `train.text.jsonl` 转换。

6. **字段名严格一致**：预训练 `q_text/k_text/q_n_text/k_n_text`；检索/重排 `q_text/q_n_text/positives[{k_text,k_n_text}]/negatives[...]`。

## 七、困难清单（按杀伤力排序）

| # | 困难 | 原因 | 影响 / 解法 |
|---|------|------|------------|
| 1 | 中文 BM25 失效 | `bm25/bm25.py:38` 用 `lower().split(" ")` 切词，中文无空格 → 整句一词 | 难负例构造失败；先用 jieba 分词或改写 bm25.py |
| 2 | 词表 / 底座 | 英文词表 + 英文 checkpoint 不可用 | 换中文 BERT 自训；预训练变必做 |
| 3 | 长文本截断 | 脚本写死 max_len=32 | 批量改脚本 + 降 batch（见第二节） |
| 4 | 邻居数据可得性 | 同 IPC 需同类别足够专利；中国专利**引用字段不全**；企业**合作伙伴专利库没有**（企查查/天眼查/供应链数据付费难拿） | 优先级：同IPC > 引用 > 合作伙伴；拿不到留空槽（模型支持 mask） |
| 5 | 专利权人归一化 | assignee 混个人/高校/企业名各种写法 | 去噪、归一、过滤非企业主体；个人/高校不进企业节点池。工程最重部分 |
| 6 | 数据量不足 | 万级链接带不动图预训练 | 预训练降级为可选项 |
| 7 | 评估口径 | 一专利通常单一 assignee（单正例），cloth 是每查询多标签 | P@1/MRR/Recall@50 可用；注意"专利授权个人再挂靠企业"等噪声标签 |
| 8 | 环境 | torch 1.8 / transformers 4.21 / faiss-gpu 1.7.2 / Py3.8；`run_*.sh` 按 Linux `/workspace` 写死 | 改 `PROJ_DIR`；Windows 下 faiss-gpu CUDA 版本安装麻烦 |

## 八、修正后路线（建议）

1. **先定语言**：中文数据 → 底座换 `bert-base-chinese`，接受放弃英文 checkpoint。
2. **预训练降级**：第一版 = 中文 BERT → 检索微调（max_len 调大）→ 重排。完整、可比、好复现。图预训练等数据到位后当增强实验。
3. **数据构造 ROI 排序**：① 专利文本+IPC+assignee（CNIPA/Google Patents 公开）→ ② 企业库（企查查公开）→ ③ 同 IPC 邻居 → ④ 引用邻居（可选）→ ⑤ 合作伙伴（最贵，拿不到就空槽）。
4. **代码改动清单**（数据到位后执行）：
   - `build_train_neg.py` 硬编码 `max_length=32` → 参数化
   - `build_train.py` 传大 `--max_length`
   - `arguments.py` 默认 `max_len`
   - 所有 `run_*.sh` / `nc_*.sh` 的 `--max_len 32`、`--tokenizer_name`、`PROJ_DIR`
   - `bm25/bm25.py` 中文切词
   - 新增：专利权人归一化 + 邻居构造 + 检索侧 k_n_text 填充（原 repo 无此步，需自研）

## 九、模型选型：Heterformer（异质网络升级路径）

**背景：** Heterformer（KDD 2023，UIUC Bowen Jin 等，[代码](https://github.com/PeterGriffinJin/Heterformer)、[论文](https://ar5iv.labs.arxiv.org/html/2205.10282)）是 Patton/GraphFormers 的后续工作，核心动机就是打破"同质假设"（所有节点有文本、同类型）。

**与课题的对应关系：**

| Heterformer 机制 | 对应本课题网络 |
|---|---|
| 边类型感知注意力（不同边用不同投影矩阵） | 专利权人边 / 同IPC边 / 企业拥有专利边 / 合作边——Patton 的 `GraphAggregation` 对它们一视同仁 |
| 文本富/文本贫节点分别聚合（两种 virtual neighbor token） | 本课题专利+企业都有文本，此条用不上 |
| 类型特定变换矩阵，异类节点映射同一语义空间 | 目前专利/企业走同一 encoder，节点类型是隐式的 |

论文在 DBLP/Twitter/Goodreads 上显著超过 GraphFormers 与 GraphFormers++（异质扩展基线），增益随网络异质/稠密程度增大——正是本课题（两类节点 + 四类边）的场景。

**关键取舍：**
- **数据格式不同：** Heterformer 输入是"邻接表 + 节点文本/特征"，与 Patton 的 JSONL 格式不兼容。**数据未收集前切换代价最小**，按 Patton 格式收好后改就贵了。
- **下游覆盖不同：** Patton repo 自带 DPR 式对比检索 + FAISS 索引 + trec_eval 评估，检索→重排基础设施现成；Heterformer 官方代码侧重链接预测/节点分类/聚类，**无现成密集检索管线**（本课题核心任务是检索→重排）。
- **中文是共同的坎：** 两个模型都要换 `bert-base-chinese` 初始化，英文 checkpoint 都不可用。

**三选项（改动量递增）：**
1. **守住 Patton**——数据规划不浪费，但模型忽略边类型（简单，异质信息利用不足）；
2. **借 Heterformer 思路**——沿用 Patton 检索/重排基础设施，把编码器改成"边类型感知"的异质版本（论文创新点更足，改动中等）；
3. **整体切换 Heterformer**——最贴合异质网络，但需自搭检索管线（改动最大）。

**倾向：** 数据到位后先跑通 Patton 基线，再用同一份数据评估 Heterformer 思路，作为对比/消融。模型选型待定，**不影响数据收集格式**（两类方案都要"专利全量 IPC + 企业文本 + 归一化 assignee"）。

## 十、Patton 之后的技术进展与趋势（2024+）

**技术地图三支演进（2022→2024+）：**

```
2022-23  嵌套式（图进BERT）  Patton/GraphFormers ─┐
                          Heterformer（异质扩展） ├─ 本课题技术基础
                          GLEM（EM 解耦）         ┘
          ─────────────────────────────────────────
2024+    LLM 时代图学习        GraphLLM 一族（图 token 化 + 指令微调）
                            生成式图预训练       GraphGPT（欧拉路径 + 下一节点预测）
```

**两大主流方向（趋势确实成立）：**

1. **图 token 化 / 序列化（GraphLLM 一族）**：把图结构转成 token 序列喂给 decoder LLM。代表：GraphGPT（[Alibaba，arXiv 2401.00529](https://ar5iv.labs.arxiv.org/html/2401.00529)，[ICML 2025](https://icml.cc/virtual/2025/poster/46483)）用**欧拉路径**无损序列化图 + "下一节点预测"预训练，可扩到 2B 参数，分子/蛋白/引文图 SOTA；同族 LLaGA、GraphTranslator、LGTL、SOG_k、RGLM 等。
2. **机制清醒剂**：[2026 机制分析](https://arxiv-org.ezproxy.obspm.fr/html/2606.03712v1)发现 LLM 中的"图汇点 token"激活异常高但**并不真正承载图语义**——图-LM 融合仍是开放问题，未成熟。

**对本课题的三不匹配（趋势真，但不宜作主线）：**

| 维度 | GraphLLM/GraphGPT 趋势 | 本课题 |
|---|---|---|
| 算力 | 亿级参数 LLM，几十卡起 | 中文 BERT 级（110M），单机可跑 |
| 任务 | 分子/蛋白/节点分类为主 | 专利→企业 **密集检索→重排** |
| 检索 | 该趋势很少做 dense retrieval，检索仍以双编码器为主 | 正落在此 |

**建议（低成本蹭趋势的口子）：**
- 主线不变：中文 BERT + 检索→重排（双编码器做检索仍是标准做法）。
- **重排阶段换 LLM-as-reranker**：精排对 Top-K 候选用小型中文 LLM 打交叉注意力分。不动预训练与粗排架构，只换最后一环，论文即可写"与 LLM 时代接轨"。
- 论文定位：GraphLLM/GraphGPT 放 related work 的"未来方向/对比小节"，作 outlook 而非主线。

## 十一、邻居构造设计（专利邻居 q_n_text）

**主选：同 IPC。** 中国专利引用/被引字段残缺严重，靠引用建邻居很多专利会落空；IPC 是专利局强制分类、覆盖率 100%，是最可靠的邻居来源。

**坑 A — IPC 颗粒度：** IPC 分级（`G06F` 大类 → `G06F40` 小类 → `G06F40/35` 主组）。按完整主组符号建桶太细，很多桶只有几条专利 → 邻居稀疏。**建议按大类或小类建桶**（如 `G06F40`），桶里专利多、才凑得够 5 个。

**坑 B — 同 IPC ≠ 技术相关：** 同类别跨度可以很大。IPC 只能保证"大致同领域"，不够精准。

**推荐做法：同 IPC + BM25 文本相似 混合选邻居。** 反正要跑 BM25 生成硬负例，同一个索引顺手就能给每个专利找"文本最相似的 N 条专利"：
1. 先取同 IPC 桶内的专利（保证领域相关性）；
2. 不够 5 条用 BM25 从全文找相似专利补齐（保证数量和精准度）；
3. 排除它自己。

这样既解决引用不全，又解决 IPC 稀疏，还能捕捉 IPC 抓不到的跨类相似（如"大模型"专利散在 G06F40/G06N/G10L 多个类下）。

**5 槽约束澄清：** 不需要凑满 5 个真实邻居。代码要求的是 `q_n_text` **固定 5 个槽位，拿不到的槽放 `[]`**，空槽会被 mask 掉不影响训练。`q_n_text = [A, B, C, [], []]` 合法（真实 cloth 数据就有大量空槽样本）。邻居构造不用强求 5 个，有多少算多少。

**对数据收集的要求：** 专利数据里**保留全部 IPC 分类号字段**（别只留主分类），粗细颗粒度以后才能自由切换。邻居构造是后处理脚本的活，与原始数据收集解耦。

## 十二、待数据到位后的 TODO

- [ ] 拍板语言与底座（中文 BERT？）
- [ ] 跑通最小 demo：10 条专利 / 5 家企业 走完整检索管线，验证格式
- [ ] 批量改 max_len 与 tokenizer
- [ ] 写 assignee 归一化脚本
- [ ] 写邻居构造脚本（同 IPC 优先）
- [ ] bm25.py 中文化
- [ ] 决定预训练做不做（取决于链接量是否 ≥10 万）

---

## 十三、数据准备进度（2026-08-07 快照）

### 已完成：阶段一（原始数据清洗入库）
- 6 个年份文件（中国专利数据库2020-2025年.csv，共 2514 万行）全部流式清洗进 **`data_pipeline/patents.db`（25GB 母库）**
- 母库：**861.9 万**唯一专利（申请 2016-2025）、**70.7 万**企业（统一社会信用代码）、**876 万**链接
- 原始 CSV 已全部删除（按"处理完一份删一份"），磁盘 36GB 空余

### 已定决策
- 领域 = **G 段（物理/计算）**，因课题作者为计算机专业
- 企业池门槛 = **≥10 项 G 段专利** → **26,442 家**
- 高校/科研院所**保留并标注类型**（类型启发式见 `build_company_pool.py`）

### G 段工作库（阶段二进行中）
- **`data_pipeline/g_domain.db`**（7.5GB）：260 万 G 段专利、273 万链接、2.6 万池
- **`data_pipeline/g_company_pool.csv`**：26,442 家**经营范围补全清单**（信用代码/名称/专利数/类型）
- 查询候选：**98.9 万**（2022-2024 申请、正例在池内）

### 下一步（接着做）
1. 【用户】拿 `g_company_pool.csv` 的**信用代码列**补经营范围（经营范围+成立日期+注册资本），26,442 家
2. 【代码】建专利侧训练输入（脚本未写）：查询取样、正例链接、同 IPC 邻居（q_n_text 5 槽）、时间切分
3. 经营范围到位后：企业文本 → `documents.json`，拼装训练文件
4. 之后：中文 BERT 底座、批量改 max_len（见第二、三节）、跑 Patton 检索→重排

### 脚本清单（`data_pipeline/`）
- `clean_patents.py` — 清洗 CSV→母库（流式：滤发明/滤个人/拆申请人/拼 q_text）
- `build_company_pool.py` — 全量企业池 + 类型标注
- `domain_stats.py` — IPC 领域统计
- `build_g_domain.py` — G 段子库 + 企业池清单（`--pool-threshold` 参数）
