# 根据专利推荐企业：新版数据协议

主任务：已知一项专利所属企业A，输入截止时点以前已公开的专利q及A的历史技术背景，检索之后首次在选定来源中观察到与A共同申请专利的企业B。

共同申请是企业合作的代理信号。一个企业对的后续共同申请不能证明B参与了q对应的具体项目，也不能证明首次真实商业合作。旧“专利→所属申请企业”的数据和模型保留为技术归属辅助对照；新版主任务评估截止前尚未观察到共同申请关系的候选企业推荐。

## 时间与答案

| 划分 | 输入资料截止点（不含当天） | 目标首次观测公开窗口 |
|---|---|---|
| train | 2022-01-01 | 2022年 |
| val | 2023-01-01 | 2023年 |
| test | 2024-01-01 | 2024年 |

1. 共同申请关系按该企业对在所选2016—2024年G领域来源中的最早可信公开事件定时，包含没有本地可用正文的旧关系事件。
2. 原始申请人名称/无歧义曾用名与同一企业代码必须同时出现在该事件中，不按申请人和代码列表位置配对。名称冲突排除；无生效日的曾用名不当作完整历史沿革。
3. 目标关系的公开日还必须等于该申请在当前来源中的最早公开日。已公开申请后来新增申请人、转让或更正产生的晚期组成变化，不作新伙伴标签。
4. 一次查询保留全部符合条件的目标企业；自身、截止前已经观察到的伙伴及明显同集团名字风险候选排除。名字风险规则使用截止前事件中出现的申请人名称，仍不能替代历史股权控制核验。
5. 未观察到共同申请的候选只作未标注干扰项，不宣称真实不合作。

“首次”始终是当前选定来源内首次观察。新增事件范围为主IPC为G的发明及已有G申请号的其他公开版本；这不是所有领域和所有年代的完整合作历史。

## 输入证据与候选池

- 每个年度池先独立生成：现有身份快照内、截止前至少两篇可用历史专利的全部企业均纳入，不按未来答案补候选。
- 同申请公开版本不互相覆盖。正文原始SHA256与对应事件完全一致，且该事件已经支持该企业的名称/代码归属时，才取得历史输入资格。不能将较晚B授权正文回溯到A公开日。
- 每企业按该正文版本对该企业的最早可见日期排序，最多保留20篇；每篇带申请号、事件ID、公开日、IPC、当时申请人名称及原始正文SHA。
- 正文先核对原始指纹，再移除已知供应商宣传词并保留最多1600字符。当前企业名称供展示，模型文本只用历史专利正文。
- 查询中心为源企业最新的一篇合格专利，不根据未来答案挑选。`q_n_text`只给出接下来五篇的兼容视图；规范候选证据保留完整的至多20篇集合。
- 干扰项由同年度历史候选池生成，保存16个词法困难项和16个随机项；所有正例和排除项均不作为干扰项。

候选池仍受现有企业快照及本地正文覆盖影响，应报告候选/查询规模及拒绝计数，不能称全国企业全集。缺正文或没有核实早期版本的事件仍可提供关系观测，但不能成为历史文本输入。

## 文件

数据独立保存在 `data/patent/cooperation_v2/`，旧 `cooperation_v1`、原库和旧checkpoint保留。

| 文件 | 用途 |
|---|---|
| `identity/identities.jsonl`、`summary.json` | 无歧义身份、旧名匹配辅助及来源指纹 |
| `../publication_metadata/events.sqlite`、`store_manifest.json` | 多版本公开事件、逐行出处、排除项及G正文匹配记录 |
| `preparation.sqlite` | 首次观察企业对及按企业/正文版本核对的历史资格缓存 |
| `evidence.jsonl.gz` | 全划分共享的逐篇历史正文摘录，每申请保留一次 |
| `histories.{split}.jsonl` | 每企业每年度至多20篇独立证据引用和时间/身份出处 |
| `documents.{split}.jsonl` | 独立历史候选池；词法检索每篇用前400字符 |
| `queries.{split}.jsonl` | 查询专利、源企业、多正例、排除项与标签公开证据 |
| `{split}.pairs.jsonl` | 正例/困难/随机候选ID；通过documents/histories获取20篇候选证据 |
| `evidence.tokens.jsonl.gz`、`tokenization_manifest.json` | 共享专利ID对应的最长256 token输入，已包含CLS/SEP |
| `token_cache/evidence.tokens.sqlite3` | 数据读取器可重建的共享压缩缓存，按规范token源指纹验证 |
| `dataset_manifest.json`、`data_audit.json` | 实际规模、协议、限制、来源和产物核验 |
| `baseline/metrics.{split}.json` | 新候选池上的词法基线结果，含输入指纹 |
| `dataset_report.md` | 实际规模、任务解释、CPU结果与尚未训练的边界 |

## 执行与核验

在仓库根目录执行，已完成产物的安全复用规则由各脚本控制：

```powershell
python -B data_pipeline/build_publication_event_store.py --help
python -B data_pipeline/build_enterprise_identities.py --help
python -B data_pipeline/build_patent_partner_data.py
python -B data_pipeline/audit_patent_partner_data.py --data-dir data/patent/cooperation_v2
python -B data_pipeline/tokenize_partner_evidence.py --data-dir data/patent/cooperation_v2
python -B data_pipeline/patent_partner_baseline.py evaluate --data-dir data/patent/cooperation_v2 --split val
python -B data_pipeline/patent_partner_baseline.py evaluate --data-dir data/patent/cooperation_v2 --split test
python -B data_pipeline/partner_evidence_dataset.py inspect --data-dir data/patent/cooperation_v2 --split train --index 0
```

词法基线使用中文bigram、SQLite FTS5 BM25和固定128个查询词预算。只命中排除项的词不占预算，其余按全局文档频率排序；不按验证或测试答案挑词。补零排名仅供候选采样兜底，零分补位不计检索命中，漏召回查询计0。它与旧Python全词BM25实现及旧小候选池不同，不能直接将指标差值解释为方法提升。

本版规范训练材料采用专利ID引用和逐篇证据格式，**不是**旧 `OpenLP.driver.train_neg` 的平铺JSONL。不能直接将DATA_DIR改成v2后运行旧训练脚本。新的集合编码/多原型训练入口应读取共享token、按候选收集至多20篇证据，并保留全部已知正例；不要重新拼成长文本后截断，也不要重复添加CLS/SEP。

`partner_evidence_dataset.EvidenceDataset` 提供逐样本读取接口：查询中心、五个邻居槽、每候选20个独立专利槽及其可见日期/IPC。正例mask放在训练目标中，候选顺序按查询/候选ID固定散列排序，不按正负标签排序。共享token使用独立的可重建SQLite缓存，不把全库token转换为常驻内存的Python整数列表。当前企业名称、标签证据及未来合作日期不进入输入特征。

读取器创建schema 2压缩缓存；外来或源指纹过期的缓存会被拒绝。已有本工具schema 1缓存时，可使用 `migrate_partner_token_cache.py --data-dir ... --report <新报告路径>` 显式迁移。它先只读检查旧缓存，在单独临时库中压缩，并将全部源token与旧/新库逐值核对，复核源指纹后才原子替换；不会改动规范证据、token或标签文件。

2026-10-26已新增独立 `partner_dense_baseline.py` 的encode/train/evaluate入口：冻结原始中文编码器，逐篇token均值，查询前6篇/企业至多20篇均值，训练共享线性投影，多正例目标，仅按完整验证池选best，测试独立评估。限量编码仅作smoke，禁止进入正式训练/评测。它保持逐篇证据，提供均值聚合控制，不是GraphFormers或贡献保持创新方法。新服务器配置与加密传输见 [4090操作手册](PARTNER_4090_GUIDE.md)。

数据、CPU基线及新入口准备不代表真实GPU流程已验证，或新GraphFormers/创新压缩机制已训练。后续仍需接入相应创新模型并实际运行；严格时间实验不使用含后续数据的旧专利checkpoint。
