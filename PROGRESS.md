# 实验进度（专利→企业合作推荐）

> 本文件记录实验流水线的当前状态，跨会话维护。新开会话先读这里。
> **完整实验全景（任务/数据/设计/结果/资产/教训）见 [EXPERIMENTS.md](EXPERIMENTS.md)——新接手者先读那个。**
> 最近更新：2026-10-27（轻量证据交互教师及同结构均值对照已实现，CPU检查通过，待4090配对验证；逐篇max验证Recall100为7.69%，低于冻结均值16.35%。固定预算压缩与贡献保持创新机制仍未实现/训练）

## 轻量证据交互教师：配对可行性实验（2026-10-27）

- 用户授权由我决定下一步并提供服务器操作指导。新增 `partner_evidence_teacher.py train-pair` 和 `partner_evidence_teacher_model.py`：复用完整434,047篇缓存，冻结既有 `ckpt/partner_dense/run1/best.pt` 投影作为基础分数；另训共享768→128投影、集合聚合及512→64→1小MLP，残差为0.25*tanh(MLP)，末层零初始化，epoch0复现原投影。
- **同一命令依次训练mean/attention两个残差头**：参数数目、初始化seed42、全部正例+原困难/随机未观察项、查询顺序、温度0.05、lr0.001、10epochs均相同，唯一架构差异是逐篇均匀聚合或查询条件softmax（温度0.2）；不把增加参数/MLP的收益算给attention。候选向量至多20篇，query仍为前6篇均值；没有训练GraphFormers，也没有完成固定R预算/删除响应保持模型。
- 先复核基础checkpoint/cache/train/val全部指纹并重算基础val，自动进行一次可丢弃的设备前反向检查；正式训练只加载train/val，完整20,389候选池的val Recall100选best，含epoch0。证据在验证时预投影一次，query batch8×candidate chunk512双分块评分，空槽屏蔽、排除自身/历史关系、按ID稳定处理并列。保存仅复制state_dict到CPU，活模型及optimizer引用不变。
- 输出 `ckpt/partner_teacher/run1/{mean,attention}/best.pt/history.json/training_summary.json` 及根目录 `base.val.json`、`preflight.json`、`run_config.json`、`comparison.json`。报告attention归一化熵/最大权重。attention若未同时胜过base和同结构mean对照，先停止压缩；若胜过也先多seed复核，单次微小优势不表示显著提升。**本轮不访问test，不新增编码，不改数据。**
- 服务器入口 `bash src/run_partner_teacher.sh` 自动nohup启动，拒绝复用已有run或日志；无需重装环境。新模型7项与集成8项CPU检查已通过，涵盖零残差基线等价、mask/置换/分块、参数公平、真实head更新、冻结base、拒绝错误指纹/缓存、删除test分割文件后仍可完整训练，以及overwrite参数转换下optimizer引用保持。**尚未在真实4090运行，也没有新的教师实验成绩或耗时承诺。** 按步命令见 `data_pipeline/PARTNER_4090_GUIDE.md` 第10节。
- 将23项上传白名单文件单独复制到 `tmp/partner_teacher_isolated_20261027/` 后，59项CPU检查全部通过（21.441秒），不依赖其他未上传研究文件；Git Bash `-n` 启动脚本语法检查及CLI帮助通过。检查记录见 `research_audit/2026-10-27/teacher_cpu_validation.json`。

## 4090新服务器迁移准备（2026-10-26）

- 用户当前服务器最新nvidia-smi仍为RTX4090、24564MiB，py312；**最新驱动595.80、显示CUDA13.2**，覆盖早先实例的610.57.04/CUDA13.3。原定torch2.6.0 cu124方案保留，实际CUDA运行已由64篇真实模型编码通过确认。沿用 `softwarefresh/Patton`，当前本地分支patent；用户确认仓库Public。**后续用户明确要求由我直接上传，数据不需要加密，安全不重要。该指示覆盖先前加密建议。** 现按小源文件正常Git提交、数据原始压缩分片直接放公开Release执行；用户不需要安装gh或7-Zip。
- 上传前核对远程heads：main=d715eb06ca4393815cfbfe9b846352300669dbf4，patent=7a8f8000e26eb4071757428747aab74872e89543，与本地基点相同。本次16个运行源文件提交为112cfec60e88ded3cae4e2e97baf714ff84ced02，已追加推送patent。将本次暂存文件单独复制到隔离目录后37项运行检查通过，补齐词法共用模块，未借助其他未上传文件；没有强推或改旧历史，其他研究文件未提交。
- 真实本地包 `tmp/partner_release/` 已生成且全量回读核验：**27文件（20数据/报告＋7底座）**，源2,549,000,484字节，gzip后1,095,628,722字节，3分片536,870,912/536,870,912/21,886,898字节。`partner-transfer.json` SHA256为a9863797846195a5f874462dacbdd047ebb69b9027a410a6f8004fd6d74d8d61。按用户最新明确授权，公开上传manifest及这3个未加密分片；此前自动生成的本地警示README不作为Release附件。未带原DB/年度CSV/旧训练checkpoint/可重建索引与缓存。
- 新增 `package_partner_data.py` 的export/verify/install：流式打包，内层逐文件与分片指纹，固定安装根，拒绝路径越界、链接、重复、篡改和覆盖既存数据；安装中断可回滚。原加密辅助保留在本机，本次按用户指示不使用、不上传，也不要求安装7-Zip。
- `requirements.partner4090.txt` 和 `src/setup_partner_4090.sh` 创建独立venv：Python3.10—3.12，torch2.6.0官方cu124、transformers4.44.2、numpy1.26.4；脚本执行CUDA前向/反向检查并写环境记录。官方wheel/驱动兼容性已核查；**用户已激活新环境并回传64篇真实GPU编码成功，尚无正式训练结果**，不使用旧requirements.txt/setup.sh。
- 新 `partner_dense_baseline.py` 提供encode/train/evaluate：冻结原始中文底座，逐篇非padding token均值（既有CLS/SEP只用一次），查询前6篇/企业至多20篇均值，训练共享线性投影和全部正例平均log-softmax。完整val候选池Recall100选择best（含epoch0控制），test不参与选轮次。完整编码向量float32约1.33GB、磁盘映射读取；限量smoke缓存禁止正式训练/评测。模型、tokenizer、规范数据、缓存及checkpoint均绑定指纹，test的canonical manifest也须匹配训练记录。
- 新增16项相关测试（8传输、8神经入口），加旧39项共**55项本地检查通过**；神经端到端使用CPU假编码器，**不能据此声称4090真实模型/全量训练已通过**。本地原始中文底座权重已用torch2.6 weights_only读取核对，tokenizer指纹与规范数据一致。
- 已公开发布：https://github.com/softwarefresh/Patton/releases/tag/partner-v2-20261026 ，版本标签绑定112cfec代码提交，4个附件为 `partner-transfer.json` 和 `partner-assets.tar.gz.part001/002/003`。所有远端附件大小/上传状态/SHA256与本地一致。另以无认证请求下载整个manifest并核对SHA、逐个读取分片首字节及Content-Range总长，HTTP200/206均通过；服务器不需要GitHub登录、解密工具或密码。实际记录：`research_audit/2026-10-26/github_publication.json`。
- 按步手册：`data_pipeline/PARTNER_4090_GUIDE.md` 已改为未加密分片直接下载流程；本轮本机上传已完成，用户从第4步服务器下载继续。`stage_partner_code.ps1` 只用于后续自行更新。已用本机原Git登录完成推送/Release API上传，没有要求用户安装gh或7-Zip。未连接用户服务器、未执行远程训练。
- 用户已回传服务器安装成功日志：来源 `/workspace/partner_release`，目标 `/workspace/Patton`，status=installed、files_verified=27、manifest SHA=a9863797846195a5f874462dacbdd047ebb69b9027a410a6f8004fd6d74d8d61，canonical_sources_modified=false。下一步建独立环境、64篇真实GPU编码smoke、完整编码及均值/投影基线。它是新增可运行基线，**GraphFormers、贡献保持和创新方法训练入口仍待实现**，不混用旧所属企业任务结果。
- 用户已激活 `/workspace/Patton/.venv_partner4090`，回传真实模型CUDA smoke成功：status=passed、patents=64、smoke_only=true、output_dir=/workspace/Patton/artifacts/partner_smoke、elapsed_seconds=1.491。证明当前4090可执行本地中文底座编码，尚不代表完整编码/训练已完成。下一步在新目录 `artifacts/partner_vectors` 完整编码434,047篇，不传max-patents；建议nohup后台运行并保存日志，正式投影训练必须等待完整缓存生成成功后再开始。无需安装额外后台管理工具。
- 用户已回传**全量真实CUDA编码完成**：status=passed、patents=434047、smoke_only=false、output_dir=/workspace/Patton/artifacts/partner_vectors、elapsed_seconds=621.984（约10分22秒）；nohup任务Done。现可训练，不再执行encode或重装数据。下一步 `train --embedding-dir artifacts/partner_vectors --output-dir ckpt/partner_dense/run1 --epochs 10 --batch-size 32 --seed 42 --device cuda`，只训练共享投影，验证集全池Recall100选best（含epoch0控制），测试集仍不参与。
- 开跑前改进checkpoint：仅对state_dict张量detach/CPU/clone再保存，删除活模型的cpu()/to(device)迁移，保持GPU参数及optimizer引用不变；不改变编码格式或指纹，可复用现有434,047篇缓存。增加检查缓存/聚合训练与验证证据的阶段日志，以及最终status=completed摘要和耗时。CPU独立审查表明overwrite-module-parameters转换配置下旧迁移方式会使optimizer引用失配；这不表示用户默认配置已经失败，尚未运行正式训练。
- 该训练修正的dense组9项检查通过，新增完整CPU训练回归：开启overwrite、关闭swap转换行为，确认optimizer与活模型参数为同一对象，且weight_decay=0时训练真实更新权重；两个future flags均finally恢复。此前8项多正例、缓存/标签隔离和端到端检查继续通过。当前没有正式GPU投影训练/测试成绩。
- 用户已回传真实4090投影基线训练摘要：`ckpt/partner_dense/run1/training_summary.json` status=completed、best_epoch=1、best_validation_recall@100=0.18382785882011374、test_accessed=false、encoder_frozen=true、innovation_model_trained=false、elapsed_seconds=143.407。完整编码缓存继续复用；正式检索基线训练已完成，GraphFormers/创新模型仍未训练。下一步按预定方案各评测一次best.pt与无checkpoint冻结均值对照，使用test全部2,112查询与22,312候选；输出 `logs/partner_v2/projection.test.json`、`frozen.test.json`，不得根据测试结果选epoch或调参。
- **本轮测试结果已回传并记录**：投影/冻结均值/BM25：Recall20=5.65%/6.33%/6.10%，Recall100=15.01%/14.01%/12.08%，Hit100=23.06%/21.50%/17.14%，P1均1.61%，MRR100=0.03102/0.03282/0.03144。投影比同缓存冻结均值R100增1.00个百分点，R20降0.68个百分点，不能称全面或显著提升；Top100命中查询数分别487/454/362，Recall100不是查询成功率。测试均2,112查询、22,312候选、4,202正例。已对照本地文件验证用户输出8个输入SHA，以及BM25的3个重叠SHA一致；两神经方法embedding_manifest SHA均435a4ba935b0c20376de550eefb1f4e5224d824c9761ede73880615796c9c5a1，最佳checkpoint回传SHA5056b7f11cbcfa8c2dde1898002311d529ba0eb8878b0120f018065393716887。
- 结果保存在 `research_audit/2026-10-26/server_reported_projection.test.json`、`server_reported_frozen.test.json`、`retrieval_baseline_comparison.json/.md`，神经结果为用户终端JSON转录，未独立取回服务器checkpoint/向量文件。仅seed42，尚无置信区间或多种子稳定性证据；与BM25的差别不只来自投影，文本处理/评分及索引预算也不同。GraphFormers及创新机制仍未实现/训练。
- 按10月15日预定顺序，下一步先用既有冻结缓存在**验证集**比较逐篇最大匹配，再检验证据交互教师是否胜过均值/max，之后才值得开展固定R预算压缩与删除响应保持。逐篇max保留至多20个向量，是未压缩参照，不能称性能数学上界或相同索引预算的创新收益；本轮不根据已知测试成绩调参，也不重复测试现有两种方法。
- 新增 `partner_evidence_max_baseline.py evaluate`，复用原始冻结缓存和同一query前6篇均值，候选逐篇L2归一取最大余弦，空槽-inf、全池排除与稳定ID排序；不加载投影W、不加载BERT、不训练新参数。报告实际向量数/维度及含padding张量字节，不把这些值称为实测峰值或FAISS索引大小。其6项CPU检查通过（负分空槽、max与mean区别、全池排除/并列/标签无关性、真实缓存fixture、指纹与时间拒绝），尚未在真实GPU/完整验证集运行；下一步仅 `--split val --output logs/partner_v2/fullmax.val.json --batch-size 16`。
- 新对照加入上传白名单后，将18个源文件的Git暂存内容独立复制到临时目录，完整运行环境相关44项CPU检查全部通过，确认不依赖未上传的构建脚本。逐篇max的真实GPU验证结果仍待用户运行回传。
- 服务器接续反馈：首次安装在check_parts阶段发现001仅50MiB、002仅60MiB（均应512MiB），003大小21,886,898字节正确，尚未写入安装目标。断点续传HTTP206有效，但附件节点实测速率约9KB/s，ETA约10小时；这是当前下载链路的表现，不能据此断言平台或GitHub限速。用户已提供控制台SSH登录命令（端口23），Windows现有OpenSSH/scp可用，改建议本机完整 `tmp/partner_release` 通过SCP直传 `/workspace/partner_release`，再从该目录install。首次主机确认及SSH密码由用户在本机终端交互输入，未索取密码、未由我实际上传到服务器。
- **新会话接续：逐篇max已在真实4090完整val跑完**。用户终端记录显示代码更新至03d8a22，236,949项候选历史证据（非去重专利数），耗时18.611秒；val为1,732查询、20,389候选、3,112正例。max/冻结均值epoch0：Recall20=2.93%/7.07%，Recall100=7.69%/16.35%，Hit100=11.14%/23.38%，P1=0.92%/1.73%，MRR100=0.01554/0.03486。max的R100低8.67个百分点，Top100命中查询193/405，五项指标均低于均值。该负结果只否定当前冻结向量的直接max规则，不能据此否定训练后的证据交互或多原型方法，也未做显著性检验。
- 已保存 `research_audit/2026-10-26/server_reported_fullmax.val.json`、`server_reported_frozen.epoch0.val.json` 和 `evidence_aggregation_validation_comparison.json/.md`。max的8项输入SHA对本机全部一致，规模与dataset manifest一致，embedding manifest SHA与此前服务器冻结均值报告一致；均值来自用户读取training_history[0].validation的摘录，未附run_config/指纹，未独立取回服务器文件。现有434,047篇缓存继续复用，**不重跑encode或本次max，也不新增test评测**。下一步沿10月15日预定方案实现并验证轻量证据交互教师，与均值/max及训练投影对照，教师若无优势先停止压缩蒸馏；固定R预算与删除响应保持仍未实现/训练。

## 根据专利推荐企业：新版数据已完成（2026-10-24）

> 下列为10月24日数据完成时的记录；10月26日已新增上方独立神经基线入口，尚无真实GPU训练结果。

- 用户授权继续加工并由我选择任务口径。主任务确定为“已知源企业的一项历史专利及背景→下一年度首次在所选来源观察到的共同申请企业”，旧申请人归属匹配保留作辅助，不将旧分数当新任务结果。公司关系对具体查询专利仅是弱监督，不能称该项目真实伙伴。
- 九年来源已合并为 `data/patent/publication_metadata/events.sqlite`（2,602,905,600字节），4,095,897个不同公开事件、3,160,207个申请号；17,131条完全重复来源折叠，已知1条复核记录排除。G正文2,278,062/2,348,484篇有精确SHA匹配版本；历史资格仍逐企业/版本核对。
- 企业身份辅助完成：`data/patent/cooperation_v2/identity/` 含22,914个身份、32,643个无歧义名称/旧名；同事件原申请人名与代码交叉核验，不按位置强配。历史名字用于同集团名字风险筛查，当前名仅展示。
- `data/patent/cooperation_v2/` 已实际生成：train/val/test目标年2022/2023/2024，查询 **1,341/1,732/2,112**，独立历史候选池 **16,243/20,389/22,312** 家，有向正例 **2,418/3,112/4,202**。全部合格源查询企业均纳入，不按未来答案补池。合计5,185条年度查询、434,047篇去重历史输入；每企业2—20篇独立证据，保留多IPC方向。源企业可跨年度出现，不将查询数当独立企业数。
- 目标标签来自该申请在来源中的最早公开日的A/B种类事件，不接受已公开后新增申请人组成的伪新合作。无正文事件也参与判断旧关系；自身、历史伙伴及明显集团名字风险排除，复核申请整件排除。候选保存全部正例及16困难+16随机未观察项，后者不是真实负例。
- `data_audit.json` 全量独立审计passed：核对434,047篇唯一输入及436,717项源事件元数据，重建完整候选/查询/答案，检查4,866个无向标签对跨划分不重叠；每划分100篇对照原库正文。共享token全量检查并抽300篇独立重新编码，CLS/SEP已包含、最长256。10月24日另复核13个规范产物SHA、事件库SHA、构建来源大小/修改时间和基线输入指纹，记录 `research_audit/2026-10-24/patent_partner_completion_checks.json`。
- CPU中文bigram FTS5 BM25已在完整val/test查询和候选池运行：Recall@20 **7.05%/6.10%**，Recall@100 **13.18%/12.08%**，P@1 **1.10%/1.61%**。答案仅用于评测，零分补位不计命中。任务、候选池及词法实现与旧任务不同，不能横比旧分数；当前结果只证明流程可运行，未证明创新方法效果。
- `partner_evidence_dataset.EvidenceDataset` 已提供查询中心、5个邻居槽、每候选20个独立专利槽和多正例mask；候选顺序不按标签排序，共享token不全库装入Python整数列表。规范材料是ID引用集合，**不能直接改目录运行旧train_neg脚本**。需接入集合编码/多原型训练、统一基线及消融；新GraphFormers、教师和贡献保持模型均尚未训练。
- 10月24日共享读取缓存已完成schema 1→2安全迁移：2,028,662,784→295,538,688字节，释放1,733,124,096字节；434,047条与规范token源逐值核对、旧/新SQLite完整性及源文件重哈希均通过，记录 `research_audit/2026-10-24/token_cache_migration.json`。train/val/test真实读取复测通过并共享该缓存，20篇槽和多正例保持、标签证据不进输入，记录同目录 `evidence_adapter_smoke.json`。构建/基线/适配器/迁移合计39项测试通过。
- 本机仍为torch CPU且缺训练依赖；严格时间实验不使用含后续数据的旧checkpoint。协议见 `data_pipeline/PATENT_PARTNER_V2.md`，实际报告见 `data/patent/cooperation_v2/dataset_report.md`。旧库、旧原型和旧checkpoint保留；当前不需继续批量下载年度CSV。

## 基本实验条件复核（2026-10-22）

> 以下为新版构建前的历史核查；当前数据状态以上方10月24日记录为准。

- 当前结论：**资料基础处理完成，可以开展现有小规模检索基线验证；正式创新实验尚不能直接训练。** 详见 [`data_readiness.md`](research_audit/2026-10-22/data_readiness.md)。
- 九个年度压缩包重新做大小/SHA256检查，全部与manifest一致；共402,898,063字节、4,113,029条尚未跨年合并的事件记录。记录：`research_audit/2026-10-22/readiness_checks.json`。
- 原型 `cooperation_v1` 只读审计重新通过：300/100/100查询、638/323/374候选、7,039篇母库输入核对、695个无向标签对跨划分不重叠、分词结构有效。val/test BM25指标所对应的6个输入指纹均一致。新审计：`research_audit/2026-10-22/readiness_cooperation_recheck.json`；原数据和旧审计未改。
- 年度事件仍需合并、连接母库核验正文版本/时间及企业身份，处理2022年1条复核记录；正式集需重建独立历史候选池、扩大样本并保留逐篇证据。年度包齐全不能代表所有正文/版本已覆盖。
- 本机实测仍为torch2.6.0+cpu、CUDA不可用且训练依赖不全。神经基线正式训练前还需统一训练与建索引的特殊token输入路径。未重建正式集或启动训练，后续先做覆盖与缺口核验，不要求用户继续批量下载年度包。

## 2016年元数据提取、核验与原CSV删除（2026-10-22）

- 用户要求对 `D:\dev\中国专利数据库2016年.csv` 做基础处理，沿用逐年提取、核验后删除本地原表的约定。原大小4,837,337,284字节，UTF-8 BOM、34列表头；独立前5000条核心字段、日期和标识检查通过，公告年2016。前部含1997—2010年申请及正常旧制申请号，按本轮范围筛选，不将其误判为读取失败。
- 全表扫描2,598,155条CSV记录，压缩产物 `data/patent/publication_metadata/publications.2016.g.jsonl.gz` 保留 **87,545条公开事件、87,464个申请号**，大小 **7,895,347字节（7.90MB）**，缩小99.84%。全部导出公告年2016、日期有效，无导出复核标记；另保留277条单按当前行G过滤会漏掉的已有申请事件。
- 压缩包全部87,545条重读、SHA256、范围计数对账通过；另独立全表重读核对源CSV总条数和字段宽度，按导出序号均匀抽取12条对照全部映射字段及正文指纹。源CSV和G库大小/修改时间检查通过。验证记录：`research_audit/2026-10-22/compact_metadata_validation.2016.json`。
- 核验后已仅删除本地2016年原CSV，释放4.84GB；manifest记录删除状态。2016—2024九个精简主包合计 **402,898,063字节（402.90MB）**，共 **4,113,029条事件记录（尚未跨年合并去重）**；既有manifest、核验记录和2022异常原行备份均保留。母库、G库及其他原始文件未动，训练未启动。
- 本轮计划的年度元数据来源已补齐，下一步合并公开事件、连接母库核对正文版本/可用性，核验企业身份并构建独立历史候选池及扩大的训练评测集。2016年仅3,547条事件匹配现有G库申请号白名单，其余正文覆盖仍须核对；精简包不是正文备份，也不能据此宣称真实首次公开、首次合作或完整更早历史。2022年1条复核记录仍需排除或修正后正式入集。

## 2017年元数据提取、核验与原CSV删除（2026-10-22）

- 用户要求继续处理 `D:\dev\中国专利数据库2017年.csv`，沿用“每年提取核验后删除本地原表”的约定。原大小5,560,961,540字节，UTF-8 BOM、34列表头；独立前5000条核心字段、日期和标识检查通过，公告年2017，旧制申请号和特殊文献种类按源值保留，无阻止提取的结构异常。
- 全表扫描2,860,285条CSV记录，压缩产物 `data/patent/publication_metadata/publications.2017.g.jsonl.gz` 保留 **205,279条公开事件、204,557个申请号**，大小 **18,890,493字节（18.89MB）**，缩小99.66%。所有导出公告年2017、日期有效，无导出复核标记；另保留1600条单按当前行G过滤会漏掉的已有申请事件。
- 压缩包全部205,279条重读、SHA256及范围计数对账通过；另独立全表重读核对源CSV总条数和字段宽度，并按导出序号均匀抽取12条对照全部映射字段及正文指纹。源CSV和G库大小/修改时间检查通过。验证记录：`research_audit/2026-10-22/compact_metadata_validation.2017.json`。
- 核验后已仅删除本地2017年原CSV，释放5.56GB；manifest记录删除状态。2017—2024八个精简主包合计395.00MB，既有manifest、核验记录和2022异常原行备份均保留。母库、G库及其他原始文件未动，训练未启动。下一份优先2016年公开公告包。
- 2017年产物中37,802条事件匹配现有G库申请号白名单；正式构建仍需连接母库核对其余条目的正文和版本，精简包不是正文全集。信用码源缺失未填造，申请人/信用码也未按位置强行配对。

## 2018年元数据提取、核验与原CSV删除（2026-10-22）

- 用户要求继续处理 `D:\dev\中国专利数据库2018年.csv`，沿用“每年提取核验后删除本地原表”的约定。原大小7,531,293,317字节，UTF-8 BOM、34列表头；独立前5000条核心字段、标识、日期检查通过，公告年2018，无阻止提取的异常。
- 全表扫描3,695,711条CSV记录，压缩产物 `data/patent/publication_metadata/publications.2018.g.jsonl.gz` 保留 **302,900条公开事件、301,636个申请号**，大小 **28,267,600字节（28.27MB）**，缩小99.62%。所有导出公告年2018、日期有效，无导出复核标记；另保留689条单按当前行G过滤会漏掉的已有申请事件。
- 压缩包全部302,900条重读、SHA256、条数对账及原CSV12条字段/正文指纹核验通过，源CSV和G库大小/修改时间检查通过。验证记录：`research_audit/2026-10-22/compact_metadata_validation.2018.json`。
- 核验后已仅删除本地2018年原CSV，释放7.53GB；manifest记录删除状态。2018—2024七个精简主包合计376.11MB，既有manifest、核验记录和2022异常原行备份均保留。母库、G库及其他原始文件未动，训练未启动。下一份优先2017年公开公告包。
- 2018年产物中81,692条事件匹配现有G库申请号白名单；正式构建仍需连接母库核对其余条目的正文和版本，精简包不是正文全集。

## 2019年元数据提取、核验与原CSV删除（2026-10-22）

- 用户要求继续处理 `D:\dev\中国专利数据库2019年.csv`，沿用“每年提取核验后删除本地原表”的约定。原大小7,850,256,131字节，UTF-8 BOM、34列表头；独立前5000条核心字段、标识、日期检查通过，公告年2019，无阻止提取的异常。
- 全表扫描3,827,095条CSV记录，压缩产物 `data/patent/publication_metadata/publications.2019.g.jsonl.gz` 保留 **416,268条公开事件、405,719个申请号**，大小 **39,366,450字节（39.37MB）**，缩小99.50%。所有导出公告年2019、日期有效，无导出复核标记；另保留1220条单按当前行G过滤会漏掉的已有申请事件。
- 压缩包全部416,268条重读、SHA256、条数对账及原CSV12条字段/正文指纹核验通过，源CSV和G库大小/修改时间检查通过。验证记录：`research_audit/2026-10-22/compact_metadata_validation.2019.json`。
- 核验后已仅删除本地2019年原CSV，释放7.85GB；manifest记录删除状态。2019—2024六个精简主包合计347.84MB，既有manifest、核验记录和2022异常原行备份均保留。母库、G库及其他原始文件未动，训练未启动。下一份优先2018年公开公告包。
- 2019年产物中128,193条事件匹配现有G库申请号白名单；其余不能直接假定已有可用正文，正式构建时仍需连接母库核对文本与版本。精简包只保留元数据和正文指纹，不是正文全集。

## 2020年元数据提取、核验与原CSV删除（2026-10-22）

- 用户要求继续处理 `D:\dev\中国专利数据库2020年.csv`，沿用“每年提取核验后删除本地原表”的约定。原大小9,958,307,792字节，UTF-8 BOM、34列表头；独立前5000条核心字段、日期及标识检查通过，公告年2020，无阻止提取的异常。
- 全表扫描4,852,120条CSV记录，压缩产物 `data/patent/publication_metadata/publications.2020.g.jsonl.gz` 保留 **468,603条公开事件、454,620个申请号**，大小 **45,371,001字节（45.37MB）**，缩小99.54%。所有导出公告年2020、日期有效，无导出复核标记；另保留7762条单按当前行G过滤会漏掉的已有申请事件。
- 压缩包全部468,603条重读、SHA256、条数对账和原CSV12条字段/正文指纹核验通过，源CSV及G库大小/修改时间检查通过。验证记录：`research_audit/2026-10-22/compact_metadata_validation.2020.json`。
- 核验后已仅删除本地2020年原CSV，释放9.96GB；manifest记录删除状态。2020—2024五个精简主包合计308.48MB，既有manifest、核验记录和2022异常原行备份均保留。母库、G库及其他原始文件未动，训练未启动。下一份优先2019年公开公告包。

## 2021年元数据提取、核验与原CSV删除（2026-10-22）

- 用户要求继续处理 `D:\dev\中国专利数据库2021年.csv`，沿用“每年提取核验后删除本地原表”的约定。原大小12,633,487,446字节，UTF-8 BOM、34列表头；独立前5000条核心字段、标识和日期检查通过，公告年2021，无阻止提取的异常。
- 全表扫描6,017,671条CSV记录，压缩产物 `data/patent/publication_metadata/publications.2021.g.jsonl.gz` 保留 **588,400条公开事件、557,385个申请号**，大小 **57,591,008字节（57.59MB）**，缩小99.54%。所有导出公告年2021、日期有效，无导出复核标记；另保留6517条单按当前行G过滤会漏掉的已有申请事件。
- 压缩包全部588,400条重读、SHA256、条数对账及原CSV12条字段/正文指纹核验通过，源CSV和G库大小/修改时间检查通过。验证记录：`research_audit/2026-10-22/compact_metadata_validation.2021.json`。
- 核验后已仅删除本地2021年原CSV，释放12.63GB；manifest记录删除状态。2021—2024四个精简主包合计263.11MB，既有manifest、核验记录和2022异常原行备份均保留。母库、G库及其他原始文件未动，训练未启动。下一份优先2020年公开公告包。

## 2022年元数据提取、核验与原CSV删除（2026-10-21）

- 用户要求继续处理 `D:\dev\中国专利数据库2022年.csv`，并询问资料用途。沿用前面“每年提取核验后删除本地原表”的约定。原大小11,976,208,555字节、UTF-8 BOM、34列表头；独立前5000条检查字段与日期对齐，公告年2022，无阻止提取的异常。
- 全表扫描5,731,871条CSV记录，精简包 `data/patent/publication_metadata/publications.2022.g.jsonl.gz` 保留 **677,858条公开事件、634,371个申请号**，大小 **66,795,409字节（66.80MB）**，缩小99.44%。全部导出公开年2022、日期有效；额外保留3270条仅按当前行G过滤会漏掉的已有申请事件。
- 整包重读、SHA256、条数对账及原CSV13条字段/正文指纹对照通过。其中1条为非空尾字段复核标记，其完整原始行另存 `publications.2022.g.review.jsonl.gz`（1235字节）并重读核验；正式构建数据集前排除或修正该标记，不能当作已审定金标。核验记录见 `research_audit/2026-10-21/compact_metadata_validation.2022.json`。
- 核验后已仅删除本地2022年原CSV，释放11.98GB；manifest记录删除状态。2022—2024三个精简包、manifest及2022异常原行备份均保留，压缩主包合计205.52MB。母库、G库和其他原始文件未动，训练未启动。下一年度优先2021年公开公告包。
- **这些信息的用途**：为“保留企业多个技术方向、改善粗检索召回”的实验构建历史企业证据。公开事件/日期用于确定截止前可见性；申请人和信用码原串用于身份归属核验；IPC用于技术分组和版本核对；正文指纹帮助核实已有正文对应哪版。只补元数据不等于已验证真实首次公开、合作金标或已训练新方法，不能把后来授权文本回溯到更早公开日期。

## 2023年元数据提取及原CSV删除（2026-10-20）

- 用户提供并授权“处理完删除” `D:\dev\中国专利数据库2023年.csv`，原大小10,569,460,880字节。原表34列、UTF-8 BOM，与现有提取规则兼容；独立前5000条抽样核心字段和日期对齐，无阻止处理的结构异常。
- 已使用 `compact_patent_metadata.py` 扫描4,919,838条CSV记录，保留 **756,476条公开事件、682,593个申请号**；压缩产物 `data/patent/publication_metadata/publications.2023.g.jsonl.gz` 大小 **75,703,073字节（75.70MB）**，缩小99.28%。所有导出公告年为2023，日期有效，无导出复核标记；另保留785条单按当前行G过滤会漏掉的已有申请事件。
- 压缩包全部756,476条重读、SHA256、源CSV抽12条字段和正文指纹均核验通过，源CSV和G库大小/修改时间检查通过。验证记录见 `research_audit/2026-10-20/compact_metadata_validation.2023.json`。
- 已按用户授权仅删除本地2023年原CSV，释放10.57GB；2023年精简包、manifest及2024年产物均保留，manifest标记源已删除。母库、G库及其他原始文件未动。后续按年继续，下一份优先2022年公开公告包。

## 本地2024年原CSV已删除（2026-10-20）

- 用户明确授权删除 `D:\dev\中国专利数据库2024年.csv`。删除前再次核验原CSV大小/修改时间、精简包SHA256，并完整解压解析609,700条记录，均通过。
- 已仅删除该本地CSV，释放6,606,164,739字节（6.61GB / 6.15GiB）；`data/patent/publication_metadata/publications.2024.g.jsonl.gz`及manifest保留。母库、G库和另一份上市公司CSV未动。manifest已记录用户授权及源CSV删除状态；完整原件仍由用户网盘保存。

## 存储受限：逐年提取元数据（2026-10-20）

- 用户无法同时存放所有年度原表，**不再要求全部年份CSV同时下载留存**。现有库已有历史正文；每次只下载一个公开公告年度，流式提取小补充包并核验，保留云端原件及本地压缩结果，本地年度副本可在确认原件可恢复后周转。脚本不删除原文件。下一份优先2023年度，不必等所有年度回补后才做保守协议下的检索流程试跑。
- 新增 [`compact_patent_metadata.py`](data_pipeline/compact_patent_metadata.py) 和 [使用说明](data_pipeline/PUBLICATION_METADATA.md)。保存原G申请号全部对应事件，合并当前行G类发明新增记录；保留公开版本、日期、原申请人/信用码原串、IPC和出处，不复制正文。仅存与原清洗拼接规则一致的正文SHA256，用于防止将后来版本正文回溯到更早公开日。
- 已实际提取 `D:\dev\中国专利数据库2024年.csv`：**6,606,164,739字节 → 63,017,682字节**（63.02MB，缩小99.05%）；产物 `data/patent/publication_metadata/publications.2024.g.jsonl.gz` 及manifest，609,700条事件、579,513个申请号，包含37条单按行G过滤会漏掉的已知申请事件。
- 压缩产物全部609,700条重读通过，原表抽12条字段和正文指纹复核通过，导出范围计数与独立全表审计相符；原CSV和G库大小/修改时间均未变。原始文件未删除，训练未启动。精简包是时间/身份辅助材料，不是完整正文备份；仅补日期不等于已验证首次公开或更早正文版本。

## 原始专利数据源已定位（2026-10-20）

- 用户提供 `D:\dev\中国专利数据库2024年.csv`（6.61GB），34列表头含原清洗脚本全部12个必需字段，额外保留公开号、申请/公开/授权精确日期。全表只读扫描3,126,328条CSV记录；分析3,126,326条，另2条额外非空尾字段待审。多数35列记录只是尾部多空字段，不能按34列等长过滤而丢失。
- **按2024年公开公告分包，不是2024年申请包**。非空公告年均2024，申请年覆盖2008—2024。符合旧脚本发明+申请年≥2016规则的1,840,754行，对应1,742,563个不同申请号。全表合规行均匀1000条抽样976条匹配母库，匹配的申请人、IPC与年份完全一致；高度支持同系列/兼容来源，不能证明采购副本逐字节相同。
- 可用同系列2016—2023年公开公告包补历史输入，再由2024年包核验相应记录；仍要区分A/B/更正公开事件、校验申请人与信用码对应、清理派生正文推广语。引用字段仅计数，不能直接当引用边。本轮原文件和旧数据库未改、未启动训练。详见 [`national_patents_2024_review.md`](research_audit/2026-10-20/national_patents_2024_review.md)。

## 用户提供的2024年上市公司专利数据（2026-10-20）

- 来源 `D:\dev\上市公司专利明细_2024年申请.csv`，约2.12GB，全表只读扫描得到 **969,027行、692,874个不同申请号**；全部为2024年申请，公开公告年份跨2024—2026。含精确申请/公开日期、公开公告号、独立授权列、申请人、IPC和正文，但没有统一社会信用代码。
- 属于上市公司及其子公司/合营/联营企业子集。全表均匀1,000行抽样中466行匹配母库，其中G类发明206行有155行匹配G库；不能由记录重合确认同一采购源。存在股票关联重复，证券代码不能替代信用代码；公开公告日期混有B文献授权事件，不能直接当首次公开日期。
- 可作为部分日期补充源，尚未导入或修改原库。优先继续核对网盘的全国/全企业原始版本，不应直接用上市子集替换现有候选池。详见 [`listed_patents_2024_review.md`](research_audit/2026-10-20/listed_patents_2024_review.md)。

## 创新实验数据就绪核查（2026-10-20）

- 按 10 月 15 日最新“固定预算企业证据贡献保持检索＋多正例一致训练”方案核对。重新执行现有合作数据审计通过，7,039 篇输入专利核对母库，15/15 个数据文件指纹与旧审计一致；原数据未改。报告及新审计见 [`research_audit/2026-10-20/data_readiness.md`](research_audit/2026-10-20/data_readiness.md)。
- 本次只读实数：母库 **8,364,662** 篇专利（旧 EXPERIMENTS 的“861.9 万”不是当前实数）；G 库 **2,348,484** 篇。当前输入专利 IPC 字段均非空，可本地派生技术分组。正式实验仍需扩大 300/100/100 原型、建立不按答案选取的历史候选池、导出至多 20 篇逐篇历史证据；这些主要从本地库完成。
- 用户提供原企业目录 `C:\Users\Tang\Desktop\企业数据`：28 个 XLS 表头一致，含曾用名；当前企业池中 9,519 家有非占位曾用名字段，尚未做别名/生效时间核验。没有历史股权控制字段。C 用户常用目录及 D 盘相关目录未定位到原始专利明细 CSV；另找到的专利转让/许可表不直接当共同研发标签。
- 公开号/最早公开日期仍需原导出或公开数据补充；引用图不是当前主方法启动必需项。本地 torch 为 CPU 版且训练依赖不全；本轮仅审计和查找原文件，没有训练或修改现有模型/数据。

## 合作伙伴推荐首版（2026-10-14）

- **任务已改为企业→企业伙伴**：输入企业当年之前已公开／公告的 G 专利，答案为之后数据库首次观察到共同申请的企业。限定企业—企业、双方原申请人全名匹配；首次仅限现有库可观察范围，不能宣称首次真实商业／研发合作。
- **实际数据已生成**：独立 `data/patent/cooperation_v1/`。train/val/test 目标2022/2023/2024，输入分别≤2021/2022/2023；查询300/100/100，试验候选638/323/374。只使用目标年内已经公开／公告的标签记录；原库只读挖出11,646个名称和明显同集团筛查后的企业对，未视为全量核验金标。
- **数据来源及分词核对通过**：7,039个不同输入专利逐一与母库核对申请／公告年份、申请人及正文前缀；500个查询含789条有向正例、695个跨划分不重叠的无向企业对。500条均已离线分词，`data_audit.json` 的 `tokenized_checked=true`；152组真实文本／特殊字符与本地 `tokenizer.json` 的token IDs完全一致（`tokenizer_reference_check.json`）。36项协议测试通过。
- **可运行CPU检索器完成**：中文bigram BM25，过滤自身、已知旧伙伴及明确品牌同集团疑似。val/test 试验候选池 Recall@20=0.3164/0.3759，Recall@100=0.6721/0.6377；test P@1=0.16、MRR=0.2400，100查询中50条Top20至少命中1位伙伴。排名只用历史文本，答案仅用于评测。
- **GraphFormers新训练入口已准备**：`src/train_cooperation.sh` 从原始中文底座初始化，独立输出目录、max_len256、batch1×累积16、hn4、每epoch验证保存。本地torch仅CPU且缺transformers等训练依赖，尚未启动新任务GraphFormers训练；旧专利ckpt含后来数据，不用于新时间划分实验。
- 说明及命令见 [`data_pipeline/COOPERATION_V1.md`](data_pipeline/COOPERATION_V1.md)。从 `src/` 执行 `bash run_cooperation_v1.sh` 准备／核验／评测；在配置好的GPU环境加 `TRAIN=1` 可训练。
- **限制**：试验候选池按标签端点+随机补充构建，不能称2.56万家全库召回；候选快照、名称旧名漏标、公告年份非首次公开日、同集团仅启发式筛查、负候选只是未观察关系等限制写入数据manifest。

## 真实合作标签样本（2026-10-14）

- 新增独立样本包 `data/patent/cooperation_pilot/`：`public_examples.jsonl` 含 **8 条有公开证据的合作／原始共同申请关系**（4 企业—企业、4 企业—高校）；`local_candidates.jsonl` 含本地抽出的 10 条候选（含 2 条同集团风险），其中 3 条已与公开专利原始申请人和日期交叉核验。两类文件有关系重叠，不能相加当独立样本。
- 具体可用案例：百度网讯—浙江恒逸石化（CN115931905A，原始专利首页及技术标准说明）；华为—科大讯飞（官方联合产品发布）；国能榆林能源—百度网讯（CN115983456A）；厦门大学—宁德时代（CN117054527A）。证据类型、来源链接、日期及机构匹配状态见样本包 `README.md`。
- **规模只代表候选**：只读复算 G 库 2,348,484 篇专利、2,703,805 条链接；283,211 篇关联至少两个不同机构代码，222,077 篇的至少两个机构位于现有 25,598 候选池。未全量核查名称代码错配、同集团、转让及重复机构对，不能称为真实合作金标数量。
- **尚不是正式训练评测集**：当前只有关系证据，首次合作日期、控制关系和合作前可用专利组合尚未核实。共有专利、联合研发、联合产品及实验室成立分别标注；未观察到合作不能直接当真实负例。未来合作评测不能把作为答案的共同专利／合作新闻／合作边放进模型输入。
- 从 `src/` 执行 `python ../data/patent/cooperation_pilot/build_pilot.py` 可离线重建导出；脚本仅只读匹配本地候选池，公开证据是本次人工核验记录。采集、重建不代表新模型已训练或验证。

## 引用邻居小样本（2026-10-13）

- **200 篇可用性检查完成**：固定 seed=42、train/val/test=66/67/67，覆盖 165 个 IPC 主分类号 / 30 个 IPC 小类。全部 200 篇成功解析，最终 0 失败 / 0 待处理；129 篇（64.5%）列出直接引用，71 篇来源未列出。每篇最多 5 条，共选出 475 条引用关系（474 个不同公开号），全部取得摘要或本地完整文本；87 篇至少 3 条，63 篇（31.5%）达到 5 条。
- 完整邻居中 441 条按字符判为中文为主、34 条为外语为主；127 篇至少有 1 条中文为主完整邻居。train/val/test 引用覆盖分别为 54/66（81.8%）、43/67（64.2%）、32/67（47.8%），不能只看 train 成功样本推断覆盖。**采集路线可用，建议引用优先、IPC 补足，并保留允许空槽的纯引用对照**；200 篇验证的是数据流程，不是模型效果。
- 独立产物 `data/patent/citation_audit_200/` 含 `sample_manifest.json`、`audit_state.json`、`neighbors.jsonl` / `neighbors.csv`、`audit_report.md` / `audit_metrics.json`；首轮统计另存 `initial_audit_metrics.json`。失败和零引用均未补抽；重试修复 2 次 SSL 中断和 1 篇分案检索定位，外文原语言页面补齐摘要。累计 938 次网络请求，约 28.8 分钟。
- 续跑：从 `src/` 执行 `python ../data_pipeline/audit_citation_neighbors.py --sample 200 --resume`；重试失败或正文缺失可加 `--retry-failed --retry-incomplete-text`。申请号、公开号严格核验；分案仅通过明确原申请映射取得候选，外语文本保留原文、未自动翻译。
- 最终核验通过：抽样清单、状态和 JSONL 的 200 篇映射一致，CSV 475 行与关系记录一致；独立复算总体 / split / 中文字符统计无差异，跨 3 个 split 抽查 5 条缓存关系均为直接引用。离线续跑 0 网络请求，JSONL、CSV 和清单字节完全一致。
- 新增 `data_pipeline/fetch_citation_neighbors.py`：默认仅取 3 篇查询，每篇最多 5 个直接引用邻居，申请号经公开检索转换成公开号并核验；Python 3.8+ 标准库，缓存复用、请求数量限制、SQLite 只读。
- 已实际采集 train 分组前 3 篇：CN114120438A / CN114112197B / CN115357112A，来源页面直接引用分别 3 / 4 / 0 条，共 7 条；6 条有摘要或本地完整文本，1 条日本文献仅标题。0 条仅表示来源未列出。
- 产物：`data/patent/citation_pilot/{neighbors.jsonl,neighbors.csv,summary.md}`；网页缓存：`data_pipeline/build_cache/citation_html/`。现有 IPC 邻居及训练数据未修改。
- 离线重跑验证记录一致、每查询 5 个邻居槽位；使用说明见 `data_pipeline/CITATION_NEIGHBORS.md`。从 `src/` 运行 `python ../data_pipeline/fetch_citation_neighbors.py --sample 3 --neighbors 5`。
- **尚不能直接用于历史评测**：引用关系首次公开时间未知，全部标记 `temporal_safety_verified=false`；`--before` 仅过滤被引用文献公开日期。新方案尚未训练或评估效果。

## 历史申请人匹配实验：已收口（下表为原任务结果）

| 阶段 | 状态 | 产物 | 指标 |
|---|---|---|---|
| ① 预训练（MLM+对比，中文底座→专利语料） | ✅ 2026-08-25 完成（30h） | `ckpt/patent/pretrain/graphformer/1e-5/` | train_loss 1.29；val MRR **0.83** / NDCG@10 **0.87**。**不受泄漏影响** |
| ② 检索训练（泄漏版） | ❌ 作废，已存档 | 本地 `ckpt/patent/nc_retrieval_leaked/`（scp 于 08-28） | P@1 0.39 / MRR 0.63 —— 泄漏灌水，仅作对照组 |
| ②' 检索训练（修复版） | ✅ 2026-08-28 完成（10h02m，1231 步） | `ckpt/patent/nc_retrieval/graphformer/1e-5/` | eval_loss 1.30；**P@1 0.60 / MRR 0.75 / NDCG@10 0.81**（真实值，且高于泄漏版） |
| ③ 检索推理 → FAISS 检索 + recall | ✅ 2026-09-07 重跑完成（`retrieve2.log`） | `node_label_embed/` + trec 结果 | **稠密 recall@50 0.343 / @100 0.432**（n=318,596）；BM25 本地补算（`data_pipeline/bm25_recall.py`，与训练负例同口径）recall@20 0.013 / @50 0.022 / @100 0.034 / @200 0.051 —— **粗筛必须用稠密**；BM25 仅 12 倍弱于稠密，只配当负例来源 |
| ④ 重排训练（100k 样本） | ✅ 完成（08-29，6h16m，781 步） | `ckpt/patent/nc_rerank/graphformer/1e-5/checkpoint-500` | 底座用 ②' 的 checkpoint；train_loss 0.2087；**无训练中 eval**（781 步 < eval_steps 1000） |
| ⑤ 重排测试 | ✅ 2026-09-05 完成（38h41m，159,298 步） | `rerank_test.log` | 全量 **P@1 0.817 / MRR 0.876 / NDCG@5 0.891 / NDCG@10 0.901**（冒烟 0.842/0.893/0.914，全量略降正常——test 每查询最多 10000 候选） |
| ⑥ 结果回传本地 | ✅ 完成（2026-09-09 补齐） | 本地 `ckpt/patent/` + `logs/patent/` | pretrain/检索/重排/消融 ckpt 与全部日志到齐（含 ③recall、⑧v2 负结果日志）；服务器 tokenize 数据（33GB）未回传（本地有文本版可重建） |
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
