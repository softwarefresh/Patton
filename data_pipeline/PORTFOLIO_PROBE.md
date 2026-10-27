# 企业多技术方向表示探针（val-only）

状态：first6/center/center_history5 三次只读探针均完成（2026-10-27），最终判定 `no_reliable_signal`。多技术方向表示在当前任务上收口，不进入训练版或压缩蒸馏。

## 最终判定（2026-10-27，val）

| 查询 | mean R@100 | 最好的 IPC 版本 | vs 同模式 mean | vs 随机分组 | 相对 first6 均值 16.35% |
|---|---:|---:|---:|---:|---:|
| first6 | 16.35% | 14.81% | −1.54 pt，CI 下界 −2.91 | +0.19～+0.49，CI 跨 0 | −1.54 |
| center | 8.46% | 9.44% | +0.98，CI [−0.03, +2.01] | +1.06，CI [+0.21, +1.92] | −6.91 |
| center_history5 | 14.24% | 13.63% | −0.61，CI 跨 0 | +0.83，CI 跨 0 | −2.72 |

- 只有 `center` 窄查询下出现 IPC 分组显著赢随机分组的相对信号，但该查询基线仅 8.46%，绝对性能远低于 first6 均值。
- `center_history5` 保留了上下文，但 IPC 分组重新输给 mean，且不再显著赢随机分组；11–20 桶的增益消失。
- 没有任何模式同时满足：赢同模式 mean、赢随机分组、绝对 R@100 超过 16.35%。
- 结论：不训练多原型/注意力聚合；不做压缩蒸馏或删除响应保持；不把少数方向相对效应包装成创新。

## 第一轮结果（2026-10-27，val）

- 两个基线精确复现：`mean/max` R@100 = 0.16352341，`per_patent/max` = 0.07686405。
- 最好的 IPC 多方向版本 `ipc_rare/max` R@100 = 0.1481，比均值低 1.54 个百分点（bootstrap CI 下界 −0.0291）；`ipc_top4/max` = 0.1474。
- IPC 分组相对随机同尺寸分组仅 +0.19～+0.49 个百分点，CI 跨 0；分类分组没有显著独立价值。
- 少数方向损失较小（`ipc_rare/max`：少数 −1.05 pt vs 多数 −4.04 pt），但绝对召回仍低于均值（17.48% vs 18.52%）；这是“没那么差”，不是“更好”。
- 结论：不启动训练版、压缩蒸馏或删除响应保持。先做 query 侧诊断，排除 first-6 均值稀释焦点专利的混淆。

## 要回答的问题

在企业历史专利很多、查询只对应其中某一项技术方向时，**固定 R 个向量/企业**的多方向表示，能否比单向量均值找回更多正确伙伴？如果能，增益是否来自“保留查询需要的那一个方向”，而不是仅仅因为向量变多？

## 不做的事

- 不训练任何参数；全部使用 `artifacts/partner_vectors` 的冻结向量。
- 不访问 test；只读 `cooperation_v2` 的 val 划分。
- 不修改规范数据、不重编码、不生成新标签。
- 不把 val 上的探索性结果当作最终显著性证明。

## 表示与对照

| 表示 | 说明 |
|---|---|
| `mean` | 每企业 1 个向量：所有保留专利的原始均值 |
| `per_patent` | 每专利 1 个向量，最多 20 个；已有负对照 |
| `ipc_top4` | 按 `technical_group`（IPC 小类）分组；最多 4 组：前三大组 + 其余合并 |
| `ipc_rare` | 前两大组 + 最“独特”的剩余组 + 其余合并；测试稀有方向保护 |
| `random_top4_s*` | 与 `ipc_top4` 完全相同的组大小，但随机分配专利（3 seeds） |
| `random_rare_s*` | 与 `ipc_rare` 完全相同的组大小，但随机分配专利（3 seeds） |

聚合方式：

- `max`：组间最大余弦；
- `top2`：组间前二余弦平均；
- `softmax_T`：`T * log(mean_g exp(cos_g / T))`，在 max 与 mean 之间插值；T 网格 `0.01,0.03,0.1,0.3,1.0,3.0`，只在 val 上选，属于探索性结果。

查询侧固定为现有 first-6 原始均值，保证候选表示是唯一变量。

## 分层诊断

- **少数方向**：用查询中心专利的 IPC 与候选企业完整 IPC 组做重叠匹配；匹配组占该企业专利比例 < 25% 视为少数方向。另报告 1/3、50% 敏感性。
- **专利数桶**：2–5、6–10、11–20。
- **Top100 组成**：平均候选专利数、11 篇以上候选占比，用于排除“大企业因向量多而占便宜”。
- **bootstrap**：query 级配对重采样 1000 次，报告 R@100 差值的 95% 区间；不做多重比较校正。

## 判定规则（预先写死）

主判定用无温度选择的 `max`：

- `ipc_top4/max` 与 `ipc_rare/max` 分别与 `mean/max`、匹配随机分组比较；
- **Go**：整体 R@100 的 bootstrap 95% CI 下界 > 0（同时赢均值和随机分组），且少数方向增益 > 多数方向增益，且各专利数桶没有明显负向（> -0.005）；
- **Stop**：IPC 分组 ≈ 随机分组，或少数方向层没有增益；
- softmax/top2 的结果作为次要信号；若只有次要变体通过，标记为 `secondary_signal_only`，不直接宣布成功。

## 服务器运行

代码更新后（`git pull --ff-only origin patent`），在仓库根目录：

```bash
nohup bash src/run_partner_portfolio_probe.sh > logs/partner_v2/portfolio_probe.log 2>&1 &
tail -f logs/partner_v2/portfolio_probe.log
```

运行脚本的第二个参数是查询模式（默认 `first6`）：

```bash
# 中心专利查询
nohup bash src/run_partner_portfolio_probe.sh logs/partner_v2/portfolio_probe.center.val.json center > logs/partner_v2/portfolio_probe.center.log 2>&1 &
# 中心专利 + 5 篇历史
nohup bash src/run_partner_portfolio_probe.sh logs/partner_v2/portfolio_probe.center_history5.val.json center_history5 > logs/partner_v2/portfolio_probe.center_history5.log 2>&1 &
```

默认输出：

- `logs/partner_v2/portfolio_probe.val.json`：完整结果；
- `logs/partner_v2/portfolio_probe.val.summary.md`：主表、关键 bootstrap 和判定，便于直接查看/粘贴。

脚本会用已知的两个基线做复现检查：

- `mean/max` 的 val R@100 = `0.16352341425902947`
- `per_patent/max` 的 val R@100 = `0.0768640544991725`

容差 `0.001`。如果检查失败，脚本会中止；先不要解释新方法，先排查数据/缓存/打分路径。

## 预期成本

- 9 个表示 × 8 个聚合变体，共享同一批余弦分数；
- 4090 上预计数分钟到十几分钟；CPU 不建议跑全量；
- 主要开销是加载 20,389 个候选的 20 槽证据和每表示一次分组向量构建。

## 结果怎么读

- 先看 `baseline_reproduction` 是否为 `passed`；
- 再看 `decision.decision` 与 `decision.candidates`；
- 主表看 `ipc_top4/max`、`ipc_rare/max` 是否同时高于 `mean/max` 和对应的 `random_*`；
- 分层表看少数方向正例召回是否比多数方向更受益；
- 若整体没有信号，但 `per_patent/max` 也如预期远低于均值，说明当前冻结表示/候选池下“多向量直接打分”确实不成立；不要直接启动训练版或压缩蒸馏。

## 如果探针没有信号

2026-10-27 已完成下面的 query 侧诊断；`center` 和 `center_history5` 都没有产生可部署增益，按第 3 条收口。以下步骤保留为以后复现或在其他任务上检查的清单，不在当前任务继续执行：

1. **查询侧诊断（已纳入脚本）**：用中心专利或“中心 + 5 篇历史”替代 first-6 均值，只读重跑同一套候选表示。
   - 若中心查询下均值基线明显变化，说明 first-6 均值确实稀释了焦点专利；此时先解决查询表示，而不是继续加候选原型。
   - 若中心查询下 IPC 分组仍不胜均值和随机分组，停止这条路线。
   - 非 first6 模式不做预期基线复现检查；直接比较主表中的 `mean` 和 `ipc_*` 行。
2. 加入 k-means/文本聚类原型，判断 IPC 分组是否只是分组方式之一。
3. 若 IPC 分组仍不胜随机分组，停止“多技术方向表示”作为创新主线，保留为工程分析。
