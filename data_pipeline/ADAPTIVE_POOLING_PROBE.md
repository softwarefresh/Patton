# 自适应池化探针：A（专利数分组）+ B（企业类型分组）

状态：代码已准备；尚未在服务器运行。  
目的：验证一个很小的改动——**把“所有企业共用一个打分规则”换成“不同企业组用不同的规则”**，是否能带来稳定的小幅提升。它对应你实验室同学“一个统一阈值换成多个阈值”的思路在 Patton 检索上的版本。

## 1. 改的是什么

现在每家企业用同一个均值向量；查询和候选的分数是余弦相似度。

探针把候选企业的分数改成：

```text
score = (1 - α_g) * 均值分数 + α_g * 最高分
```

- `α_g = 0`：只用均值（当前基线）；
- `α_g = 1`：只用 max（之前的失败对照）；
- 每个组 g 在 0 到 1 之间选一个 `α_g`。

分组方式：

- **A：按企业专利数**：2–5 篇、6–10 篇、11–20 篇；
- **B：按企业类型**：企业、高校、科研院所、未知；
- **A+B：按（专利数 × 类型）组合**，小组回退到全局 α。

## 2. 为什么这算一个小创新

它检验的假设是：

> 不同企业的技术组合规模/类型不同，一个统一的池化规则不是最优；按组校准可以同时保留均值的稳定性和 max 的方向性。

关键对照：

| 方法 | 含义 |
|---|---|
| `mean` | 所有企业 α=0（当前基线） |
| `max` | 所有企业 α=1（负对照） |
| `global` | 所有企业用同一个 α（单阈值） |
| `size_group` | 每个专利数桶一个 α（多阈值 A） |
| `type_group` | 每个企业类型一个 α（多阈值 B） |
| `combined_group` | 每个（专利数 × 类型）一个 α |
| `random_size` / `random_type` | 组大小相同但随机分配候选（3 seeds），控制“分组本身是否真的有用” |

只有 `size_group` / `type_group` 同时满足：

1. 赢 `mean`；
2. 赢 `global`（说明“多个 α”比“一个 α”好）；
3. 赢对应随机分组（说明“有意义的组”比“随便分组”好）；
4. bootstrap 95% 区间下界 > 0；

才算 `candidate_signal`。否则只能算全局混合或噪声。

## 3. 运行

服务器仓库根目录：

```bash
cd /workspace/Patton
git pull --ff-only origin patent
mkdir -p logs/partner_v2

nohup bash src/run_partner_adaptive_pooling.sh \
  > logs/partner_v2/adaptive_pooling.log 2>&1 &

tail -f logs/partner_v2/adaptive_pooling.log
```

跑完看：

```bash
cat logs/partner_v2/adaptive_pooling.val.summary.md
```

输出：

- `logs/partner_v2/adaptive_pooling.val.json`：完整结果；
- `logs/partner_v2/adaptive_pooling.val.summary.md`：主表、分组 α、bootstrap、判定。

## 4. 数据与企业类型

- 只使用 `cooperation_v2` 的 train（2022）和 val（2023）；**test 不访问**。
- 复用 `artifacts/partner_vectors` 的现有冻结向量，不重新编码。
- 企业类型优先读 `data_pipeline/g_company_pool.csv`（`信用代码 → 类型`）。
- 如果没有这个 CSV，脚本会从 `data/patent/cooperation_v2/identity/identities.jsonl` 读企业名称，再用名称启发式：含“大学/学院”→高校，含“研究院/研究所/科学院”→科研院所，否则企业；summary 的 `type_source` 会标记 `identity_name_heuristic`。
- 需要精确类型时，把本地 `data_pipeline/g_company_pool.csv` 上传到服务器同一路径即可；此时 `type_source` 会显示 CSV 路径。

## 5. 成本与预期

- 无神经网络训练；主要成本是加载 train+val 证据、计算 mean/max 分数、在 train 上网格搜索 α。
- 4090 上预计 10–20 分钟；CPU 会很慢，不建议。
- 预期：小改动通常只有 0.5–2 pt；可能是噪声。若分组 α ≈ 随机分组，不能算创新。

## 6. 判定怎么读

- `candidate_signal`：分组 α 同时赢 mean、global、random，且 CI 下界 > 0 → 可以作为小创新继续做（例如升级为轻量 LTR 校准层）。
- `global_only`：一个全局 α 赢 mean，但分组不赢 global/random → 只是“全局混合”，不是多阈值创新。
- `no_signal`：连全局 α 都不赢 mean → 这条小改动也不成立，停止。

## 7. 注意

- α 只在 train 上拟合，val 只用于比较；不要用 val 反复调 α。
- test 不参与调参；最终如果要用，只跑一次。
- 企业类型启发式是近似；如果类型分组的收益主要来自启发式，需要在论文中说明。
- 这个探针不是论文级创新证明；它只回答“值不值得继续做这个方向”。
