# 长文本重建与重新编码计划

状态：第 1 步（恢复完整原文）和第 2 步（新 tokenizer 长度审计）脚本已准备；编码脚本等长度分布出来后按实际覆盖选择 max length。

## 为什么要做

当前专利表示有**两层截断**：

| 截断 | 位置 | 影响 |
|---|---|---|
| `text[:1600]` | `build_patent_partner_data.py` 存储 evidence 时 | 2.47% 的专利受影响；完整清洗文本 p99=2,027 字，最大 13,351 字 |
| 256 token | `tokenize_partner_evidence.py` 默认 `--max-length 256` | **99.39%** 的专利正好卡在 256 token，信息损失更大 |

所以不能只把 token 上限从 256 改成 2048；必须先从 `g_domain.db` 恢复完整清洗文本。

## 第 1 步：恢复完整清洗文本

```bash
cd /workspace/Patton
python data_pipeline/build_partner_full_text.py
```

- 读取 `data/patent/cooperation_v2/evidence.jsonl.gz` 的 434,047 个 application；
- 从 `data_pipeline/g_domain.db` 的 `patents_g.q_text` 取原文；
- 用与建库完全相同的 `clean_patent_text` 清洗；
- 核对 `raw_text_sha256` 和 `full_cleaned_text_characters`；
- 输出 `data/patent/cooperation_v2/full_text/evidence.full.jsonl.gz` 和 manifest。

特点：

- 不改规范数据、不读标签、不做 tokenization；
- 每篇记录包含完整清洗文本、`stored_text_characters`、`recovered_characters`；
- 全量预计几分钟量级（取决于磁盘；SQLite 主键查询）。

本地全量实测（2026-10-27）：

| 项目 | 数值 |
|---|---:|
| 专利数 | 434,047 |
| 完整文本超过 1600 字符 | 10,720（2.47%） |
| 找回的字符总数 | 5,702,247 |
| 最大完整清洗长度 | 13,351 字符 |
| 输出大小 | 239,318,921 字节（约 228 MiB gzip） |
| 耗时 | 268.66 秒 |

如果服务器上没有 `data_pipeline/g_domain.db`（7.4GB），就在本地跑第 1 步，把生成的 `evidence.full.jsonl.gz`（约 228MiB）上传到服务器，不要上传 7.4GB 原库。

## 第 2 步：用新模型 tokenizer 统计真实长度

先下载 Qwen3-Embedding-0.6B 的 tokenizer（只需要 tokenizer 文件，不需要权重）：

```bash
mkdir -p ckpt/Qwen3-Embedding-0.6B
# 如果 hf 命令不存在：pip install -U huggingface_hub
hf download Qwen/Qwen3-Embedding-0.6B tokenizer.json tokenizer_config.json \
  --local-dir ckpt/Qwen3-Embedding-0.6B
```

然后审计：

```bash
python data_pipeline/audit_full_text_tokens.py \
  --tokenizer-json ckpt/Qwen3-Embedding-0.6B/tokenizer.json \
  --sample-size 0 \
  --batch-size 2000 \
  --output logs/partner_v2/full_text_token_audit.json
```

输出：

- token 长度分位数（median/p90/p95/p99/max）；
- 512/1024/1536/2048/4096/8192 等上限的覆盖率；
- 字符长度分位数；
- 只读文本，不碰标签。

**不要直接设 2048。** 看覆盖率：

- 如果 1024 已覆盖 ≥99%，用 1024 就够；
- 如果 2048 覆盖 ≥99%，用 2048；
- 如果长尾明显且显存允许，再考虑 3072/4096；Qwen3 支持 32K，但编码成本随之增加。

本地参考：旧中文 BERT tokenizer 的全量审计（434,047 篇，耗时 1,282 秒；**不是 Qwen3 的最终数字**）：

| token 上限 | 覆盖率 |
|---|---:|
| 256 | 0.62% |
| 512 | 32.9% |
| 1024 | 90.0% |
| 1536 | 97.6% |
| 2048 | **99.21%** |
| 3072 | 99.86% |
| 4096 | 99.96% |
| 8192 | 99.998% |

旧 tokenizer 分位数：中位数 587、p90 1024、p95 1256、p99 1927、最大 10,705。  
因此 2048 是合理起点，3072 更保守；最终以服务器上 Qwen3 tokenizer 的审计为准。

## 第 3 步：重新编码（待长度确定后）

编码脚本会遵守：

1. **模型官方向量提取方式**：Qwen3-Embedding 用 last-token pooling + L2 normalize；BGE-M3 用 CLS pooling + normalize；
2. **固定其他条件**：同样的 434,047 篇专利、同样的 train/val/test 时间切分、同样的候选池、同样的 2–20 篇企业专利；
3. **先补长文本均值基线**：只用均值池化，比较旧 256-token 均值 16.35% / 投影 18.38%；
4. **再比较多技术/组合感知 cross-encoder**：必须在同一个强编码器、同样的长文本、同样的候选池下证明增益。

## 重要边界

- Qwen3 预训练时间晚于目标年，严格时间隔离评测必须披露；老编码器作为对照。
- 1600 字符截断只影响 2.47%，但为了不做“只改 token 上限”的假修复，仍然恢复完整原文。
- 长窗口只解决“看见更多内容”；长文本压成一个向量后，技术方向仍可能混合。所以先补均值基线，再谈多技术画像。
- test 不参与任何长度选择或调参；所有选择在 train/val 上做。
