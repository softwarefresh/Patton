# 实验进度（专利→企业合作推荐）

> 本文件记录实验流水线的当前状态，跨会话维护。新开会话先读这里。
> 最近更新：2026-08-26

## 当前状态：② 检索训练（半数据重跑）

| 阶段 | 状态 | 产物 | 指标 |
|---|---|---|---|
| ① 预训练（MLM+对比，中文底座→专利语料） | ✅ 2026-08-25 完成（30h） | `ckpt/patent/pretrain/graphformer/1e-5/` | train_loss 1.29；val MRR **0.83** / NDCG@10 **0.87** |
| ② 检索训练 | 🔄 进行中（重跑第 5 次，~9.6h） | `ckpt/patent/nc_retrieval/` | 等第 1000 步 eval 指标 |
| ③ 检索推理 → FAISS 检索 + recall | ⏳ | logs + trec 结果 | `nc_retrieve_infer_patent.sh` → `nc_retrieve_retrieval_patent.sh` |
| ④ 重排训练（100k 样本，~6h） | ⏳ | `ckpt/patent/nc_rerank/` | 底座用 ② 的 checkpoint |
| ⑤ 重排测试 | ⏳ | P@1/MRR/NDCG | 需把 `nc_rerank_test_patent.sh` 的 STEP 改成最佳存档步数 |
| ⑥ 结果回传本地 | ⏳ 部分 | 本地 `ckpt/patent/` | 预训练 ckpt 是否已 scp 待确认；每阶段结束立刻拉回 |

## 生效配置（12G 3080 Ti 硬约束，勿改回）

- 数据：检索训练用 `train.half.jsonl`（315k 抽 50%）；验证用 `val.small.jsonl` / `val.rerank.small.jsonl`（323k 抽 1/32 ≈ 1 万条——全量验证一次要 3-4h）
- `max_len 256` + **batch 1 × 累积 128** = 有效 batch 128（每样本 36 条子图序列 × ~175MB/条，batch 2 也 OOM）
- 验证 batch 4（16 会 OOM）；`save_steps 500` < `eval_steps 1000`（HF 顺序 log→eval→save，验证 OOM 会连存档一起丢）
- **不用 grad_cache**（max_len 256 下前向缓存一样爆，且拖慢）；脚本统一 `python -u`（stdout 无缓冲，loss 实时落盘）
- 预训练保持 batch 4 × 累积 32；检索/重排训练一律 batch 1 × 累积 128

## 踩坑记录（均已修复在脚本里）

1. 原版 batch 按 max_len 32 设计；256 下注意力矩阵按长度平方增长，显存 8 倍
2. grad_cache 只省反向重算，不省前向缓存
3. nohup 下 loss 行（tqdm.write→stdout）块缓冲，攒到进程结束才落盘；tensorboard 里其实一直有 → `python -u`
4. 存档在验证之后执行：eval OOM 时 9.4h 训练零存档 → save_steps 500
5. 全量 323k 验证集一次 3-4h → 抽 1/32

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
- 检索训练 ~4h 存一次档（500 步）；中途崩溃后从零重跑（无 resume 配置）
