# 服务器运行手册（优云智算 / 3080 Ti / Linux）

本文件是「本地数据准备完成 → 租服务器跑训练」的操作清单。
依据优云智算官方文档（docs.compshare.cn）整理。

## 0. 登录与上传方式

### 0.1 登录信息（不同实例类型不能混用）

| 实例类型 | 用户名 | SSH 端口 | 说明 |
|---|---|---|---|
| 容器实例（基础镜像/社区镜像） | `root` | **23** | 必须 `-p 23`，省略或用 22 会连不上 |
| 虚机实例（Ubuntu 系统镜像） | `ubuntu` | 22 | |

- 密码：控制台「实例列表 → 登录 → 复制 SSH 密码」（与 Jupyter 密码可能不同）
- 命令行登录（容器实例）：
  ```bash
  ssh -p 23 root@<外网IP>
  ```

### 0.2 代码用 git，数据用直传（官方推荐）

**代码 → `git clone`**（仓库已在 GitHub，代码量小，平台对 GitHub/HuggingFace 有学术加速）：

```bash
# 在服务器上执行
git clone https://github.com/softwarefresh/Patton.git
cd Patton
```

**数据 → 不走 git，用 scp / 云存储**（`data/patent/` 51GB、`g_domain.db` 7.2GB 太大且未纳入 git）。

scp 在**本地机器**执行，注意是大写 `-P` + 端口 `23` + 用户 `root`：

```bash
# 容器实例（root + 23）
scp -rP 23 data/patent                      root@<外网IP>:/root/Patton/data/
scp -rP 23 ckpt/chinese-roberta-wwm-ext     root@<外网IP>:/root/Patton/ckpt/
scp  -P 23 data_pipeline/g_domain.db        root@<外网IP>:/root/Patton/data_pipeline/
scp  -P 23 data_pipeline/g_company_pool.csv root@<外网IP>:/root/Patton/data_pipeline/
scp  -P 23 data_pipeline/g_company_info.csv root@<外网IP>:/root/Patton/data_pipeline/

# 从服务器下载回本地
scp -rP 23 root@<外网IP>:<服务器路径> <本地路径>
```

其他直传方式（官方文档列出，任选其一）：
- **云存储**：控制台上传至云储存，挂载该云储存的实例可立即读取（大文件首选，断点续传更稳）
- **JupyterLab**：网页端「登录 → JupyterLab」，左侧目录直接拖入
- **FileZilla / XFTP**：协议 SFTP，主机填外网 IP，端口 23，用户名 root，密码为 SSH 密码

## 1. 环境安装（先换源）

### 1.1 conda 换源（清华源）

写入 `~/.condarc`：

```bash
cat <<'EOF' > ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
conda config --show channels   # 验证是否生效
```

### 1.2 pip 换源（清华源）

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.3 装环境

```bash
bash setup.sh        # conda python3.10 + torch 1.13.1+cu117 + transformers 4.21.1
```

> 注意：`setup.sh` 里 torch 用 `pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html`。
> 清华 pip 源**不含** `+cu117` 变体 wheel，这一步仍走官方 `download.pytorch.org`（平台学术加速通常可覆盖）。
> 若极慢，可改用 conda 装 pytorch（上一步已配清华 pytorch 云镜像）：
> ```bash
> conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.7 -c pytorch -c conda-forge
> ```

## 2. 上传清单

| 内容 | 路径（本地） | 上传方式 | 说明 |
|---|---|---|---|
| 代码 | 整个 repo | **git clone** | 88 个文件，体积小 |
| 工作库 | `data_pipeline/g_domain.db`（7.2GB）| scp / 云存储 | 已重建、已建索引、quick_check ok |
| 中文底座 | `ckpt/chinese-roberta-wwm-ext/`（393MB）| scp | hfl/chinese-roberta-wwm-ext |
| 训练输入 | `data/patent/nc/`、`data/patent/pretrain/`（51GB）| scp / 云存储 | 本地已组装好 |
| 候选池 | `data_pipeline/g_company_{pool,info}.csv` | scp | 备查 |

> 服务器上只训练、不建数据。训练数据（tokenize 后）在服务器上由 `src/build_patent.sh` 生成。

## 3. Tokenize（CPU 可跑）

```bash
bash src/build_patent.sh   # 生成 train.jsonl / train.16.jsonl / train.rerank.32.jsonl / test.rerank.10000.jsonl
```

## 4. 训练顺序

```bash
# ① GPU 冒烟（可选但建议）：先跑 200 步确认显存/速度
# ② 预训练（MLM + 对比，中文底座 → 专利语料）
bash src/run_pretrain_patent.sh

# ③ 检索训练（底座用预训练 checkpoint）
bash src/nc_retrieve_train_patent.sh
# ④ 检索：建企业向量索引 + 测试查询检索 + recall
bash src/nc_retrieve_infer_patent.sh
bash src/nc_retrieve_retrieval_patent.sh

# ⑤ 重排训练（底座用检索 checkpoint）
bash src/nc_rerank_train_patent.sh
# ⑥ 重排测试（把 STEP 改成最佳 checkpoint 步数）
bash src/nc_rerank_test_patent.sh
```

## 5. 关键配置（3080 Ti 12GB 适配）

- `max_len 256`（原版 32，中文专利文本长）
- `fp16 + grad_cache` 全开
- batch 4-8 + 梯度累积（显存不够就把 batch 减半、累积翻倍）
- 单卡：`CUDA_VISIBLE_DEVICES=0`，未用 `negatives_x_device`（多卡专属）

## 6. 数据口径备忘

- 查询：2022-2024 专利，assignee 在候选池（25,598 家）
- 时间切分：2022=train / 2023=val / 2024=test（时序评估，防泄漏）
- 预训练对：申请年份 ≤2023 的全部池内链接（排除测试年）
- 检索负例 8 个（4 BM25 硬 + 4 随机）；重排负例 20 个纯 BM25 硬
- 邻居：q_n 同 IPC ×5；k_n 企业专利组合 ×3 + 合作方 ×2（负例的 k_n 留空）
- 指标：检索 recall@50/100；重排 P@1/MRR/NDCG（模型自算）

## 7. 已知注意点

- `g_domain.db` 曾因 journal_mode=OFF 下杀进程损坏，已从母库重建。**任何写库操作不要中途强杀**；母库 `data_pipeline/patents.db`（25GB）不要动。
- `bm25/trec_eval` 二进制缺失，检索 recall 用 `src/scripts/eval_trec.py`（纯 Python）替代。
- 服务器上旧训练脚本（`run_pretrain.sh` 等）是原版英文领域的，专利项目一律用 `*_patent.sh`。
- 按量计费下**关机不收费**，但关机后 7 天实例自动释放、系统盘数据清空——**结果记得下载回本地**（重要 checkpoint 用 scp 拉回，或挂载云存储持久化）。
