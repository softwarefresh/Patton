# 新服务器实验操作手册（2026-10-26）

适用当前机器：优云智算，RTX 4090 24GB，用户终端为Python 3.12，驱动报告610.57.04。仓库沿用 `softwarefresh/Patton`，用户已确认Public，并明确要求数据不加密、直接上传。代码推送到 `patent` 分支，数据与原始中文底座直接放到公开Release附件。

本地已完成：传输包27文件（20个数据/报告文件、7个原始中文底座文件），未压缩2,549,000,484字节，压缩1,095,628,722字节，3个≤512MiB分片，全部文件核验通过。路径为 `D:\dev\Patton\tmp\partner_release`。本次只上传其中3个 `partner-assets.tar.gz.part001/002/003` 和 `partner-transfer.json`，不上传旧的本地警示README。原始数据库、年度CSV、旧模型、词法索引和读取缓存没有打包。

本手册的首轮方法是“冻结中文编码器→逐篇专利向量→查询前6篇/候选至多20篇均值→多正例投影训练”。它是检索基线。GraphFormers、贡献保持和创新方法尚未接入；原始 `requirements.txt`、`setup.sh`、`train_cooperation.sh` 不用于本流程。任务仍是未来首次观察共同申请伙伴的代理标签，不是具体专利项目的已核实商业合作。

## 1. 本轮使用现有Git上传

本机已有Git，无需再安装GitHub CLI或7-Zip，也无需设置归档密码。本轮由助手检查远程 `patent` 分支、推送代码，并将4个数据附件上传到Release；上传完成后，用户直接从第4步的服务器下载开始。

以下第2—3步保留为未来自行更新的操作说明，不需要在助手已完成本轮上传后再执行一次。

## 2. 上传本次代码

当前本地分支为 `patent`。先查看本次允许提交的源文件，再暂存：

```powershell
cd D:\dev\Patton
powershell -ExecutionPolicy Bypass -File .\data_pipeline\stage_partner_code.ps1
powershell -ExecutionPolicy Bypass -File .\data_pipeline\stage_partner_code.ps1 -Stage
git diff --cached --stat
git commit -m "Add v2 retrieval baseline and 4090 server workflow"
git push origin HEAD:patent
```

辅助脚本只暂存16个小源文件（含词法检索共用模块），不提交数据，也不自动commit/push。如果已有其他暂存文件，它会停止，让你先检查；不要通过 `git add .` 将其他研究文件混入。若push报告远端更新冲突，保留当前内容并处理该提示，不使用强推。

## 3. 通过GitHub网页直接上传数据附件

本次传输包已经生成，无需再次export或加密。未来自行上传时，打开 [仓库Releases页面](https://github.com/softwarefresh/Patton/releases)，选择创建新Release：

1. 标签填写 `partner-v2-20261026`，目标选择已推送代码的 `patent` 分支。
2. 标题填写 `Partner v2 server bundle`。
3. 从本地 `D:\dev\Patton\tmp\partner_release` 拖入以下4个文件，等待上传完成。
4. 发布Release。

附件名称必须保留：

```text
partner-assets.tar.gz.part001
partner-assets.tar.gz.part002
partner-assets.tar.gz.part003
partner-transfer.json
```

这4个附件会直接公开，不需要解密。GitHub普通Git限制单文件100MiB，Release附件单文件须小于2GiB；这也是采用分卷附件的原因。`git clone`只获取代码，不会自动获取这些附件。[GitHub大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Release限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

如果提示版本已存在，先确认是不是先前已经上传成功；不要创建同名的不同数据版本。重做时换一个新标签，并同步替换以下下载URL和clone标签。

## 4. 在服务器下载代码和数据

以下全部在优云智算服务器终端执行。`/workspace`是用户当前工作目录，本手册以它为例；先查看空间及挂载：

```bash
df -h /workspace
findmnt -T /workspace
```

完整环境、下载及解包副本、运行缓存和编码向量建议预留至少20GB可用空间。持久化范围取决于平台可用区和挂载；需要长期保留的数据/训练输出放到已确认的持久化存储。实例删除前另行备份结果。[平台存储差异](https://www.compshare.cn/docs/operation/introduce/region)

```bash
apt-get update
apt-get install -y git wget tmux
cd /workspace
git clone --depth 1 --branch partner-v2-20261026 https://github.com/softwarefresh/Patton.git Patton
mkdir -p /workspace/partner_downloads
cd /workspace/partner_downloads
wget -c https://github.com/softwarefresh/Patton/releases/download/partner-v2-20261026/partner-transfer.json
for part in 001 002 003; do
  wget -c "https://github.com/softwarefresh/Patton/releases/download/partner-v2-20261026/partner-assets.tar.gz.part$part"
done
```

网络中断可重跑对应wget命令继续下载。clone指定本次版本标签，会显示detached HEAD，这是在使用固定实验版本；无需切换回旧main分支。已有 `/workspace/Patton` 时不要覆盖原目录，先确认它是什么版本，或选一个新的安装目录并同步修改路径。

```bash
cd /workspace/Patton
python -B data_pipeline/package_partner_data.py verify --bundle-dir /workspace/partner_downloads
python -B data_pipeline/package_partner_data.py install --bundle-dir /workspace/partner_downloads --repo-dir /workspace/Patton
```

verify成功应显示 `status: verified`，它与install都只需要Python标准库，会自动连接分片并检查每个归档文件，无需手动拼接或解压。install拒绝覆盖已有的 `data/patent/cooperation_v2` 或 `ckpt/chinese-roberta-wwm-ext`，父目录可存在。安装后的数据保持本地已审计的字节和格式，没有重建/修改标签。

## 5. 建立独立环境并做实际GPU检查

```bash
cd /workspace/Patton
bash src/setup_partner_4090.sh
source /workspace/Patton/.venv_partner4090/bin/activate
```

脚本创建项目独立环境，安装 **torch2.6.0 cu124、transformers4.44.2、numpy1.26.4**，检查依赖并在CUDA执行矩阵乘法与反向传播，结果写入 `tmp/server_environment.json`。nvidia-smi显示CUDA13.3是驱动提供的能力，不要求安装CUDA13.3工具链；官方提供的cu124轮子可在兼容新驱动上运行。[PyTorch安装版本](https://docs.pytorch.org/get-started/previous-versions/)、[NVIDIA兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/610/minor-version-compatibility.html)

新终端/新tmux窗口都要再次source环境。如果venv模块缺失，先安装该镜像Python3.12对应的venv组件再重跑；不要退回旧setup.sh。脚本不会更改平台NVIDIA驱动，也不需要faiss、datasets或torchvision。

## 6. 检查真实数据输入，跑小样本编码

```bash
cd /workspace/Patton
source .venv_partner4090/bin/activate
python -B -m unittest discover -s tests -p 'test_*partner*.py'
python -B data_pipeline/partner_evidence_dataset.py inspect --data-dir data/patent/cooperation_v2 --split train --index 0
python -B data_pipeline/partner_evidence_dataset.py inspect --data-dir data/patent/cooperation_v2 --split val --index 0
python -B data_pipeline/partner_evidence_dataset.py inspect --data-dir data/patent/cooperation_v2 --split test --index 0
```

应看到测试OK，三个dataset_size依次为1341、1732、2112。首次inspect会从共享token重建约0.30GB读取缓存，后两次复用。包内已有原始审计报告；不在服务器运行依赖原数据库的全源审计程序。

```bash
python -u data_pipeline/partner_dense_baseline.py encode \
  --data-dir data/patent/cooperation_v2 \
  --model-dir ckpt/chinese-roberta-wwm-ext \
  --output-dir artifacts/partner_smoke \
  --batch-size 16 --max-patents 64 --device cuda
```

这一步实际加载中文底座，在4090编码64篇，验证权重、tokenizer、输入和GPU。小样本输出标记smoke_only，禁止用于正式训练或评测；下一步使用不同目录。现有token已经含CLS/SEP，程序不再添加。

## 7. 在tmux中完成正式编码、训练和评测

```bash
tmux new -s partner
cd /workspace/Patton
source .venv_partner4090/bin/activate
python -u data_pipeline/partner_dense_baseline.py encode \
  --data-dir data/patent/cooperation_v2 \
  --model-dir ckpt/chinese-roberta-wwm-ext \
  --output-dir artifacts/partner_vectors \
  --batch-size 16 --device cuda
```

正式编码434,047篇专利，float32向量约1.33GB，按磁盘映射读取。遇到显存不足，使用8的batch和新的输出目录重试；不要降低证据篇数或混用smoke缓存。编码结果目录已存在时程序拒绝覆盖，避免将中断输出当完整缓存。编码完成后继续：

```bash
python -u data_pipeline/partner_dense_baseline.py evaluate \
  --data-dir data/patent/cooperation_v2 \
  --embedding-dir artifacts/partner_vectors \
  --split val --output logs/partner_v2/frozen.val.json --device cuda

python -u data_pipeline/partner_dense_baseline.py train \
  --data-dir data/patent/cooperation_v2 \
  --embedding-dir artifacts/partner_vectors \
  --output-dir ckpt/partner_dense/run1 \
  --epochs 10 --batch-size 32 --seed 42 --device cuda

python -u data_pipeline/partner_dense_baseline.py evaluate \
  --data-dir data/patent/cooperation_v2 \
  --embedding-dir artifacts/partner_vectors \
  --checkpoint ckpt/partner_dense/run1/best.pt \
  --split test --output logs/partner_v2/projection.test.json --device cuda
```

无checkpoint的evaluate是原始冻结均值控制；训练仅更新共享投影层、保留全部正例。每轮在完整验证候选池计算Recall@100选best，epoch0也参与比较；如投影未改善，best可以仍是epoch0。测试集不参与选轮次。需要训练后验证结果时可用同一best.pt、`--split val`及新输出文件评测。

为完成同条件控制比较，按预定方案评测原始冻结模型的test一次：

```bash
python -u data_pipeline/partner_dense_baseline.py evaluate \
  --data-dir data/patent/cooperation_v2 \
  --embedding-dir artifacts/partner_vectors \
  --split test --output logs/partner_v2/frozen.test.json --device cuda
```

这些测试结果不用于反复调参。新任务指标只与同协议、同候选池的方法比较；先前所属企业匹配及v1小池结果不能作直接提升基准。

按Ctrl+B，松开，再按D离开tmux，任务会继续。返回：`tmux attach -t partner`。在另一个终端用 `nvidia-smi` 查看GPU占用。第二次实验使用新的run2及报告文件名，编码向量可复用，且程序会验证缓存和源数据指纹。

## 8. 保存实验结果

本轮保留：`tmp/server_environment.json`、`tmp/partner_environment_freeze.txt`、`ckpt/partner_dense/run1/` 下的run_config/training_history/training_summary/best.pt，以及 `logs/partner_v2/` 评测报告。记录服务器 `git rev-parse HEAD`。将这些结果下载回本机或复制到确认的持久化存储，之后再处理实例。没有使用含后续数据的旧checkpoint。

当前本地验证覆盖小样本CPU假编码器端到端、标签隔离、多正例、缓存指纹、打包/解包及安全拒绝；4090上的安装、真实中文底座编码和正式训练需按上面的步骤实际运行后才能确认。
