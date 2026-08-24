#!/bin/bash
# 专利 Patton 训练环境一键安装 (Ubuntu 服务器, 3080 Ti / sm_86)
# 用法: bash setup.sh
# 注意: torch 用 1.13.1+cu117 (30系卡 sm_86 兼容, 与 transformers 4.21.1 匹配)
#       原 repo 钉的 torch==1.8.0 对 30 系卡支持差, 不要用

set -e

# ---- 用 conda 建 Python 3.10 环境 (也可直接用系统 python3.10)
if command -v conda &>/dev/null; then
    # 环境已存在则跳过创建（幂等，可重复运行）
    conda env list | grep -qw patton || conda create -n patton python=3.10 -y
    # 激活环境：先 source conda.sh 定义 shell 函数，否则脚本里 conda activate 可能失效
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate patton
    PY="python"
elif command -v python3.10 &>/dev/null; then
    PY="python3.10"
    $PY -m venv venv && source venv/bin/activate && PY="python"
else
    PY="python3"
fi

echo "Python: $($PY --version)"

# ---- PyTorch (CUDA 11.7)
$PY -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ---- 其余依赖
$PY -m pip install \
    transformers==4.21.1 \
    datasets==2.11.0 \
    fsspec==2023.6.0 \
    pyarrow==12.0.1 \
    numpy==1.23.5 \
    scikit-learn==1.2.2 \
    scipy==1.10.1 \
    tensorboard==2.12.1 \
    sentencepiece==0.1.97 \
    rank_bm25==0.2.2 \
    faiss-cpu==1.7.4 \
    grad-cache==0.0.3 \
    ipython \
    tqdm

# ---- 验证
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$PY -c "import transformers; print('transformers', transformers.__version__)"
echo "环境安装完成。上传代码与数据后: cd Patton && bash src/run_pretrain_patent.sh"
