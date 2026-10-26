#!/usr/bin/env bash
# Create an isolated environment for the independent cooperation_v2 baseline.
# Run from any directory: bash src/setup_partner_4090.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BASE_PYTHON="${PARTNER_BASE_PYTHON:-python}"
ENV_DIR="$PROJ_DIR/.venv_partner4090"

"$BASE_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11, or 3.12"'
if [[ -e "$ENV_DIR" ]]; then
    if [[ ! -f "$ENV_DIR/.patton-partner-env" ]]; then
        echo "Existing environment is not owned by this setup script: $ENV_DIR" >&2
        exit 2
    fi
else
    mkdir -p -- "$ENV_DIR"
    printf 'patton_partner4090_v1\n' > "$ENV_DIR/.patton-partner-env"
fi
# Retrying after an interrupted venv creation is safe for this owned directory.
"$BASE_PYTHON" -m venv "$ENV_DIR"

PYTHON="$ENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$PYTHON" -m pip install -r "$PROJ_DIR/requirements.partner4090.txt"
"$PYTHON" -m pip check
mkdir -p -- "$PROJ_DIR/tmp"
"$PYTHON" -m pip freeze > "$PROJ_DIR/tmp/partner_environment_freeze.txt"
"$PYTHON" - "$PROJ_DIR/tmp/server_environment.json" <<'PY'
import json
from pathlib import Path
import platform
import sys
import torch
import transformers
import numpy

assert torch.cuda.is_available(), "CUDA unavailable in the new environment"
assert torch.__version__.split('+')[0] == '2.6.0', "Unexpected torch version"
assert torch.version.cuda == '12.4', "Expected the official cu124 torch build"
x = torch.randn(256, 256, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
info = {
    'status': 'passed', 'python': sys.version, 'platform': platform.platform(),
    'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'transformers': transformers.__version__, 'numpy': numpy.__version__,
    'gpu': torch.cuda.get_device_name(0),
    'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    'cuda_forward_backward_checked': True,
    'model_training_performed': False,
}
Path(sys.argv[1]).write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(info, ensure_ascii=False, indent=2))
PY

printf '\nEnvironment ready. Activate it in the current terminal:\n  source "%s/bin/activate"\n' "$ENV_DIR"
