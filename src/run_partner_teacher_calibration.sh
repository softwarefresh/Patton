#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
PROJ_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"
RUN_NAME="${1:-dense1}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf 'Use a run name containing letters, digits, underscores or dashes.\n' >&2
    exit 1
fi
PYTHON_BIN="$PROJ_DIR/.venv_partner4090/bin/python"
RUN_DIR="$PROJ_DIR/ckpt/partner_teacher/$RUN_NAME"
OUTPUT_FILE="$PROJ_DIR/logs/partner_v2/teacher_${RUN_NAME}.calibration.val.json"
LOG_FILE="$PROJ_DIR/logs/partner_v2/teacher_${RUN_NAME}.calibration.log"
if [[ ! -x "$PYTHON_BIN" || ! -f "$RUN_DIR/mean/last.pt" || ! -f "$RUN_DIR/attention/last.pt" ]]; then
    printf 'The existing environment and both diagnostic last.pt files are required.\n' >&2
    exit 1
fi
if [[ -e "$OUTPUT_FILE" || -e "$LOG_FILE" ]]; then
    printf 'Calibration output already exists; inspect it before rerunning.\n' >&2
    exit 1
fi
mkdir -p logs/partner_v2
nohup "$PYTHON_BIN" -u data_pipeline/partner_teacher_calibration.py \
    --data-dir data/patent/cooperation_v2 \
    --embedding-dir artifacts/partner_vectors \
    --run-dir "$RUN_DIR" \
    --base-checkpoint ckpt/partner_dense/run1/best.pt \
    --output "$OUTPUT_FILE" --device cuda --batch-size 8 --candidate-batch-size 512 \
    > "$LOG_FILE" 2>&1 < /dev/null &
CHECK_PID=$!
printf 'Validation-only calibration started. PID: %s\nLog: %s\nReport: %s\n' "$CHECK_PID" "$LOG_FILE" "$OUTPUT_FILE"
