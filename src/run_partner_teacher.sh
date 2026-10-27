#!/usr/bin/env bash
set -euo pipefail
set -o noclobber

PROJ_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_DIR"
RUN_NAME="${1:-run1}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf 'Use a run name containing letters, digits, underscores or dashes.\n' >&2
    exit 1
fi
PYTHON_BIN="$PROJ_DIR/.venv_partner4090/bin/python"
BASE_CHECKPOINT="$PROJ_DIR/ckpt/partner_dense/run1/best.pt"
OUTPUT_DIR="$PROJ_DIR/ckpt/partner_teacher/$RUN_NAME"
LOG_FILE="$PROJ_DIR/logs/partner_v2/teacher_${RUN_NAME}.log"
if [[ ! -x "$PYTHON_BIN" || ! -f "$BASE_CHECKPOINT" || ! -f artifacts/partner_vectors/embedding_manifest.json ]]; then
    printf 'The partner4090 environment, run1 dense checkpoint and complete vector cache are required.\n' >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" || -e "$LOG_FILE" ]]; then
    printf 'Run already exists. Inspect its results before choosing a new run name: %s\n' "$RUN_NAME" >&2
    exit 1
fi
mkdir -p logs/partner_v2
nohup "$PYTHON_BIN" -u data_pipeline/partner_evidence_teacher.py train-pair \
    --data-dir data/patent/cooperation_v2 \
    --embedding-dir artifacts/partner_vectors \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 10 --batch-size 32 --seed 42 \
    --eval-batch-size 8 --candidate-batch-size 512 --device cuda \
    > "$LOG_FILE" 2>&1 < /dev/null &
TRAIN_PID=$!
printf 'Training started. PID: %s\nLog: %s\nResults: %s/comparison.json\n' "$TRAIN_PID" "$LOG_FILE" "$OUTPUT_DIR"
printf 'Monitor: tail -f "%s"\n' "$LOG_FILE"
