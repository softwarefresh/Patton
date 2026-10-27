#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv_partner4090/bin/activate
mkdir -p logs/partner_v2

# usage: bash src/run_partner_portfolio_probe.sh [output] [first6|center|center_history5]
QUERY_MODE="${2:-first6}"
case "$QUERY_MODE" in
  first6|center|center_history5) ;;
  *) echo "unknown query mode: $QUERY_MODE" >&2; exit 1 ;;
esac

if [ "$QUERY_MODE" = "first6" ]; then
  DEFAULT_OUTPUT="logs/partner_v2/portfolio_probe.val.json"
  EXTRA=(--expected-mean-r100 0.16352341425902947 --expected-fullmax-r100 0.0768640544991725)
else
  DEFAULT_OUTPUT="logs/partner_v2/portfolio_probe.${QUERY_MODE}.val.json"
  EXTRA=()
fi
OUTPUT="${1:-$DEFAULT_OUTPUT}"
if [ -e "$OUTPUT" ]; then
  echo "refusing existing output: $OUTPUT" >&2
  exit 1
fi

python data_pipeline/partner_portfolio_probe.py evaluate \
  --data-dir data/patent/cooperation_v2 \
  --embedding-dir artifacts/partner_vectors \
  --split val \
  --query-mode "$QUERY_MODE" \
  --output "$OUTPUT" \
  --device cuda \
  --batch-size 16 \
  --bootstrap 1000 \
  "${EXTRA[@]}"
