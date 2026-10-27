#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv_partner4090/bin/activate
mkdir -p logs/partner_v2

OUTPUT="${1:-logs/partner_v2/adaptive_pooling.val.json}"
if [ -e "$OUTPUT" ]; then
  echo "refusing existing output: $OUTPUT" >&2
  exit 1
fi

python data_pipeline/partner_adaptive_pooling_probe.py evaluate \
  --data-dir data/patent/cooperation_v2 \
  --embedding-dir artifacts/partner_vectors \
  --output "$OUTPUT" \
  --device cuda \
  --batch-size 16 \
  --bootstrap 1000 \
  --expected-mean-r100 0.16352341425902947 \
  --expected-max-r100 0.0768640544991725
