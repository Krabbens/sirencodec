#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== NL-deep: k=4, L=4(fsq), 4 hidden layers (h=64) ==="
python3 train_vocos_vq.py \
  --nl-dim 4 --nl-fsq-levels 4 --nl-layers 4 \
  --mel-fps 94 --warmup-steps 15000 --steps 50000 \
  --batch-size 8 \
  --exp-dir experiments/nl4_fsq4_L4_94fps_warm15k_50k

echo "=== Done ==="
echo "Logs: experiments/nl4_fsq4_L4_94fps_warm15k_50k/log.tsv"
