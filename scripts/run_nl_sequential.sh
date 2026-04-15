#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Waiting for NL-A (PID $1) to finish ==="
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== NL-A finished ==="

echo "=== Starting NL-B: k=4, L=4 → 752 bps ==="
python3 train_vocos_vq.py \
  --nl-dim 4 --nl-fsq-levels 4 \
  --mel-fps 94 --warmup-steps 15000 --steps 50000 \
  --batch-size 8 \
  --exp-dir experiments/nl4_fsq4_94fps_warm15k_50k

echo "=== NL-B complete ==="
echo "Logs: experiments/nl4_fsq4_94fps_warm15k_50k/log.tsv"
