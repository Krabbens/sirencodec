#!/usr/bin/env bash
# Track A: 94fps direct ResidualVQ bitrate sweep + upper-bound runs (2x/4x1024 @ 94fps).
# Usage: STEPS=50000 BS=8 ./scripts/run_thesis_sweep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STEPS="${STEPS:-50000}"
BS="${BS:-8}"
DATA="${DATA:-data}"

echo "=== Thesis sweep: STEPS=$STEPS BS=$BS ==="

for CB in 32 64 128 256 1024; do
  LOG="log_sweep_94fps_1x${CB}.tsv"
  echo "--- 1x${CB} @ 94fps ---"
  python3 run.py train_vocos_vq --steps "$STEPS" --rvq --n-codebooks 1 --codebook-size "$CB" \
    --mel-fps 94 --bottleneck-dim 0 --batch-size "$BS" --data-dir "$DATA" \
    --log-tsv "$LOG" 2>&1 | tee "train_run_sweep_94fps_1x${CB}.log"
done

echo "--- 2x1024 @ 94fps (1875 bps) ---"
python3 run.py train_vocos_vq --steps "$STEPS" --rvq --n-codebooks 2 --codebook-size 1024 \
  --mel-fps 94 --bottleneck-dim 0 --batch-size "$BS" --data-dir "$DATA" \
  --log-tsv log_sweep_94fps_2x1024.tsv 2>&1 | tee train_run_sweep_94fps_2x1024.log

echo "--- 4x1024 @ 94fps (3750 bps) ---"
python3 run.py train_vocos_vq --steps "$STEPS" --rvq --n-codebooks 4 --codebook-size 1024 \
  --mel-fps 94 --bottleneck-dim 0 --batch-size "$BS" --data-dir "$DATA" \
  --log-tsv log_sweep_94fps_4x1024.tsv 2>&1 | tee train_run_sweep_94fps_4x1024.log

echo "Done."
