#!/usr/bin/env bash
# New NL+FSQ-4 run — no resume. Set OUT=experiments/my_run_name
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-experiments/nl4_fsq4_94fps_fresh_$(date +%Y%m%d_%H%M)}"
STEPS="${STEPS:-100000}"
LAMBDA_STFT="${LAMBDA_STFT:-8}"
LAMBDA_MEL_Q="${LAMBDA_MEL_Q:-15}"

mkdir -p "$OUT"

python3 train_vocos_vq.py \
  --nl-dim 4 --nl-fsq-levels 4 \
  --mel-fps 94 \
  --warmup-steps 15000 \
  --lr-warmup-steps 15000 \
  --lr-start-factor 0.0 \
  --lr-min-ratio 0.05 \
  --segment-ramp-steps 12000 \
  --segment-length-min 6000 \
  --steps "$STEPS" \
  --batch-size 8 \
  --compile \
  --lambda-stft "$LAMBDA_STFT" \
  --lambda-mel-q "$LAMBDA_MEL_Q" \
  --exp-dir "$OUT"

echo "fresh done → $OUT/resume_state.json after first ckpt"
