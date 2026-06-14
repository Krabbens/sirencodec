#!/usr/bin/env bash
# Sequential thesis sweep: runs after current A2 finishes.
# Each run ~1-2.5h on RTX 3090 (smaller codebooks are faster).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEPS="${STEPS:-50000}"
BS="${BS:-8}"

run_exp() {
  local ncb="$1" cb="$2" tag="$3"
  local log="train_run_sweep_94fps_${tag}.log"
  local tsv="log_sweep_94fps_${tag}.tsv"
  echo "=== $tag: ${ncb}x${cb} @ 94fps ==="
  python3 train_vocos_vq.py --steps "$STEPS" --rvq \
    --n-codebooks "$ncb" --codebook-size "$cb" \
    --mel-fps 94 --bottleneck-dim 0 --batch-size "$BS" --data-dir data \
    --commit-weight 0.25 --log-tsv "$tsv" 2>&1 | tee "$log"
  echo "--- $tag done ---"
}

# A1: 1x32 = 470bps
run_exp 1 32 "1x32"

# A3: 1x128 = 658bps
run_exp 1 128 "1x128"

# A4: 1x256 = 752bps
run_exp 1 256 "1x256"

# A5: 1x1024 = 940bps
run_exp 1 1024 "1x1024"

# A6: 2x1024 = 1875bps
run_exp 2 1024 "2x1024"

# A7: 4x1024 = 3750bps
run_exp 4 1024 "4x1024"

echo "=== ALL SWEEP RUNS DONE ==="

# Mel refiner on best checkpoint
echo "=== REFINER: training on codec_final.pt ==="
python3 train_mel_refiner.py \
  --codec-checkpoint checkpoints_vocos_vq/codec_final.pt \
  --steps 20000 --batch-size 16 --data-dir data \
  --out-dir checkpoints_mel_refiner 2>&1 | tee train_run_mel_refiner.log
echo "--- refiner done ---"

# Student vocoder distillation
echo "=== DISTILLATION: Vocos -> StudentVocoder ==="
python3 train_vocos_distill.py --steps 50000 --batch-size 8 --data-dir data \
  --base 384 --out-dir checkpoints_student_vocoder 2>&1 | tee train_run_distill.log
echo "--- distillation done ---"

echo "=== FULL PIPELINE COMPLETE ==="
