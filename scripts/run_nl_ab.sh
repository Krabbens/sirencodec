#!/usr/bin/env bash
# NL-A / NL-B from Nonlinear Bottleneck + FSQ plan (596 bps / 752 bps @ 94fps)
set -euo pipefail
cd "$(dirname "$0")/.."

# NL-A: k=4, L=3 → 6.34 bit/frame × 94 ≈ 596 bps
python3 train_vocos_vq.py \
  --nl-dim 4 --nl-fsq-levels 3 \
  --mel-fps 94 --warmup-steps 15000 --steps 50000 \
  --batch-size 8 \
  --exp-dir ""

# NL-B: k=4, L=4 → 8 bit/frame × 94 = 752 bps
python3 train_vocos_vq.py \
  --nl-dim 4 --nl-fsq-levels 4 \
  --mel-fps 94 --warmup-steps 15000 --steps 50000 \
  --batch-size 8 \
  --exp-dir ""

echo "Done. Logs: experiments/nl4_fsq3_94fps/log.tsv and experiments/nl4_fsq4_94fps/log.tsv"
