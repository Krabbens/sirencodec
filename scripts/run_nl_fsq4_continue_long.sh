#!/usr/bin/env bash
# NL+FSQ-4 long run: compile, cosine LR, segment ramp. Auto-resume from OUT/resume_state.json or newest ckpt.
set -euo pipefail
cd "$(dirname "$0")/.."

INITIAL_CKPT="${INITIAL_CKPT:-experiments/nl4_fsq4_94fps_warm15k_50k/checkpoints/codec_step45000.pt}"
OUT="${OUT:-experiments/nl4_fsq4_94fps_warm15k_100k_compile}"
STEPS="${STEPS:-100000}"
# Spectral detail: multi-res STFT + quantized-mel L1 (override with env)
LAMBDA_STFT="${LAMBDA_STFT:-8}"
LAMBDA_MEL_Q="${LAMBDA_MEL_Q:-15}"

mkdir -p "$OUT"

mapfile -t _pick < <(OUT="$OUT" INITIAL_CKPT="$INITIAL_CKPT" EXPLICIT="${RESUME:-}" python3 << 'PY'
import json, os, sys
from pathlib import Path

import torch

out = Path(os.environ["OUT"])
initial = os.environ.get("INITIAL_CKPT", "")
explicit = os.environ.get("EXPLICIT", "").strip()

def out_lines(ckpt: Path, step: int) -> None:
    ckpt = ckpt.resolve()
    print(str(ckpt))
    print(int(step))

if explicit:
    p = Path(explicit)
    if not p.is_file():
        print("RESUME is not a file:", explicit, file=sys.stderr)
        sys.exit(1)
    c = torch.load(p, map_location="cpu", weights_only=False)
    out_lines(p, int(c.get("step", 0)))
    sys.exit(0)

state_path = out / "resume_state.json"
if state_path.is_file():
    d = json.loads(state_path.read_text())
    p = Path(d["checkpoint"])
    if not p.is_file():
        p = out / "checkpoints" / p.name
    if p.is_file():
        step = d.get("step")
        if step is None:
            step = torch.load(p, map_location="cpu", weights_only=False).get("step", 0)
        out_lines(p, int(step))
        sys.exit(0)

ckdir = out / "checkpoints"
best_p, best_s = None, -1
if ckdir.is_dir():
    for f in sorted(ckdir.glob("*.pt")):
        try:
            c = torch.load(f, map_location="cpu", weights_only=False)
            s = int(c.get("step", -1))
            if s > best_s:
                best_s, best_p = s, f
        except Exception:
            pass
if best_p is not None:
    out_lines(best_p, best_s)
    sys.exit(0)

if initial:
    p = Path(initial)
    if p.is_file():
        c = torch.load(p, map_location="cpu", weights_only=False)
        out_lines(p, int(c.get("step", 0)))
        sys.exit(0)

print("No checkpoint (set RESUME= or INITIAL_CKPT)", file=sys.stderr)
sys.exit(1)
PY
)
RESUME_CKPT="${_pick[0]}"
START_STEP="${_pick[1]}"

echo "[resume] start_step=$START_STEP ckpt=$RESUME_CKPT target_steps=$STEPS | state=$OUT/resume_state.json"
echo "[resume] lambda_stft=$LAMBDA_STFT lambda_mel_q=$LAMBDA_MEL_Q (LAMBDA_STFT=0 LAMBDA_MEL_Q=0 to disable)"

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
  --exp-dir "$OUT" \
  --resume-ckpt "$RESUME_CKPT"

echo "Done. log=$OUT/log.tsv state=$OUT/resume_state.json"
