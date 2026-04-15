#!/usr/bin/env bash
# moj_nowy_etap on 8 GPUs: DDP via torchrun, global batch 80 (10 per GPU), LR scaled vs batch-8 baseline.
# Resume logic matches scripts/run_nl_fsq4_continue_long.sh.
# Env: OUT STEPS LAMBDA_STFT LAMBDA_MEL_Q RESUME INITIAL_CKPT NPROC (default 8)
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

NPROC="${NPROC:-8}"
OUT="${OUT:-experiments/moj_nowy_etap}"
INITIAL_CKPT="${INITIAL_CKPT:-}"
STEPS="${STEPS:-300000}"
LAMBDA_STFT="${LAMBDA_STFT:-8}"
LAMBDA_MEL_Q="${LAMBDA_MEL_Q:-15}"
# Linear LR scale: previous single-GPU scripts used batch 8; global batch 80 => ×10 on base lrs (1e-4 / 2.5e-5).
LR_GEN="${LR_GEN:-1e-3}"
LR_DISC="${LR_DISC:-2.5e-4}"

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

echo "[ddp8] nproc=$NPROC global_batch=$((10 * NPROC)) per_gpu_batch=10 lr_gen=$LR_GEN lr_disc=$LR_DISC"
echo "[resume] start_step=$START_STEP ckpt=$RESUME_CKPT target_steps=$STEPS | state=$OUT/resume_state.json"

exec torchrun --standalone --nproc_per_node="$NPROC" train_vocos_vq.py \
  --nl-dim 4 --nl-fsq-levels 4 \
  --mel-fps 94 \
  --warmup-steps 15000 \
  --lr-warmup-steps 15000 \
  --lr-start-factor 0.0 \
  --lr-min-ratio 0.05 \
  --segment-ramp-steps 12000 \
  --segment-length-min 6000 \
  --steps "$STEPS" \
  --batch-size 10 \
  --lr-gen "$LR_GEN" \
  --lr-disc "$LR_DISC" \
  --lambda-stft "$LAMBDA_STFT" \
  --lambda-mel-q "$LAMBDA_MEL_Q" \
  --exp-dir "$OUT" \
  --resume-ckpt "$RESUME_CKPT"
