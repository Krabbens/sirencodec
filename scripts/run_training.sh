#!/bin/bash
# SEANet pipeline launcher (arch-a-v2b). From repo root via scripts/.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STEPS=${1:-100000}
RESUME="--resume"
EXTRA_ARGS="${@:2}"

if [[ "$*" == *"--no-resume"* ]] || [[ "$*" == *"no-resume"* ]]; then
    RESUME=""
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           SirenCodec — train_pipeline (SEANet)           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Steps:    $STEPS"
echo "║  Resume:   ${RESUME:+Yes}${RESUME:-No}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
    echo "GPU: $GPU ($VRAM MiB)"
elif python3 -c "import torch; import sys; sys.exit(0 if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "GPU: Apple MPS"
else
    echo "WARNING: No CUDA/MPS — CPU training is very slow."
fi

if [ ! -d "data/librispeech/LibriSpeech" ]; then
    echo "ERROR: Data not found. Run: python3 run.py train download data"
    exit 1
fi

if [ -z "$RESUME" ]; then
    echo "Starting fresh (removing old log.tsv)"
    rm -f log.tsv
fi

echo "Monitor: python3 run.py watch --live"
echo ""

# Stage 1 = encoder→decoder mel AE. Discriminator is optional here; MRSTFT disc is the main cost on MPS.
# --no-adv-stage1: mel-only Stage 1 (fast). VQ+adversarial still run in Stage 2.
# Stage 2: --adv-every 8 thins discriminator updates vs every step.
python3 run.py train_pipeline \
    --steps "$STEPS" \
    --arch arch-a-v2b \
    --batch-size 8 \
    --real-data \
    --data-dir data \
    --eval-every 5000 \
    --save-every 5000 \
    --log-every 200 \
    --codebook-size 1024 \
    --psych-mask \
    --no-adv-stage1 \
    --adv-every 8 \
    $RESUME \
    $EXTRA_ARGS

echo ""
python3 run.py watch --summary
