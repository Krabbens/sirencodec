#!/bin/bash
# CODEC-RESEARCHER Training Launcher
# Usage:
#   ./run_training.sh           # Default: 100k steps, resume if possible
#   ./run_training.sh 200000    # 200k steps
#   ./run_training.sh --no-resume  # Start fresh
#   ./run_training.sh 50000 no-eval  # 50k steps, skip eval

set -e

# Default config
STEPS=${1:-100000}
RESUME="--resume"
EXTRA_ARGS="${@:2}"

# Check for --no-resume or no-resume
if [[ "$*" == *"--no-resume"* ]] || [[ "$*" == *"no-resume"* ]]; then
    RESUME=""
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           CODEC-RESEARCHER Training Launcher             ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Steps:    $STEPS"
echo "║  Resume:   ${RESUME:+Yes}${RESUME:-No}"
echo "║  Arch:     arch-a-v2b (500bps)"
echo "║  Data:     Common Voice PL (data/cv-corpus)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
    echo "GPU: $GPU ($VRAM MiB)"
else
    echo "WARNING: No GPU detected - training will use CPU (very slow!)"
fi

# Check data
if [ ! -d "data/cv-corpus/pl/clips" ]; then
    echo "ERROR: Training data not found at data/cv-corpus/pl/clips"
    exit 1
fi

MP3_COUNT=$(find data/cv-corpus/pl/clips -name "*.mp3" | wc -l)
echo "Data: $MP3_COUNT MP3 files found"
echo ""

# Clean old log for fresh run
if [ -z "$RESUME" ]; then
    echo "Starting fresh training (removing old log.tsv)"
    rm -f log.tsv
fi

# Launch training
echo "Starting training..."
echo "Monitor in another terminal with: python watch.py --live"
echo ""

python3 train_pipeline.py \
    --steps $STEPS \
    --arch arch-a-v2b \
    --batch-size 8 \
    --real-data \
    --data-dir data/cv-corpus \
    --eval-every 5000 \
    --save-every 5000 \
    --log-every 200 \
    --codebook-size 1024 \
    --psych-mask \
    $RESUME \
    $EXTRA_ARGS

echo ""
echo "Training finished. Results:"
python3 watch.py --summary
