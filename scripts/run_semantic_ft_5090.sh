#!/usr/bin/env bash
# Full 30-epoch semantic fine-tune on RTX 5090 (or any CUDA host with LibriSpeech under ./data).
# Requires the trunk checkpoint from a completed sub1k_5090_stable_200 run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec uv run train \
  --config configs/sub1k_semantic_ft_30.json \
  --init-from "${INIT_FROM:-/workspace/experiments/20260427_073203/checkpoints/codec_step162599.pt}"
