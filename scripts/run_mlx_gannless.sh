#!/usr/bin/env bash
# GAN-free MLX codec — train from scratch with a pure spectral recipe + SnakeBeta.
#
# v2 tuning (after diagnosing pulse-excitation failure mode at step ~42k):
#   - SC + complex STFT weights bumped ~3–5× to beat biased L1 mean-matching
#   - HF emphasis 1 → 1.5  (push formant bands F2/F3 harder)
#   - time-domain L1  1.0 → 0.5  (reduce mean-matching pressure)
#   - SnakeBeta now also inside LatentTemporalStack (periodic inductive bias everywhere)
#
# Spectral losses (discriminator-free):
#   - log-mag STFT L1 (--lambda-stft)
#   - Spectral Convergence (--lambda-sc)  ‖|S(ŷ)|-|S(y)|‖_F / ‖|S(y)|‖_F  (peak-emphasizing)
#   - Complex STFT L1 (--lambda-complex-stft)  0.5·(L1(Re)+L1(Im))  (phase)
#   - log-mag freq/time gradients (--lambda-stft-grad) + spectral cosine (--lambda-stft-cos)
#   - mel-bin log-L1 (--lambda-mel-l1), linear-mag L1 (--lambda-mag-l1)
#
# Multi-scale STFT: LibriSpeech ×3 (1024/256, 2048/512, 4096/1024).
# Checkpoints in mlx_checkpoints_snake/ by default (override with CKPT_DIR=...).

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
EPOCHS="${EPOCHS:-2}"
LOG="${LOG:-${REPO_ROOT}/mlx_snake.log}"
SPEC_DIR="${SPEC_DIR:-${REPO_ROOT}/mlx_spectrograms_snake}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/mlx_checkpoints_snake}"
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-12}"

exec uv run python tools/train_mlx.py \
  --epochs "${EPOCHS}" \
  --librispeech \
  --batch "${BATCH}" \
  --grad-accum-steps "${GRAD_ACCUM}" \
  --load-audio-threads 12 \
  --prefetch-audio \
  --segment 16384 \
  --latent-dim 512 \
  --latent-temporal-depth 2 \
  --latent-temporal-post-depth 2 \
  --activation snake_beta \
  --lr 3e-4 \
  --lr-schedule cosine \
  --lr-min-ratio 0.1 \
  --lr-warmup-steps 2000 \
  --grad-clip 5 \
  --stft-scales "1024,256;2048,512;4096,1024" \
  --stft-scale-weights "1,1.75,2.5" \
  --lambda-time 0.5 \
  --lambda-stft 0.4 \
  --lambda-sc 2.5 \
  --lambda-complex-stft 0.5 \
  --lambda-stft-grad 0.1 \
  --stft-grad-freq-weight 3.0 \
  --stft-grad-time-weight 0.5 \
  --lambda-stft-cos 0.05 \
  --stft-ramp-steps 6000 \
  --stft-ramp-start 0.25 \
  --stft-hf-emphasis 1.5 \
  --lambda-mag-l1 0.15 \
  --lambda-mel-l1 0.06 \
  --lambda-mel-l2 0 \
  --lambda-vq 5 \
  --lambda-marginal 0.35 \
  --marginal-tau 0.04 \
  --marginal-boost-steps 24000 \
  --marginal-boost-mult 2.5 \
  --vq-reset-every 1000 \
  --vq-reset-collapse-frac 0.42 \
  --vq-reset-noise 0.12 \
  --vq-reset-log-every 5000 \
  --lambda-cos 0.15 \
  --log-every 50 \
  --spectrogram-every 2500 \
  --spectrogram-dir "${SPEC_DIR}" \
  --spectrogram-seconds 8 \
  --checkpoint-every 10000 \
  --checkpoint-dir "${CKPT_DIR}" \
  2>&1 | tee "${LOG}"
