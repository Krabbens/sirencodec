#!/usr/bin/env bash
# Continue experiments/moj_nowy_etap (NL+FSQ-4, 94fps) with STFT + mel_q losses.
# Override: OUT STEPS LAMBDA_STFT LAMBDA_MEL_Q RESUME INITIAL_CKPT
set -euo pipefail
cd "$(dirname "$0")/.."

export OUT="${OUT:-experiments/moj_nowy_etap}"
export STEPS="${STEPS:-300000}"
export LAMBDA_STFT="${LAMBDA_STFT:-8}"
export LAMBDA_MEL_Q="${LAMBDA_MEL_Q:-15}"

exec bash scripts/run_nl_fsq4_continue_long.sh
