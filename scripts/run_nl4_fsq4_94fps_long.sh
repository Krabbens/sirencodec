#!/usr/bin/env bash
# Long NL+FSQ-4 @ 94fps (STFT + mel_q). Default OUT matches experiments/nl4_fsq4_94fps_long.
# Override: OUT STEPS LAMBDA_STFT LAMBDA_MEL_Q RESUME INITIAL_CKPT
set -euo pipefail
cd "$(dirname "$0")/.."

export OUT="${OUT:-experiments/nl4_fsq4_94fps_long}"
export STEPS="${STEPS:-300000}"
export LAMBDA_STFT="${LAMBDA_STFT:-8}"
export LAMBDA_MEL_Q="${LAMBDA_MEL_Q:-15}"

exec bash scripts/run_nl_fsq4_continue_long.sh
