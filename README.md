# SirenCodec

CUDA-first neural audio codec training repo with an optional MLX backend kept for Apple Silicon experiments.

## Layout

- `src/sirencodec/cuda/` - active CUDA trainer, model, data pipeline and losses
- `src/sirencodec/mlx/` - optional MLX backend
- `src/sirencodec/` - shared config and utilities
- `uv run train` - canonical training entrypoint

## Local setup

```bash
uv sync
```

## Train

Synthetic smoke test:

```bash
uv run train --epochs 1 --no-librispeech --fast
```

Real corpus:

```bash
uv run train --dataset train-clean-100 --epochs 5 --fast --batch 256
```

Config template:

```bash
uv run train --config configs/abd.json --epochs 5
```

Resume:

```bash
uv run train --continue experiments/<run_dir>
```

## Run artifacts

Each run creates:

```text
experiments/YYYYMMDD_HHMMSS/
  checkpoints/
  inference/
  logs.txt
  logs.csv
  log_mlx.tsv
  results.tsv
  run_state.json
  train_config.json
```

- numbered checkpoints: every 10 epochs by default
- `checkpoints/latest.pt`: updated every epoch
- inference exports: one subdirectory per step under `inference/XXXXXXXX/`
- `logs.txt`: plain-text console log mirrored to file
- `logs.csv`: structured per-progress-step metrics for spreadsheets / analysis

See [`configs/`](./configs) for ready-made curriculum templates.

## Docker

Build:

```bash
docker build -t sirencodec:git .
```

Run with NVIDIA runtime:

```bash
docker run --rm --gpus all -it -v $(pwd)/data:/workspace/data sirencodec:git \
  uv run train --epochs 5 --fast --batch 256
```

The image contains the repository `.git` directory and two container helpers:

```bash
# Fetch, optionally switch branch, pull, then refresh uv dependencies.
sirencodec-sync models/13b

# Download and extract LibriSpeech train-clean-360 to /workspace/data/train-clean-360.
download-train-clean-360
```

The container requires NVIDIA Container Toolkit on the host.

## PVQ-NS MLX MVP

This branch also includes an ultra-low-bitrate speech-specific codec path:

- `sirencodec.pvq_ns` - 8 kHz LPC/LSF + F0 + energy analysis, PQ/scalar quantization, 500/700/900 bps bitstream, and LPC-only fallback synthesis
- `sirencodec.mlx.pvq_ns` - tiny residual MLX TCN post-filter for improving the LPC waveform from dequantized features
- `tools/pvq_ns_mlx.py` / `pvq-ns` - codebook training, encode/decode/roundtrip, and MLX post-filter smoke checks

Examples:

```bash
PYTHONPATH=src python3 tools/pvq_ns_mlx.py train-codebooks data/train-clean-100 pvq_900_codebooks.npz --mode 900
PYTHONPATH=src python3 tools/pvq_ns_mlx.py roundtrip sample.wav out/pvq900_lpc.wav --codebooks pvq_900_codebooks.npz
PYTHONPATH=src python3 tools/pvq_ns_mlx.py train-postfilter data/train-clean-100 pvq_900_codebooks.npz pvq_pf.safetensors --steps 1000
PYTHONPATH=src python3 tools/pvq_ns_mlx.py postfilter-smoke
```

The built-in fallback codebooks are deterministic and useful for tests only. Train LSF codebooks on speech before judging quality.
