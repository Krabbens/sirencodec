# SirenCodec

CUDA-first neural audio codec training repo with an optional MLX backend kept for Apple Silicon experiments.

## Layout

- `src/sirencodec/cuda/` - active CUDA trainer, model, data pipeline and losses
- `src/sirencodec/mlx/` - optional MLX backend for Apple Silicon training
- `src/sirencodec/` - shared config and utilities
- `uv run train` - canonical training entrypoint
- `uv run train-mlx` or `uv run python tools/train_mlx.py` - MLX training entrypoints

## Local setup

```bash
uv sync
# Apple Silicon MLX:
uv sync --extra mlx
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

MLX smoke test on Apple Silicon:

```bash
uv run train-mlx --steps 1 --no-librispeech --batch 1 --segment 512 --fast
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

# Download and extract LibriSpeech corpora.
download-train-clean-100
download-train-clean-360
```

The container requires NVIDIA Container Toolkit on the host.
