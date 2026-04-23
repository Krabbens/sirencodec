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
uv run train --epochs 5 --fast --batch 256
```

Resume:

```bash
uv run train --continue mlx_runs/<run_dir>
```

## Run artifacts

Each run creates:

```text
mlx_runs/YYYYMMDD_HHMMSS/
  checkpoints/
  inference/
  log_mlx.tsv
  results.tsv
  run_state.json
  train_config.json
```

- numbered checkpoints: every 10 epochs by default
- `checkpoints/latest.pt`: updated every epoch
- inference exports: one subdirectory per step under `inference/XXXXXXXX/`

## Docker

Build:

```bash
docker build -t sirencodec .
```

Run with NVIDIA runtime:

```bash
docker run --rm --gpus all -it -v $(pwd)/data:/workspace/data sirencodec \
  uv run train --epochs 5 --fast --batch 256
```

The container requires NVIDIA Container Toolkit on the host.
