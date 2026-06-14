# SirenCodec

SirenCodec is a research repository for a low-bitrate neural speech codec
developed as part of a master's thesis. The canonical implementation uses a
CUDA-first PyTorch training path and keeps an MLX backend for selected Apple
Silicon experiments.

This repository preserves multiple experimental branches. Their model
assumptions and validated commands are documented in
[`docs/branch.md`](docs/branch.md). Read that manifest before comparing
results or resuming an older experiment.

## Repository layout

- `src/sirencodec/` - package code shared by training and inference;
- `configs/` - reproducible experiment configurations;
- `tools/` - inference, conversion, export and benchmark commands;
- `scripts/` - environment and automation helpers;
- `tests/` - checks that do not require private checkpoints or datasets;
- `cpp/` - optional C++ inference runtime;
- `overleaf/` - thesis sources on branches that maintain the document;
- `archive/legacy/` - preserved historical material that is not an active
  entrypoint.

## Setup

Install the default CUDA environment:

```bash
uv sync --extra dev
```

The canonical training entrypoint is:

```bash
uv run train --help
```

Run repository checks:

```bash
uv run python scripts/validate_branch_layout.py
uv run pytest -q
```

Generated datasets, checkpoints, audio samples, exported models and run
directories are intentionally excluded from Git. The branch manifest records
the external assets required to reproduce a specific experiment.
