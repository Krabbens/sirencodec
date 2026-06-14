---
branch: 7m-best
status: canonical
base: master
used_in_thesis: yes
install: uv sync --extra dev
test: uv run pytest -q
---

# Branch manifest

## Role

This branch is the preserved reference for the codec variant selected for the
master's thesis. It is also the structural source for the canonical `master`
branch. New cleanup or documentation work may be applied to both branches,
but the name `7m-best` remains a stable reference used by experiment notes.

## Model variant

The default configuration processes 16 kHz mono speech with eight stride-2
encoder stages, resulting in a temporal stride of 256. The continuous latent
dimension is 512. The nominal stream uses three residual vector quantizers
with 32 entries each, which gives approximately 0.94 kb/s before container or
transport overhead.

The active training path is implemented in `src/sirencodec/cuda/`. It includes
waveform and multi-resolution spectral reconstruction terms, RVQ stability
metrics and optional semantic supervision. The MLX implementation is retained
for Apple Silicon experiments and for the conditioned two-band prototype.
Optional self-attention and decoder refinements are disabled by default so
that existing checkpoints keep their original architecture.

## Requirements

- Python 3.10 or newer;
- NVIDIA GPU and a compatible CUDA runtime for full training;
- Apple Silicon and the `mlx` extra only for MLX experiments;
- LibriSpeech prepared outside Git for corpus training;
- CMake 3.24 or newer for the optional C++ runtime.

## Validated commands

Portable repository checks:

```bash
python scripts/validate_branch_layout.py
python -m compileall -q src tools tests scripts
pytest -q
```

CUDA environment and synthetic training smoke test:

```bash
uv sync --extra dev
uv run train --epochs 1 --no-librispeech --fast
```

Corpus training with an explicit configuration:

```bash
uv run train --config configs/abd.json --dataset train-clean-100 --epochs 5
```

CUDA inference:

```bash
uv run python tools/infer_cuda.py --help
```

MLX training and inference:

```bash
uv sync --extra mlx --extra dev
uv run python tools/train_mlx.py --help
uv run python tools/infer_mlx.py --help
```

C++ runtime:

```bash
cmake -S cpp/sirencodec_infer -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel
build/cpp/sirencodec_tests --help
```

## External assets

Datasets, checkpoints, reconstructed audio, LiteRT exports and benchmark
outputs are not versioned. Commands that require them must receive explicit
paths. The thesis sources and tables record the checkpoint and dataset used
for reported results.

## Known limitations

Full CUDA training and checkpoint inference are not exercised in portable CI.
The C++ integration command requires exported LiteRT models and an audio input,
so CI verifies compilation and the command-line interface instead of running
model inference.
