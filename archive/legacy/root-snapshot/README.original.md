# SirenCodec

Neural audio codec (PyTorch): **Vocos + RVQ/FSQ** + **SEANet** research.

## Quick start

```bash
pip install torch torchaudio vocos   # optional: pesq, soundfile
python3 run.py train_vocos_vq --help
python3 run.py train_pipeline --help
./scripts/run_training.sh         # SEANet arch-a-v2b + LibriSpeech
```

## Layout

Target: **≤4 regular files** per source folder (`src/sirencodec`, `core`, `docs`, `scripts`, `tools`). Repo root: `.gitattributes` / `.gitignore` + LFS (5 files at root).

| Path | Contents |
|------|----------|
| `run.py` | Single CLI: `train`, `train_pipeline`, `train_vocos_vq`, `sidecars`, `bench_*`, `watch`, … |
| `src/sirencodec/` | `data_pipeline`, `extras`, `sidecars`, `core/` (4 items) |
| `src/sirencodec/core/` | `train`, `train_pipeline`, `train_vocos_vq` (+ `__init__.py`) |
| `scripts/` | `run_training.sh`, `run_thesis_sweep.sh`, `export_codec_wav.py`, `compare_nl_vs_pca.py` |
| `tools/` | `watch`, `bench_fps`, `bench_lowfps`, `precompute_mels` |
| `tests/` | Optional smoke scripts |
| `docs/` | `GUIDE.md`, `RESEARCH.md`, `ROADMAP_2026.md`, `CONVENTIONS.txt` |

Optional: `pip install -e .`

## Docs

`docs/GUIDE.md` (overview + thesis commands), `docs/RESEARCH.md`.

## Requirements

Python **≥3.10** (`train_vocos_vq` uses modern typing).
