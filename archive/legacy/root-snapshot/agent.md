# SirenCodec — Neural Audio Codec Project

## Project Overview

SirenCodec is a research project for building a **neural audio codec** — a system that compresses raw audio waveforms into compact discrete representations and reconstructs them. The goal is to achieve high audio quality at ultra-low bitrates (target: <1kbps for speech).

The project lives in `/home/sperm/autoresearch/sirencodec` and uses **Python/PyTorch** on an NVIDIA RTX 3090 (24GB VRAM).

## Architecture

### Current Working Approach: Vocos VQ Codec

```
Audio (24kHz) → MelSpectrogram (100-dim, variable fps) → RVQ/FSQ → Upsample → Vocos Vocoder → Audio (24kHz)
```

- **Feature extractor**: 100-dim log-mel spectrogram at 24kHz (torchaudio)
- **Quantization**: RVQ (residual vector quantization, EMA codebooks) or FSQ (finite scalar quantization)
- **Vocoder**: Pretrained Vocos (`charactr/vocos-mel-24khz`, 13.5M params, fine-tuned)
- **Discriminator**: Multi-resolution STFT discriminator (3 resolutions) for adversarial training
- **Dataset**: 28,539 English speech files from LibriSpeech via `data/master_manifest.jsonl`

### Key Design Decision

The mel frame rate is decoupled from the Vocos input frame rate. Audio is analyzed at a **low frame rate** (e.g., 30fps → hop=800), VQ-quantized, then **upsampled back to 94fps** (hop=256) via bilinear interpolation before feeding Vocos. This achieves massive bitrate reduction while keeping the vocoder happy.

### Bitrate Options

| Config | Command | Bitrate | PESQ (step 10k) |
|--------|---------|---------|-----------------|
| 4 codebooks × 1024 @ 94fps | `--rvq --n-codebooks 4` | 3,750 bps | 2.15 |
| 2 codebooks × 1024 @ 94fps | `--rvq --n-codebooks 2` | 1,875 bps | — |
| **2 codebooks × 1024 @ 30fps** | `--rvq --n-codebooks 2 --mel-fps 30` | **600 bps** | (in progress) |

### Failed Architectures (for reference)

All SEANet encoder-based approaches with various decoders produced PESQ ~1.0 (noise):
- Mamba decoder (iSTFT mag/phase, waveform head, bottleneck, iSTFTNet-style)
- Zipformer decoder (LuxTTS-inspired)
- HiFi-GAN decoder
- Vocos decoder (without pretrained weights)

**Root cause**: SEANet's 320× downsampling destroys too much fine-grained speech detail. The pretrained mel spectrogram + Vocos approach bypasses this entirely.

## Key Files

| File | Purpose |
|------|---------|
| `train_vocos_vq.py` | **Main training script** — Vocos VQ codec with RVQ/FSQ |
| `train.py` | Original SEANet-based codec (all architecture variants, decoders) |
| `train_pipeline.py` | Multi-stage training pipeline for SEANet architectures |
| `data_pipeline.py` | Dataset download, manifest building, data loading |
| `RESEARCH.md` | Research log with codec benchmarks and technique catalog |
| `ROADMAP_2026.md` | Development roadmap and architecture decisions |
| `SYSTEM.md` | System design documentation |

## Dependencies

```
torch, torchaudio, vocos, pesq (optional for evaluation)
```

Install: `pip install vocos torch torchaudio pesq`

## Usage

### Train Vocos VQ Codec (current experiment)

```bash
# 600 bps: 2 codebooks × 1024 @ 30fps (current)
python3 train_vocos_vq.py --steps 50000 --rvq --n-codebooks 2 --mel-fps 30

# 1,875 bps: 2 codebooks × 1024 @ 94fps
python3 train_vocos_vq.py --steps 50000 --rvq --n-codebooks 2

# 3,750 bps: 4 codebooks × 1024 @ 94fps
python3 train_vocos_vq.py --steps 50000 --rvq --n-codebooks 4

# FSQ (experimental — projection unstable)
python3 train_vocos_vq.py --steps 50000 --fsq-dims 16 --fsq-levels 5
```

### Data Setup

Data must be in `data/master_manifest.jsonl` format:
```json
{"path": "data/librispeech/.../audio.flac"}
```

The `data_pipeline.py` script can download and build manifests for LibriSpeech, CommonVoice, VCTK.

### Checkpoints

Saved to `checkpoints_vocos_vq/codec_step*.pt` during training.

### Logs

- `log_vocos_vq.tsv` — training metrics (step, mel loss, adv loss, commit loss, VQ utilization, grad norm, LR)
- `train_run_rvq_600bps.log` — training stdout for current experiment
- `train_run_*.log` — historical training logs for all architecture variants

## Project Structure

```
/home/sperm/autoresearch/sirencodec/
├── train_vocos_vq.py          # Current experiment (Vocos VQ codec)
├── train.py                   # SEANet-based codec (6 architecture variants)
├── train_pipeline.py          # Multi-stage training for SEANet
├── data_pipeline.py           # Dataset management
├── data/                      # Audio data + master_manifest.jsonl
├── checkpoints_vocos_vq/      # Vocos VQ checkpoints
├── checkpoints/               # SEANet checkpoints
├── log_vocos_vq.tsv           # Current training metrics
├── train_run_*.log            # Training logs
├── RESEARCH.md                # Research notes
├── ROADMAP_2026.md            # Development roadmap
├── LuxTTS/                    # Cloned LuxTTS repo (reference)
└── QWEN.md                    # This file
```

## GitHub Repo

- **URL**: https://github.com/Krabbens/sirencodec
- **Contains**: `train_vocos_vq.py`, `README.md`, `requirements.txt`, `.gitignore`
- Only the current Vocos VQ experiment is pushed to GitHub (not SEANet files or logs).

## Training Metrics History

| Date | Config | Step | mel loss | SI-SDR | PESQ | VQ% | Speed |
|------|--------|------|----------|--------|------|-----|-------|
| 2025 | Vocos+RVQ 4×1024@94fps | 10k | 0.34 | -28.5 dB | 2.15 | 46% | 335ms |
| 2025 | Vocos+RVQ 2×1024@30fps (600bps) | — | — | — | — | — | — |
| — | SEANet+Mamba (all variants) | any | — | -32 to -44 dB | 1.03 | — | — |
| — | SEANet+Zipformer | any | — | -41.8 dB | 1.03 | — | — |

## Development Conventions

- Single-file scripts (no modular package structure)
- Training runs logged to both stdout and `.log` files
- Checkpoints saved every 5000 steps
- PESQ evaluation every 5000 steps on 5 validation samples
- SI-SDR and PESQ computed at 16kHz (resampled from 24kHz)
- Audio normalized to [-1, 1] range, 24kHz, 1-second segments (24,000 samples)
- LR: 1e-4 generator, 2.5e-5 discriminator, cosine annealing after 5000 step warmup
