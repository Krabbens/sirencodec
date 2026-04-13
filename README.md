# SirenCodec

Neural audio codec using pretrained Vocos vocoder + RVQ/FSQ compression.

## Architecture

```
Audio (24kHz) → MelSpectrogram (100-dim) → RVQ/FSQ → Vocos → Audio (24kHz)
```

- **Feature extractor**: 100-dim mel spectrogram at 24kHz
- **Quantization**: RVQ (residual vector quantization) or FSQ (finite scalar quantization)
- **Vocoder**: Pretrained Vocos (`charactr/vocos-mel-24khz`, 13.5M params, fine-tuned)

## Bitrate Options

| Config | Command | Bitrate |
|--------|---------|---------|
| 4 codebooks × 1024 @ 94fps | `--rvq --n-codebooks 4` | 3,750 bps |
| 2 codebooks × 1024 @ 94fps | `--rvq --n-codebooks 2` | 1,875 bps |
| **2 codebooks × 1024 @ 30fps** | `--rvq --n-codebooks 2 --mel-fps 30` | **600 bps** |

## Usage

```bash
pip install vocos torch torchaudio

# Train at 600 bps (2 codebooks, 30fps)
python3 train_vocos_vq.py --steps 50000 --rvq --n-codebooks 2 --mel-fps 30

# Train at 1,875 bps (2 codebooks, 94fps)
python3 train_vocos_vq.py --steps 50000 --rvq --n-codebooks 2

# Train with FSQ (experimental)
python3 train_vocos_vq.py --steps 50000 --fsq-dims 16 --fsq-levels 5
```

## Dataset

Expects `data/master_manifest.jsonl` with entries:
```json
{"path": "path/to/audio.flac"}
```

## Results

| Step | Bitrate | mel loss | SI-SDR | PESQ |
|------|---------|----------|--------|------|
| 10k | 3,750 bps (4×1024@94fps) | 0.34 | -28.5 dB | 2.15 |
