# SirenCodec — Neural Audio Codec

## Overview

Neural codec: waveforms → discrete latents → reconstruct. Target **<1 kbps** speech.

Repo root. Stack: **Python / PyTorch**. Device: **CUDA → MPS → CPU** via `pick_device()` in `train_pipeline.py` / `train.py`. Past GPU: RTX 3090 24GB.

## Architecture — Vocos VQ (current line)

```
Audio (24kHz) → MelSpectrogram (100-dim, variable fps) → RVQ/FSQ → Upsample → Vocos Vocoder → Audio (24kHz)
```

- **Features**: 100-dim log-mel @ 24 kHz (`torchaudio`)
- **Quant**: RVQ (EMA codebooks) or FSQ
- **Vocoder**: `charactr/vocos-mel-24khz` (~13.5M), fine-tuned
- **Disc**: multi-res STFT (3 resolutions)
- **Data**: LibriSpeech via `data/master_manifest.jsonl` (~28k EN utterances)

**Design**: Mel fps **decoupled** from Vocos input fps — e.g. **30 fps** (hop=800), **upsample to 94 fps** (hop=256) before Vocos → bitrate drop, vocoder same.

### Bitrate (examples)

| Config | Command | Bitrate | PESQ (10k, ref) |
|--------|---------|---------|-----------------|
| 4×1024 @ 94fps | `--rvq --n-codebooks 4` | 3750 bps | 2.15 |
| 2×1024 @ 94fps | `--rvq --n-codebooks 2` | 1875 bps | — |
| **2×1024 @ 30fps** | `--rvq --n-codebooks 2 --mel-fps 30` | **600 bps** | (tune) |

### Dead ends (SEANet-heavy)

SEANet + decoders (Mamba / Zipformer / HiFi-GAN / Vocos scratch) → PESQ ~1.0 noise. **Cause**: 320× downsample kills detail. **Working path**: pretrained mel + **Vocos** (or HiFi-GAN in `train.py` / pipeline).

## Key files

| Location | Role |
|----------|------|
| `run.py` | CLI: `train`, `train_pipeline`, `train_vocos_vq`, `sidecars`, tools |
| `src/sirencodec/core/train_vocos_vq.py` | **Vocos VQ** trainer (RVQ/FSQ) |
| `src/sirencodec/core/train.py` | SEANet codec variants + decoders |
| `src/sirencodec/core/train_pipeline.py` | Multi-stage SEANet pipeline |
| `src/sirencodec/data_pipeline.py` | Download, manifests, loaders |
| `src/sirencodec/sidecars.py` | `mel_refiner` / `distill` |
| `docs/RESEARCH.md` | Log, benchmarks |
| `docs/ROADMAP_2026.md` | Roadmap |

## Deps

```
torch, torchaudio, vocos, pesq (optional)
```

Install: `pip install vocos torch torchaudio pesq`

## Usage — Vocos VQ

```bash
# 600 bps
python3 run.py train_vocos_vq --steps 50000 --rvq --n-codebooks 2 --mel-fps 30

# 1875 bps
python3 run.py train_vocos_vq --steps 50000 --rvq --n-codebooks 2

# 3750 bps
python3 run.py train_vocos_vq --steps 50000 --rvq --n-codebooks 4

# FSQ (experimental)
python3 run.py train_vocos_vq --steps 50000 --fsq-dims 16 --fsq-levels 5
```

### Data

Manifest lines:

```json
{"path": "data/librispeech/.../audio.flac"}
```

`src/sirencodec/data_pipeline.py`: LibriSpeech, CommonVoice, VCTK, etc.

### Checkpoints / logs

- Checkpoints: `checkpoints_vocos_vq/codec_step*.pt`
- Metrics: `log_vocos_vq.tsv` (step, mel, adv, commit, VQ util, grad, LR)
- Stdout: `train_run_rvq_600bps.log`, `train_run_*.log`

## Tree (repo root)

```
./
├── run.py
├── README.md
├── docs/               # GUIDE, RESEARCH, ROADMAP, CONVENTIONS
├── src/sirencodec/     # data_pipeline, extras, sidecars, core/
├── tools/              # bench_*, watch, precompute_mels
├── tests/
├── scripts/
├── data/               # gitignored
├── experiments/        # optional local (gitignored)
└── checkpoints*
```

## GitHub

https://github.com/Krabbens/sirencodec — upstream may track subset (`train_vocos_vq.py`, `README.md`, `requirements.txt`, `.gitignore`).

## Metrics snapshot

| When | Config | Step | mel | SI-SDR | PESQ | VQ% | perf |
|------|--------|------|-----|--------|------|-----|------|
| ref | Vocos+RVQ 4×1024@94fps | 10k | 0.34 | -28.5 dB | 2.15 | 46% | ~335 ms/step |
| ref | Vocos+RVQ 2×1024@30fps | — | — | — | — | — | — |
| — | SEANet+Mamba / Zipformer | any | — | -32…-44 dB | ~1.03 | — | — |

## Conventions

- Mostly single-file scripts
- Log stdout + `.log`; checkpoints every 5k; PESQ every 5k on 5 val clips
- SI-SDR / PESQ @ 16 kHz resampled from 24 kHz
- Audio: [-1,1], 24 kHz, 1 s segments
- LR: gen 1e-4, disc 2.5e-5, cosine after 5k warmup

# CODEC-RESEARCHER v2

## IDENTITY

Autonomous agent. Mission: **min bitrate**, **max perceptual quality**, real-time codec. Run continuously; avoid idle repetition.

## HARD CONSTRAINTS

- Latency: encode+decode ≤20 ms  
- RT-factor: <0.5  
- RAM: ≤100 MB  
- Causal streaming (no future frames)

## METRICS (priority)

1. bitrate (bps) — minimize  
2. PESQ — maximize  
3. ViSQOL — maximize  
4. SI-SDR — maximize  
5. latency_ms — minimize  
6. params (M) — minimize  

## CYCLE PROTOCOL

```
EVERY CYCLE:
1. Read results.tsv — what worked / failed
2. Highest-priority open thread (exploit) or new hypothesis (explore)
3. One-sentence hypothesis
4. Design with numbers (layers, rates, formulas)
5. VERDICT: BREAKTHROUGH | PROGRESS | NEUTRAL | DEAD_END
6. Append row to results.tsv (mandatory)
7. DEAD_END → why, blacklist, pivot
8. BREAKTHROUGH → deep-dive next 5 cycles
9. Every 25 cycles: CHECKPOINT (≤5 lines)
10. Start next cycle
```

## EXPLORE vs EXPLOIT

- ~70% exploit best direction  
- ~20% adjacent explore  
- ~10% radical  
- No progress ~50 cycles → 100% explore for 10 cycles  

## BREAKTHROUGH FLAGS

- ≤500 bps + PESQ>3.5  
- ≤1000 bps + PESQ>4.0  
- ≤200 bps + PESQ>2.5  
- >50% bitrate vs EnCodec at same quality  

## results.tsv

Append each cycle. Columns:

`cycle|phase|hypothesis|arch_id|bitrate_bps|pesq_est|visqol_est|latency_ms|params_M|verdict|key_finding|next_action`

Verdict: `BREAKTHROUGH` / `PROGRESS` / `NEUTRAL` / `DEAD_END` / `BASELINE`

## FILE DISCIPLINE

- **SYSTEM.md** — cycle protocol; change only when protocol/rules change  
- **RESEARCH.md** — knowledge log (append; compress old sections if needed, don’t drop facts)  
- **results.tsv** — one row per cycle (append-only)  
- **train.py** / pipeline — overwrite when architecture wins  

---

# Thesis pipeline (fps sweep + refiner + student)

## A — 94 fps bitrate sweep

Native Vocos rate: **ResidualVQ** on 100-dim mels, no Zipformer bottleneck, `upsampling_factor=1`.

```bash
python3 run.py train_vocos_vq --steps 50000 --rvq --n-codebooks 1 --codebook-size 64 \
  --mel-fps 94 --bottleneck-dim 0 --batch-size 8 --data-dir data \
  --log-tsv log_sweep_94fps_1x64.tsv
```

Full sweep (32…1024, 2×1024, 4×1024 @94fps):

```bash
STEPS=50000 BS=8 ./scripts/run_thesis_sweep.sh
```

Checkpoints carry `cfg`. Logs: `train_run_sweep_94fps_*.log`, `log_sweep_*.tsv`.

## B — Mel refiner

After codec ckpt (`checkpoints_vocos_vq/codec_step*.pt` or `codec_final.pt` w/ `cfg`):

```bash
python3 run.py sidecars mel_refiner \
  --codec-checkpoint checkpoints_vocos_vq/codec_final.pt \
  --steps 20000 --batch-size 16 --data-dir data \
  --out-dir checkpoints_mel_refiner
```

Optional: `--noise-std 0.02`. Out: `checkpoints_mel_refiner/mel_refiner.pt`.

## C — Student vocoder

Teacher: Vocos. Student: `StudentVocoder` in `sidecars.py` (~0.3–2M; raise `--base` toward ~3M).

```bash
python3 run.py sidecars distill --steps 50000 --batch-size 16 --data-dir data \
  --base 384 --out-dir checkpoints_student_vocoder
```

## Files

| File | Role |
|------|------|
| [src/sirencodec/core/train_vocos_vq.py](src/sirencodec/core/train_vocos_vq.py) | Codec; `encode_to_coarse_mel()` for refiner |
| [src/sirencodec/extras.py](src/sirencodec/extras.py) | `MelRefinerNet` (+ HiFi-GAN HF helper) |
| [src/sirencodec/sidecars.py](src/sirencodec/sidecars.py) | `mel_refiner` / `distill` training |
| [scripts/run_thesis_sweep.sh](scripts/run_thesis_sweep.sh) | Batch Track A |

`python3 run.py …` — see `README.md`.
