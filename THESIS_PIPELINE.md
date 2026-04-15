# Thesis pipeline (Frame rate + generative refinement)

## Track A — 94fps bitrate sweep

Native Vocos frame rate (`mel_fps == 94`): direct **ResidualVQ** on 100-dim log-mels, **no** Zipformer bottleneck, **no** learned upsampler (`upsampling_factor=1`).

```bash
# One experiment (example: 564 bps)
python3 train_vocos_vq.py --steps 50000 --rvq --n-codebooks 1 --codebook-size 64 \
  --mel-fps 94 --bottleneck-dim 0 --batch-size 8 --data-dir data \
  --log-tsv log_sweep_94fps_1x64.tsv

# Full sweep (32,64,128,256,1024) + 2×1024 + 4×1024 @ 94fps
STEPS=50000 BS=8 ./scripts/run_thesis_sweep.sh
```

Checkpoints include `cfg` for downstream scripts. Logs: `train_run_sweep_94fps_*.log`, `log_sweep_*.tsv`.

## Track B — Mel refiner (conditional denoising)

After a codec checkpoint exists (`checkpoints_vocos_vq/codec_step*.pt` or `codec_final.pt` with `cfg`):

```bash
python3 train_mel_refiner.py \
  --codec-checkpoint checkpoints_vocos_vq/codec_final.pt \
  --steps 20000 --batch-size 16 --data-dir data \
  --out-dir checkpoints_mel_refiner
```

Output: `checkpoints_mel_refiner/mel_refiner.pt` (+ optional `--noise-std 0.02`).

## Track C — Student vocoder distillation

Teacher: pretrained Vocos. Student: `StudentVocoder` in `train_vocos_distill.py` (~0.3–2M params; increase `--base` toward ~3M).

```bash
python3 train_vocos_distill.py --steps 50000 --batch-size 16 --data-dir data \
  --base 384 --out-dir checkpoints_student_vocoder
```

## Files

| File | Role |
|------|------|
| [train_vocos_vq.py](train_vocos_vq.py) | Codec training; `encode_to_coarse_mel()` for refiner |
| [mel_refiner.py](mel_refiner.py) | `MelRefinerNet` (~1M params) |
| [train_mel_refiner.py](train_mel_refiner.py) | Train refiner on frozen codec |
| [train_vocos_distill.py](train_vocos_distill.py) | Knowledge distillation to small waveform decoder |
| [scripts/run_thesis_sweep.sh](scripts/run_thesis_sweep.sh) | Batch Track A runs |
