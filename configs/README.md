# Config Templates

Ready-made presets for `uv run train --config ...`.

## Usage

```powershell
uv run train --config configs/ab_50_50.json
```

CLI overrides the JSON file:

```powershell
uv run train --config configs/abd.json --dataset train-clean-100 --epochs 5
```

Each template can include run-length and logging defaults such as:

- `epochs`
- `logs_per_epoch`
- `viz_per_epoch`

## Presets

- `ab_50_50.json`
  - only phases `A` and `B`
  - `A=50%`, `B=50%`
  - no `D`, no `C`
  - default `dataset=train-clean-360`, `epochs=140`, `logs_per_epoch=5`

- `abd.json`
  - phases `A`, `B`, `D`
  - default split `A=15%`, `B=35%`, `D=50%`
  - no `C`
  - default `dataset=train-clean-100`, `epochs=5`, `logs_per_epoch=5`

- `abcd.json`
  - full curriculum `A`, `B`, `C`, `D`
  - default split `A=15%`, `B=35%`, `C=25%`, `D=25%`
  - `C` becomes active only when `lambda_adv > 0`
  - default `dataset=train-clean-100`, `epochs=5`, `logs_per_epoch=5`

- `sub1k_200.json`
  - recommended longer sub-1 kbps LibriSpeech run after the weak `ab_50_50` result
  - stable split `A=30%`, `B=55%`, `D=15%`
  - no waveform GAN by default; late adversarial fine-tuning was too noisy on the observed run
  - lower VQ pressure, pre-RVQ LayerNorm, and full-latent 256/128 RVQ (`~0.94 kbps`)
  - avoids factorized RVQ `out_proj` and post-RVQ stack cold-start after the AE warmup
  - uses a large STFT stack (`512/1024/2048/4096/8192`) for harmonic detail, but keeps it moderated so magnitude matching does not overpower waveform phase
  - adds a cheap SI-SDR waveform term; this protects stage A from the mag-STFT-only failure mode without paying the full grad-balancer cost
  - limits expensive spectral losses to a deterministic subset (`spectral_batch_items`) and runs `4096/8192` every 4 steps so the large FFTs stay practical
  - keeps a continuous-AE anchor during RVQ phases so the good A-stage encoder/decoder path does not drift while hard RVQ improves
  - uses a long quantization blend ramp and stronger marginal entropy to avoid low-bitrate index collapse
  - fixed validation clips and best-checkpoint saving make late-run drift easier to avoid
  - default `dataset=train-clean-360`, `epochs=200`, `logs_per_epoch=1`
