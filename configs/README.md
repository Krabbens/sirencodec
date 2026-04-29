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
  - early RVQ split `A=5%`, `B=65%`, `D=30%`; RVQ and marginal entropy start at 25% strength as soon as B begins
  - no waveform GAN by default; late adversarial fine-tuning was too noisy on the observed run
  - roughly triples AE capacity with wider encoder/decoder channels while keeping full-latent 256/128 RVQ codebooks unchanged (`~0.94 kbps`)
  - adds a small identity-initialized Conv2d bottleneck over latent time × feature bands
  - avoids factorized RVQ `out_proj` and post-RVQ stack cold-start after the AE warmup
  - uses a large STFT stack (`512/1024/2048/4096/8192`) for harmonic detail, but keeps it moderated so magnitude matching does not overpower waveform phase
  - adds a cheap SI-SDR waveform term; this protects stage A from the mag-STFT-only failure mode without paying the full grad-balancer cost
  - limits expensive spectral losses to a deterministic subset (`spectral_batch_items`) and runs `4096/8192` every 4 steps so the large FFTs stay practical
  - keeps a continuous-AE anchor during RVQ phases so the good A-stage encoder/decoder path does not drift while hard RVQ improves
  - uses a long quantization blend ramp and stronger marginal entropy to avoid low-bitrate index collapse
  - fixed validation clips and best-checkpoint saving make late-run drift easier to avoid
  - default `dataset=train-clean-360`, `epochs=200`, `logs_per_epoch=1`

- `sub1k_5090_stable_200.json`
  - long RTX 5090-oriented stable run for the observed failure mode where RVQ code usage recovers but quantized reconstruction drifts and GAN/fm hurts late quality
  - keeps the same sub-1 kbps topology as `sub1k_200.json` (`2` RVQ stages, `K=256,128`, stride `256x`) but uses a 200-epoch curriculum with `A=8%`, `B=72%`, `D=20%`
  - disables GAN for the main run; use the resulting checkpoint for a separate short adversarial fine-tune only after SI-SDR/cosine are stable
  - lowers VQ/marginal pressure, enables quantization blending, and strengthens waveform/AE anchors so RVQ learns to follow the good AE path instead of overpowering reconstruction
  - spends extra 5090 compute on spectral quality (`32` spectral items, large FFTs every `2` steps, mild excess/high-frequency losses) while keeping the effective batch at `256` for comparable update count

- `sub1k_harmonic_20.json`
  - short 20-epoch test preset for the high-frequency smear fix
  - keeps the same sub-1 kbps RVQ bitrate/codebooks as `sub1k_200.json` (`2` stages, `K=256,128`, same stride)
  - trains reconstruction on hard RVQ immediately after `A=5%`; only VQ/marginal weights ramp in `B=15%`
  - enables combined MSD+MPD (`disc_type=msmpd`) plus feature matching to reward harmonic structure while suppressing broadband noise
  - adds pre-emphasized waveform L1 and an excess log-STFT penalty so high bands do not become a smeared spectral floor
  - runs the large `4096/8192` STFT stack every 2 steps on a 24-item spectral subset for stronger high-frequency pressure without full-batch FFT cost
