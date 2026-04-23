# 2026 roadmap (cycle ~22 snapshot)

## State

**Implemented**: SEANet-small encoder (320× down, causal, GRU, running mean), HiFi-GAN decoder (~3.94M), RVQ (e.g. 8×1024×50 fps), 2-stage train (AE → VQ), MR-STFT disc, psychoacoustic masking, speaker FiLM (ARCH-A-SPK), variants ARCH-A-v2b, B, C, D.

**Numbers (70k, LibriSpeech)**: PESQ ~1.37 **without VQ** / ~1.31 **with VQ @500 bps**; VQ util ~3.3%; SI-SDR ~-25 / -27 dB; ~6.53M params; ~85 ms/step RTX 3090.

**Bottleneck**: Encoder–decoder ceiling ~**PESQ 1.37** at infinite bitrate — not the quantizer (VQ cost ~0.06 PESQ). More steps / bigger codebooks won’t close gap to ~3.5+ alone.

---

## Priorities

### 1 — Mamba-2 at fine stages (high impact)

Replace late encoder / decoder conv stacks with **Mamba-2** where it helps temporal structure; bidirectional at bottleneck, causal at decode. Expect +0.8–1.2 PESQ at continuous rate; +15–20% params; **high** integration cost.

### 2 — Dynamic FF + windows (medium–high)

Per-frame gating (2×→3× FF), RoPE/xPos on local attention; better plosives / onsets. +0.3–0.5 PESQ est.; **medium** cost.

### 3 — 4-bit Q + speculative decode (deploy)

4-bit weights, 8-bit activations at 50 Hz stage; tiny draft model for speculation. ~3× memory; ~2.5× effective speed; **medium** cost.

### Quick wins

- SwiGLU, RMSNorm — low risk  
- Flash-style kernels on attention stages — low–medium  

---

## Phases

**Phase 1 (weeks 1–4)**: Mamba-2 at 50 Hz stages + SwiGLU/RMSNorm → target PESQ **2.2+** continuous.

**Phase 2 (5–8)**: Dynamic FF routing, RoPE.

**Phase 3 (9–12)**: PTQ 4-bit + speculative decoding.

---

## Code touchpoints

| File | Change |
|------|--------|
| `train.py` | Mamba blocks, SwiGLU, RMSNorm |
| `train_pipeline.py` | Mamba LR / QAT hooks |
| `data_pipeline.py` | Usually unchanged |

**New (if pursued)**: `mamba_ssm.py`, `dynamic_routing.py`, `quantization.py`, `speculative_decode.py`

## Risk table

| Change | Risk | Impact | Revert |
|--------|------|--------|--------|
| Mamba-2 | high | very high | medium |
| Dynamic FF | medium | high | easy |
| 4-bit Q | medium | high | easy |
| SwiGLU/RMSNorm | low | medium | easy |
| Speculative dec | low | medium | easy |

## Next moves

1. SwiGLU + RMSNorm (hours)  
2. Mamba-2 research + encoder integration  
3. Dynamic routing on LibriSpeech  
4. Q + speculation for edge  

Pick: **A** Mamba-2, **B** quick norms, **C** dynamic FF, **D** other.
