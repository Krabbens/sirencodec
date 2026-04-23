# RESEARCH LOG

## KNOWN LANDSCAPE

### Pareto — codecs (reference)

| Codec | bps | PESQ | Latency | Params | Year | Trick |
|-------|-----|------|---------|--------|------|-------|
| Opus | 6000 | 4.2 | 20ms | classical | 2012 | MDCT+SILK |
| EVS | 5900 | 4.3 | 32ms | classical | 2014 | parametric |
| SoundStream | 3000 | 3.9 | 7ms | 8M | 2021 | RVQ+adv |
| Lyra v2 | 3200 | 3.8 | 20ms | 2.5M | 2022 | SS-derived |
| EnCodec | 1500 | 3.5 | 13ms | 15M | 2022 | RVQ+balancer |
| DAC | 8000 | 4.3 | N/A | 74M | 2023 | RVQ+snake |
| Mimi | 1100 | 3.6 | 13ms | ~15M | 2024 | sem+acoustic |
| WavTokenizer | 900 | ~3.3 | N/A | ~20M | 2024 | extreme VQ |
| SemantiCodec | 500 | ~3.0 | N/A | large | 2024 | semantic |
| Codec2 | 700 | 2.5 | 40ms | classical | 2017 | LPC |

### Theory (short)

- Shannon R(D); cochlear raw rate vs cortical compression  
- Speech content floor ~60–90 bps (phoneme-ish); + prosody ~50 bps + speaker one-time ~256 bits → **~150 bps** rough floor for “content” stories  

### Technique catalog

**Quant**: RVQ (N×K×fps), FSQ (no collapse), LFQ, product quant, binary/ternary  
**Enc**: SEANet, Conv1D, Conformer, ConvNeXt  
**Dec**: HiFi-GAN, Vocos (iSTFT), waveform  
**Disc**: MSD, MPD, MR-STFT  
**Tricks**: entropy coding on indices (ANS + prior), BWE, speaker conditioning, temporal prediction, sem/acoustic split, joint denoise+compress, psychoacoustic loss weighting  
**Ultra-low**: Codec2 parametric; HuBERT-ish tokens + entropy coding; TTS resynth; speaker+linguistic only  

---

## CANDIDATE ARCHITECTURES (summary)

| ID | Idea | Target bps | Status |
|----|------|------------|--------|
| **ARCH-A-v2b** | SEANet 320×@16k→50fps, GRU+running mean, RVQ/FSQ, Vocos, psych mask | ~550 raw, ~450 w/ entropy | implemented |
| **ARCH-A-SPK** | +speaker FiLM, 1024 cb content stream | ~500 | implemented |
| **ARCH-B-v1b** | 48-d sem + 80-d acoustic VQ, dual stream | 1000 (500+500) | implemented |
| **ARCH-C-v1** | novelty keyframes + interpolate | ~226 | implemented, high risk |
| **ARCH-D-v1** | 25fps coarse + 50fps fine VQ | ~700 | implemented |
| **ARCH-E** | HiFi-GAN decoder instead of Vocos (in `train.py` / pipeline) | 500 | explored |

Priority to train (historical): **A-SPK** > **A-v2b** > **B** > **D** > **C**.

---

## FINDINGS (cycles, compressed)

**C1–2**: Landscape + ARCH-A math; utilization ~40–60% typical; 2048 cb + bigram prior + GRU context sketched. **BASELINE / PROGRESS**

**C3**: Running mean μₜ=0.99μₜ₋₁+0.01xₜ; GRU residual α path; fix entropy vs 0.8 fudge. **PROGRESS**

**C4**: ARCH-B bottleneck disentangle 32/48-d sem + acoustic residual without HuBERT. **PROGRESS**

**C5**: Psychoacoustic masking (Bark spread S(Δ)=10^(-|ΔBark|/3)), weighted log-mel L1. **PROGRESS**

**C6–7**: ARCH-B dims tuned (48/80); ARCH-C adaptive keyframes + repeat bit (~226 bps est.). **PROGRESS**

**C8**: All three families specified; priority A-v2b → B → C. **PROGRESS**

**C10**: ARCH-D 25+50 fps multi-scale VQ ~700 bps. **PROGRESS**

**C12**: FSQ [3]^7 ~11.1 bits/frame ~550 bps; `use_fsq` in ResidualVQ. **PROGRESS**

**C13**: Speaker FiLM; 256-d from 500 ms; ~500 bps content. **PROGRESS**

**C14 CHECKPOINT**: 4 archs × quant variants; no breakthrough; need **real** PESQ. **NEUTRAL**

**C15**: AudioCodecASPK integrated. **PROGRESS**

**C17**: Progressive disc stages; uncertainty loss weighting. **PROGRESS**

**C18 — bugs + data**: Fixed channel indexing/padding, running-mean dims, 179M decoder bug (remove bad transposed conv), mel/feat/SDR length trims, `load_audio` soundfile for FLAC; `--real-data`; ~11–12M params real; forward+short train OK. **PROGRESS**

**C19 — real Libri**: train-clean-100 manifest; 500 steps ARCH-A-v2b: mel 3.05→2.45, VQ 0.1%→5.6%, eff ~465 bps; **log.tsv** vs **results.tsv** split per SYSTEM. torchcodec/CUDA issue on CPU → soundfile. **PROGRESS**

**C20–21 — GPU**: 85 ms/step 3090; 20k steps mel 3.7→1.0, VQ 0.1%→18%; **PESQ 1.261 @10k**; SI-SDR ~-29 dB; commit loss >> mel later (codebook stress). MRSTFT trim fix. **PROGRESS**

**C22 — diagnostic**: **No-VQ PESQ 1.369** vs **VQ 1.307 @500 bps** → VQ cost **0.062** PESQ only. Bottleneck = **encoder+Vocos path**, not quantizer. HiFi-GAN decoder direction; `decoder_type` vocos/hifigan. **DEAD_END (Vocos as sole decoder)** / **insight BREAKTHROUGH**

| Mode | PESQ | SI-SDR | VQ util |
|------|------|--------|---------|
| No VQ | 1.369 | -25.0 | — |
| VQ 500 | 1.307 | -27.3 | 3.3% |

**C23 — two-stage**: Stage1 AE no VQ; Stage2 freeze encoder, train VQ+decoder. HiFi-GAN v4-ish stack; dry runs OK; long run launched. **PROGRESS**

---

## Cycle 19 table (500 steps)

| Metric | Value |
|--------|-------|
| mel loss | 2.45 |
| VQ util | 5.6% |
| Eff. bitrate | ~465 bps |
| Params | ~6.8M (reduced cfg) |
| Speed | ~360 ms/step CPU |

## Cycle 20–21 trajectory

| Step | mel | VQ util | commit |
|------|-----|---------|--------|
| 0 | 3.74 | 0.1% | 0.68 |
| 2500 | 2.5 | 8% | 2.5 |
| 10000 | 0.95 | 15% | 6.0 |
| 19500 | 1.02 | 18% | 9.3 |

---

## Open threads

- Two-stage HiFi-GAN + VQ: measure PESQ vs Vocos line (`train_vocos_vq.py`)  
- Mamba / representation fix per ROADMAP_2026  
- Real eval at 50k–100k steps; watch commit–mel balance  

Append new cycles below; keep **results.tsv** in sync with SYSTEM.md.
