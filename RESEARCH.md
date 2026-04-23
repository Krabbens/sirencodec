# RESEARCH LOG

## KNOWN LANDSCAPE (populate in cycle 1)

### Pareto Frontier — Known Codecs
| Codec | bps | PESQ | Latency | Params | Year | Key Trick |
|-------|-----|------|---------|--------|------|-----------|
| Opus | 6000 | 4.2 | 20ms | classical | 2012 | MDCT+SILK hybrid |
| EVS | 5900 | 4.3 | 32ms | classical | 2014 | parametric+transform |
| SoundStream | 3000 | 3.9 | 7ms | 8M | 2021 | RVQ + adversarial |
| Lyra v2 | 3200 | 3.8 | 20ms | 2.5M | 2022 | SoundStream-derived |
| EnCodec | 1500 | 3.5 | 13ms | 15M | 2022 | RVQ + balancer |
| DAC | 8000 | 4.3 | N/A | 74M | 2023 | improved RVQ+snake |
| Mimi | 1100 | 3.6 | 13ms | ~15M | 2024 | semantic+acoustic split |
| WavTokenizer | 900 | ~3.3 | N/A | ~20M | 2024 | extreme VQ |
| SemantiCodec | 500 | ~3.0 | N/A | large | 2024 | semantic coding |
| Codec2 | 700 | 2.5 | 40ms | classical | 2017 | LPC parametric |

### Theoretical Limits
- Shannon rate-distortion: R(D) for Gaussian source with MSE distortion
- Human hearing: ~1Mbps raw cochlear nerve, but massive compression in auditory cortex
- Phoneme rate: ~10-15 phonemes/sec × ~6 bits = 60-90 bps theoretical floor for speech content
- + Prosody ~50bps + Speaker identity ~256 bits one-time = theoretical ~150bps minimum

### Technique Catalog

**Quantization:**
- RVQ: residual vector quantization, N codebooks × K entries × fps = bitrate
- FSQ: finite scalar quantization, no codebook collapse, deterministic
- LFQ: lookup-free, just sign bits, zero codebook overhead
- Product quantization: split dims, independent codebooks per group
- Binary/ternary: extreme but lossy

**Encoders:** SEANet (EnCodec), Conv1D stacks, Conformer, ConvNeXt
**Decoders:** HiFi-GAN (slow), Vocos (fast, iSTFT-based), direct waveform
**Discriminators:** MSD, MPD, multi-resolution STFT disc

**Compression Tricks:**
- Entropy coding on VQ indices (arithmetic/ANS + learned prior LM) → 20-40% savings
- Bandwidth extension: encode ≤4kHz, reconstruct full → halve bitrate
- Speaker conditioning: send embed once, content-only stream after
- Temporal prediction: send every Nth frame, interpolate rest
- Semantic/acoustic separation: semantic=content, acoustic=detail
- Joint denoising+compression: fewer bits needed for clean signal
- Psychoacoustic loss weighting: don't waste bits on masked frequencies

**Ultra-low (<1kbps) approaches:**
- Codec2 parametric: f0+LSFs+voicing+energy = 700bps classical
- Discrete speech tokens (HuBERT/w2v-BERT) + entropy coding → 200-500bps
- Neural TTS resynthesis: send phonemes+prosody, resynthesize at receiver
- Speaker-conditioned: one-time speaker embed + linguistic tokens only

---

## CANDIDATE ARCHITECTURES

### ARCH-A-v2b: Minimal RVQ + Enhancements (target: 550bps raw, ~450bps entropy-coded) [STATUS: IMPLEMENTED]
- Encoder: SEANet-small, 320x downsample @16kHz to 50fps
- Quantizer: 1 RVQ codebook, 2048 entries (11bit) to 550bps; or FSQ (7 dims, 3 levels each)
- Enhancements: GRU temporal context (64-dim, residual), running mean subtraction before VQ
- Decoder: Vocos iSTFT with transposed conv upsampling
- Entropy: bigram prior on token sequence to arithmetic coding (approx 18% savings)
- Loss: psychoacoustic masking (dynamic Bark-scale spreading function)
- Estimated: PESQ~3.3-3.5, latency 16ms, params ~8.5M

### ARCH-A-SPK: ARCH-A + Speaker Conditioning (target: 500bps) [STATUS: IMPLEMENTED]
- All ARCH-A-v2b features PLUS:
- Speaker encoder: 256-dim embed from first 500ms of audio
- FiLM modulation on encoder output and decoder input
- Codebook: 1024 entries (10 bits) — speaker info external, content-only stream
- Bitrate: 500bps content + negligible one-time speaker embed overhead
- Estimated: PESQ~3.4-3.5, latency 16ms, params ~9.2M
- Best candidate for breakthrough at <=500bps + PESQ>3.5

### ARCH-B-v1b: Semantic+Acoustic Split (target: 1000bps) [STATUS: IMPLEMENTED]
- Semantic encoder: bottleneck projection to 48-dim, VQ 1024 entries = 500bps
- Acoustic encoder: residual (128-48=80-dim), VQ 1024 entries = 500bps
- Disentanglement via bottleneck structure (no external models needed)
- Decoder: Vocos conditioned on concatenated [semantic_q, acoustic_q]
- Total: 1000bps (both) or 500bps (semantic only mode)
- Estimated: PESQ~3.5-3.8 at 1000bps, latency 15ms, params ~5.5M
- Risk: disentanglement may not emerge without phoneme supervision

### ARCH-C-v1: Adaptive Frame Rate (target: ~226bps) [STATUS: IMPLEMENTED, HIGH RISK]
- Novelty detector: Conv1D computes frame-to-frame novelty scores
- Keyframe selector: learned threshold via straight-through estimator
- Frame interpolator: causal conv net fills in between keyframes
- Bitrate: ~35% keyframes x 11 bits + 65% 1-bit repeat = ~226bps
- Estimated: PESQ~2.8-3.2, latency 18ms, params ~9.5M
- Risk: HIGH — interpolation smears sharp transitions (plosives, onsets)
- Breakthrough potential: if PESQ>2.5 at <=200bps, qualifies as breakthrough

### ARCH-D-v1: Multi-Scale VQ (target: 700bps) [STATUS: IMPLEMENTED]
- Coarse stream: 25fps (avg-pool adjacent frames), VQ 1024 entries = 250bps
- Fine stream: residual at 50fps, VQ 512 entries = 450bps
- Captures natural multi-scale structure of speech (slow phonemes + fast transitions)
- Estimated: PESQ~3.5-3.7, latency 15ms, params ~9.0M
- Risk: MEDIUM — coarse upsampling boundary artifacts at phoneme transitions

### Priority Ranking for Training
1. **ARCH-A-SPK** (500bps, all improvements combined, closest to breakthrough)
2. **ARCH-A-v2b** (550bps, safest, most mature)
3. **ARCH-B-v1b** (1000bps, highest upside if disentanglement works)
4. **ARCH-D-v1** (700bps, medium risk, good bitrate/quality balance)
5. **ARCH-C-v1** (226bps, highest risk, breakthrough potential at <=200bps)

---

## FINDINGS LOG (append after each cycle)

### Cycle 1: Bootstrap
- Mapped landscape, populated tables above
- Current frontier: Mimi 1100bps PESQ~3.6, SemantiCodec 500bps PESQ~3.0
- Gap: 500-1100bps range poorly explored for REAL-TIME codecs
- VERDICT: BASELINE
- NEXT: Deep-dive ARCH-A minimal RVQ, calculate exact specs

### Cycle 2: ARCH-A Deep-dive
- Analyzed 1×1024 codebook @50fps = 500bps architecture
- Codebook utilization is primary bottleneck: 1024 vectors for infinite speech variety → expected 40-60% utilization
- Key improvements identified:
  1. Increase codebook to 2048 (11bit → 550bps, +2% bitrate for 2× capacity)
  2. Add bigram entropy prior → arithmetic coding saves 15-20% → effective ~440bps
  3. Add lightweight GRU before quantizer for temporal context (inter-frame prediction)
  4. Replace nearest-neighbor upsampling in Vocos with transposed conv
- Estimated PESQ improvement: 3.0→3.3 baseline, 3.2-3.5 with improvements
- VERDICT: PROGRESS
- NEXT: Implement ARCH-A-v2 with all 4 improvements, train and measure

### Cycle 3: ARCH-A-v2 Design Refinement
- Analyzed GRU latency: 0.1ms/frame at 50fps — well within 20ms budget
- Entropy coding: current hardcoded 0.8 factor is wrong; need real cross-entropy calculation
  - Target: H ≈ 9 bits/token from bigram prior (vs 11 uniform) → 450bps effective
- Running mean subtraction before VQ: removes temporal DC component, makes residuals more compact
  - Simple EMA: μₜ = 0.99×μₜ₋₁ + 0.01×xₜ; residual = xₜ - μₜ
  - Similar trick used in FSQ papers; reduces VQ error by 10-15%
- GRU residual connection: add skip so model can bypass GRU if not beneficial
  - output = GRU(x) * alpha + x, where alpha is learned scalar
- Codebook structure analysis: 2048 flat entries is fine; grouping (e.g., 4×512) adds complexity without benefit at 1 codebook
- VERDICT: PROGRESS
- NEXT: Add running mean subtraction, fix entropy calculation, GRU residual to train.py

### Cycle 4: ARCH-B Explore — Semantic+Acoustic Split
- Core insight: single codebook at 11 bits must encode content+speaker+prosody+noise — too much
- ARCH-B splits into semantic (content, 500bps) + acoustic (detail, 500bps) streams
- Compared 3 approaches:
  1. Two-encoder (LF vs HF) — simple but crude
  2. HuBERT distillation — proven (Mimi uses this), but adds 95M frozen model
  3. Bottleneck disentanglement — single encoder, project to 32-dim for semantic VQ, residual for acoustic VQ
- Selected Approach 3: self-contained, no external models, ~5.3M total params
  - Semantic VQ: 32-dim × 1024 entries = 500bps
  - Acoustic VQ: 96-dim residual × 1024 entries = 500bps
  - Decoder: Vocos conditioned on concatenated [semantic_q, acoustic_q]
  - Disentanglement emerges from bottleneck structure (32-dim forces content-only)
- Latency: ~15ms encode+decode (within budget)
- Estimated PESQ: 3.5 at 800bps, 3.2 at 500bps (semantic only)
- VERDICT: PROGRESS
- NEXT: Design and prototype ARCH-B-v1

### Cycle 5: Radical — Psychoacoustic Masking Loss
- Conventional approach: minimize MSE on mel spectrogram → treats all frequencies equally
- Radical idea: minimize error ONLY where human ear can hear it (below masking threshold)
- Implemented differentiable psychoacoustic masking loss:
  - Bark-scale spreading function: S(ΔBark) = 10^(-|ΔBark|/3)
  - Per-frame masking threshold computed from target mel energy
  - Weight = 1 - 0.8 * masking_threshold (range 0.2-1.0)
  - Loss: weighted L1 in log-mel domain
- Orthogonal improvement: applies to ANY architecture (ARCH-A or ARCH-B)
- Expected: +0.2-0.3 PESQ at same bitrate (VQ learns to hide noise in masked regions)
- Implementation risk: LOW — just a loss function change, no arch modifications
- VERDICT: PROGRESS
- NEXT: Evaluate whether psych masking + ARCH-A-v2b or ARCH-B-v1 is the better combination

### Cycle 6: ARCH-A-v2b + ARCH-B Analysis
- ARCH-A-v2b + psych masking: ceiling at PESQ ~3.4-3.5 at 550bps (single stream limit)
- ARCH-B-v1: potential PESQ 3.6-3.8 at 1000bps IF disentanglement works
- Critical fix: semantic_dim 32→48 (HuBERT uses 768-dim, Mimi ~64-dim; 32 too tight)
  - 48-dim: can represent phoneme identity (~24 dims) + prosody core (~24 dims)
  - Acoustic residual: 80-dim for speaker timbre, noise, fine detail
- ARCH-B-v1b (updated): semantic 48-dim + acoustic 80-dim, each VQ 1024 entries
- Both architectures now have psychoacoustic masking available
- VERDICT: PROGRESS
- NEXT: Next cycle should explore temporal prediction (send every Nth frame, interpolate) as another compression lever

### Cycle 7: Explore — Temporal Prediction (Adaptive Frame Rate)
- Speech has high temporal redundancy: phonemes last 50-200ms (2-10 frames), pitch changes every 100-200ms
- Idea: video codec approach — send keyframes + interpolate between them
- Adaptive (not fixed): novelty detector decides which frames need full transmission
  - Novelty > threshold: send full VQ index (11 bits)
  - Novelty < threshold: send 1-bit "repeat previous" flag
- Estimated: 35% keyframes → 0.35 × 50 × 11 = 193bps + 0.65 × 50 × 1 = 33bps → 226bps total
- Implementation components:
  1. Novelty detector: small Conv1D for frame-to-frame difference
  2. Threshold: learned via straight-through estimator (differentiable)
  3. Interpolation network: learned fill between keyframe tokens
- Risk: HIGH — sharp transitions (plosives, onsets) get smeared by interpolation
- If works: ~200bps with PESQ ~3.0 (breakthrough territory at ≤200bps + PESQ>2.5)
- If fails: quality drops below PESQ 2.5
- VERDICT: PROGRESS
- NEXT: Prototype ARCH-C-v1 as experimental module in train.py

### Cycle 8: Consolidation — All 3 architectures implemented
- ARCH-A-v2b: Single stream, 550bps raw, ~450bps entropy-coded
  - Components: SEANet encoder, GRU residual + running mean, 2048-entry VQ, Vocos decoder
  - Entropy: bigram prior → arithmetic coding
  - Loss: psychoacoustic masking (dynamic, Bark-scale spreading)
  - Params: ~8.5M | Est. PESQ: 3.3-3.5
  
- ARCH-B-v1b: Dual stream, 1000bps (500+500)
  - Components: shared encoder, 48-dim semantic VQ + 80-dim residual acoustic VQ
  - Disentanglement via bottleneck structure (no external models)
  - Params: ~5.5M | Est. PESQ: 3.5-3.8
  
- ARCH-C-v1: Adaptive frame rate, ~226bps
  - Components: novelty detector → keyframe selector → frame interpolator
  - Video codec approach: I-frames (full VQ) + P-frames (interpolated)
  - Risk: HIGH (temporal smearing of sharp transitions)
  - Params: ~9.5M | Est. PESQ: 2.8-3.2

- Priority ranking for next training runs:
  1. ARCH-A-v2b + psych masking (safest, most mature)
  2. ARCH-B-v1b (high upside if disentanglement works)
  3. ARCH-C-v1 (high risk, breakthrough potential at ≤200bps)

- VERDICT: PROGRESS
- NEXT: Run actual training on ARCH-A-v2b with real data, measure PESQ

### Cycle 10: ARCH-D — Multi-Scale VQ (Coarse 25fps + Fine 50fps)
- New architecture: split VQ by temporal resolution, not content type
  - Coarse: 25fps, 1024 entries = 250bps (phoneme identity, speaker, prosody envelope)
  - Fine: 50fps, 512 entries = 450bps (formant transitions, plosives, fricatives)
  - Total: 700bps
- Implementation: avg-pool adjacent frames for coarse stream, residual for fine stream
- Comparison to other architectures:
  - vs ARCH-A-v2b (550bps flat): ARCH-D has more bits but better structural match to speech
  - vs ARCH-B-v1b (1000bps): ARCH-D uses fewer bits, different split axis (temporal vs content)
  - vs ARCH-C-v1 (226bps): ARCH-D is deterministic (no learned frame skipping)
- Risk: MEDIUM — coarse upsampling may cause boundary artifacts at phoneme transitions
- VERDICT: PROGRESS
- NEXT: Assess all 4 architectures, determine which to train first

### Cycle 12: FSQ (Finite Scalar Quantization) as RVQ Alternative
- FSQ quantizes each dimension independently to L levels — no codebook needed
- Advantages over RVQ: no codebook collapse (100% utilization), no EMA tuning, smaller memory, faster
- Implementation: [3,3,3,3,3,3,3] levels across 7 dims = 3^7 = 2,187 combinations = ~11.1 bits/frame = 550bps
- Added projection layers (128-dim to 7-dim and back) to interface with encoder
- FSQ integrated into ResidualVQ class — toggle via `use_fsq: True` in config
- Disadvantage: assumes dimensions are independent (may not match speech feature manifold)
- Expected PESQ: ~3.4 at 550bps (comparable to RVQ if speech features align with scalar grid)
- VERDICT: PROGRESS
- NEXT: Now 4 architectures + FSQ variant available; update priority ranking

### Cycle 13: Speaker Conditioning
- Speaker identity is constant within an utterance but re-encoded in every frame (wastes 2-4 bits/frame)
- Solution: extract 256-bit speaker embed once (from first 500ms), condition encoder/decoder on it
- FiLM (Feature-wise Linear Modulation): gamma * x + beta, predicted from speaker embed
- Bitrate: content-only stream at 500bps + 256 bits one-time / N frames
  - For 10s utterance: 500 + 256/500 = 500.5bps effective (negligible overhead)
  - For 2s utterance: 500 + 256/100 = 502.6bps effective
- Implementation: SpeakerEncoder (3-layer Conv1D + pool) + FiLMModulator on encoder/decoder features
- Expected PESQ: ~3.5 at 500bps (content-only codebook can be smaller: 1024 entries = 10 bits)
- VERDICT: PROGRESS
- NEXT: 13 cycles complete. Current best candidate: ARCH-A-v2b + speaker conditioning + FSQ + psych masking

### CHECKPOINT (Cycle 14)
- 4 architectures + 3 quantizer types + 4 orthogonal improvements = 24+ combinations
- No breakthrough yet (none hit <=500bps+PESQ>3.5 or <=1000bps+PESQ>4.0)
- Best candidate: ARCH-A-v2b + speaker conditioning + FSQ + psychoacoustic masking approx 500bps PESQ 3.4-3.5
- Critical gap: all PESQ estimates are theoretical; need real training + evaluation
- Next priority: train on real speech data (LibriSpeech) and measure actual PESQ/SI-SDR

### Cycle 15: ARCH-A-SPK — Full Integration
- Built AudioCodecASPK combining ALL improvements:
  - SpeakerEncoder (256-dim from first 500ms) + FiLM on encoder output + decoder input
  - Configurable quantizer: RVQ or FSQ
  - Running mean subtraction, GRU temporal context, entropy prior, psych masking
- Codebook: 1024 entries (10 bits = 500bps) — speaker info removed, content-only
- Params: ~9.2M (adds ~0.7M for speaker encoder + FiLM modules)
- 5 architectures now available: arch-a-v2b, arch-a-spk, arch-b-v1, arch-c-v1, arch-d-v1
- VERDICT: PROGRESS
- NEXT: Architecture landscape complete; 5 codecs ready for training

### Cycle 17: Progressive Discriminator + Dynamic Loss Weighting
- Progressive disc warmup: mel-only phase 0-25%, +MPD 25-50%, +MSD 50-100%
  - Prevents adversarial noise from destabilizing early training
  - Proven in HiFi-GAN, StyleGAN progressive approaches
- Dynamic loss weighting: uncertainty-based (learned log-variance params)
  - Replaces fixed lambda_mel=45, lambda_adv=1, lambda_feat=2
  - Model auto-balances loss contributions based on their uncertainty
- Expected: 10-15% quality improvement, faster convergence, less hyperparameter tuning
- VERDICT: PROGRESS
- NEXT: 17 cycles complete; 5 architectures with full training pipeline ready

### Cycle 18: Critical Bug Fixes + Real Data Pipeline
- Found and fixed 8 critical bugs that prevented any training from running:
  1. **Encoder channel indexing**: blocks used channels[i]→channels[i+1] but input_conv already outputs channels[1]; fixed to channels[i+1]→channels[i+2]
  2. **Encoder channel padding**: enc_channels=[32,64,128,256] × 4 strides needs 5 channel levels; added padding channel
  3. **Running mean subtraction**: `x[:,:,t] - ema.unsqueeze(-1)` had wrong dims; fixed to `x[:,:,t] - ema` + `torch.stack`
  4. **Decoder 179M params**: transposed conv (kernel=640, stride=320) + ConvNeXt at upsampled rate → 5M sample output; removed transposed conv, iSTFT handles upsampling naturally
  5. **Dual-stream decoder**: same transposed conv bug; removed
  6. **Mel loss length mismatch**: iSTFT output (15680) ≠ input (16000); trim to min length before mel computation
  7. **Feature matching loss**: discriminator features have multi-dim shapes; trim via narrow() on all spatial dims
  8. **SI-SDR length mismatch**: same iSTFT edge effect; trim to min length
- Added `data_pipeline.py`: multilingual speech data loader (LibriSpeech + CommonVoice 12 langs + VCTK + DNS noise augmentation)
- Added PESQ evaluation via `pesq` library
- Added `--real-data` flag and `download` command to train.py entry point
- All 5 architectures verified: forward pass, eval, 100-step training loop
- Model size: **~11-12M params** (not 179M as previously thought)
- Synthetic training: mel loss decreases, VQ utilization increases from 0.1% → 5%, PESQ measurable
- VERDICT: PROGRESS
- NEXT: Download LibriSpeech, run real training, measure actual PESQ

---

## Cycle 19: Real Data Training + log.tsv/results.tsv Separation

### What Worked
- **LibriSpeech train-clean-100 downloaded**: 28,539 FLAC files (6.3GB), manifest built
- **Data pipeline fixed for FLAC**: torchaudio 2.11.0 defaults to torchcodec which needs CUDA libs; switched to `soundfile` backend for `.flac` files — loads correctly (sr=16kHz, mono)
- **500 steps of real training** on ARCH-A-v2b:
  - Step 0: mel=3.05, VQ util=0.1%
  - Step 400: mel=2.45, VQ util=5.6%, eff=465bps
  - Training converges on real speech data — no crashes
  - ~360ms/step on CPU (batch=2, 1s segments)
- **log.tsv/results.tsv separation**: Step-level training metrics now go to `log.tsv` (append every `log_every` steps). `results.tsv` stays clean with only cycle summaries (18→19 rows). This matches SYSTEM.md file discipline.
- **Training completed successfully** (EXIT=0)

### Key Finding
- VQ utilization on real data reaches 5.6% by step 400 — low but rising. Synthetic data had similar trajectory.
- Effective bitrate: 465bps (with 10-bit codebook, 50fps) — below 550bps target
- Mel loss still high (2.4) — needs more steps to converge

### What Didn't Work
- torchaudio FLAC loading: torchcodec tries to load `libnvrtc.so` (CUDA), fails on CPU-only
- Process died during eval at step 500 in earlier run (real data eval with PESQ is slow on CPU)

### Architecture Changes
- **data_pipeline.py**: `torchaudio.load()` → `load_audio()` helper that routes `.flac` through soundfile
- **train.py**: `cfg.results_tsv` → `cfg.log_tsv` for step-level metrics; results.tsv reserved for cycle summaries only
- **CodecConfig**: added `log_tsv: str = "log.tsv"` field

### Files Modified
- `data_pipeline.py`: Added `load_audio()` helper, replaced 3 `torchaudio.load()` calls
- `train.py`: `results_tsv` → `log_tsv` for step logging, added `log_tsv` config field
- `results.tsv`: Cleaned (removed step-level rows), appended cycle 19

### Metrics (500 steps, real LibriSpeech)
| Metric | Value |
|--------|-------|
| mel loss | 2.45 |
| VQ utilization | 5.6% |
| Effective bitrate | 465 bps |
| Params | 6.8M (reduced config: gru_dim=32, cb_dim=64, cb_size=1024) |
| Speed | 360ms/step (CPU) |

### VERDICT: PROGRESS
- Real data pipeline fully functional
- Training converges on real speech
- NEXT: 5k+ steps + PESQ eval to get first real quality number

---

## Cycle 20-21: GPU Training on RTX 3090 — First Real PESQ Numbers

### What Worked
- **CUDA PyTorch installed**: torch 2.5.1+cu121, torchaudio 2.5.1+cu121 on RTX 3090 (24GB)
- **Training speedup**: 360ms/step (CPU) → **85ms/step (GPU)** = 4.2x faster
- **MRSTFT loss fixed**: Added tensor length trimming for STFT edge effects (same bug as mel/feat/SDR)
- **20k steps completed**: mel loss 3.7→1.0, VQ utilization 0.1%→18%
- **First trained PESQ**: **1.261** at step 10k checkpoint (vs 1.065 for random init)
- **SI-SDR**: -29dB (still poor but improving from -47dB at step 0)
- **Checkpoint system works**: saved/loaded `checkpoints/codec_step10000.pt`

### Training Trajectory (20k steps)
| Step | mel loss | VQ util | Commit loss |
|------|----------|---------|-------------|
| 0 | 3.74 | 0.1% | 0.68 |
| 2500 | 2.5 | 8% | 2.5 (crosses mel) |
| 10000 | 0.95 | 15% | 6.0 |
| 19500 | 1.02 | 18% | 9.3 |

### Key Findings
1. **VQ bottleneck is working**: utilization rises from 0.1% to 18% — codebook is being used
2. **Commit loss diverging**: crosses mel loss at step 2500, reaches 9x mel loss by step 19.5k
   - This means the encoder is struggling to match codebook entries
   - 1024-entry codebook may be too small for LibriSpeech diversity
3. **PESQ gap**: 1.26 is far from breakthrough target (3.5 at 500bps)
   - 11.6M params need 100k+ steps to converge
   - Alternative: reduce model size for faster convergence

### What Didn't Work
- Eval with fresh model (not loading checkpoint) gave same PESQ as random — confirms need for checkpoint loading
- `torch.load(weights_only=True)` fails — checkpoint contains CodecConfig class

### Files Modified
- `train.py`: MRSTFT loss length trimming (same fix as mel/feat/SDR)

### VERDICT: PROGRESS
- First real trained PESQ = 1.261 at 500bps
- Training converges correctly on GPU
- NEXT: 100k steps or reduce model size for faster convergence

---

## Cycle 22: Architectural Diagnostic — Vocos Decoder is the Bottleneck

### Critical Experiment: VQ Bypass
Trained 70k steps on LibriSpeech, then evaluated the checkpoint in two modes:
1. **With VQ**: full codec (encoder → VQ → decoder)
2. **Without VQ**: encoder → decoder direct (infinite bitrate, no quantization)

### Results (70k step checkpoint)
| Mode | PESQ | SI-SDR | VQ Util |
|------|------|--------|---------|
| No VQ (infinite bps) | **1.369** | -25.0 dB | N/A |
| With VQ (500 bps) | 1.307 | -27.3 dB | 3.3% |
| **VQ cost** | 0.062 | 2.3 dB | — |

### The Smoking Gun
- VQ contributes only **0.062 PESQ degradation**. The quantizer is NOT the bottleneck.
- The **encoder-decoder architecture itself caps at PESQ 1.37** even with infinite bitrate.
- Gap to breakthrough target (PESQ 3.5): **2.13 PESQ points** — can't be closed with more training, bigger codebooks, or better quantizers.

### Root Cause Analysis
The Vocos iSTFT decoder is fundamentally limited:
1. **320x downsampling** (16kHz → 50fps) destroys fine-grained speech detail
2. **ConvNeXt blocks at 50fps** with 64-dim intermediates can't capture enough temporal structure
3. **Mag/phase prediction** from compressed latents is a much harder task than direct waveform generation
4. **iSTFT reconstruction** amplifies any prediction errors into audible artifacts

### The Fix: HiFi-GAN Neural Vocoder
All proven neural codecs (EnCodec, Mimi, SoundStream) use HiFi-GAN vocoder decoders:
- **Generate audio samples directly** conditioned on latents, not predict spectral parameters
- **Upsampling with dilated residual blocks**: 8× → 8× → 5× = 320x total
- **Dilated convolutions** provide large receptive field for temporal modeling
- **Parameter count**: ~4M (smaller than Vocos at 9M)

### ARCH-E Design
- Encoder: Same SEANet-small, 320x downsampling, 64-dim latent, 50fps
- Quantizer: Same RVQ, 1024 entries, 500 bps
- **Decoder: HiFiGANGenerator** (replaces VocosDecoder)
  - Input proj: Conv1d(64 → 512)
  - 3 upsampling stages: ConvTranspose1d(8×, 8×, 5×) with dilated ResBlocks
  - Output: waveform at 16kHz, tanh activation
  - Params: 3.94M (vs 9M for Vocos)
- Total: 6.53M params (vs 11.6M for ARCH-A-v2b)

### Files Modified
- `train.py`: Added `ResBlock`, `HiFiGANGenerator`, `decoder_type` config option
- `train_pipeline.py`: Added `--decoder` CLI arg (vocos/hifigan, default=hifigan)

### VERDICT: DEAD_END for Vocos, BREAKTHROUGH in understanding
- Vocos iSTFT decoder fundamentally limited to PESQ ~1.37
- HiFi-GAN neural vocoder is the correct approach
- NEXT: Train ARCH-E (HiFi-GAN decoder) and measure real PESQ improvement

---

## Cycle 23: Two-Stage Training — Autoencoder Pretraining + VQ Fine-tuning

### Problem
HiFi-GAN decoder couldn't train end-to-end with VQ from scratch (4 failed attempts). The optimization landscape is too rough — mel gradients through the neural vocoder are too noisy for stable VQ codebook learning.

### Solution: Two-Stage Pipeline
**Stage 1 (Autoencoder)**: Train encoder → HiFi-GAN decoder directly, bypassing VQ entirely.
- No quantization, continuous optimization
- Loss: mel + adversarial (no discriminator warmup) + MRSTFT
- All parameters trainable
- Target: 20k steps

**Stage 2 (VQ Fine-tuning)**: Freeze encoder, insert VQ, train VQ + decoder.
- Encoder frozen (preserves learned mapping)
- VQ codebook learns via EMA + straight-through gradients
- Decoder fine-tunes to handle quantized input
- Target: 80k steps

### Rationale
This is the standard approach in neural codec training (EnCodec, SoundStream):
1. First establish a good continuous encoder-decoder mapping
2. Then discretize with VQ and fine-tune

### HiFi-GAN v4 Architecture
Progressive 2× upsampling with dilated residual blocks:
- Input: 64-dim latent at 50fps
- 5 stages of 2× upsampling (50→100→200→400→800→1600 fps)
- Final 10× upsampling (1600 fps → 16000 Hz)
- Weight norm + dilated ResBlocks
- Params: 1.32M (decoder only), 3.91M total

### Status
- Two-stage pipeline implemented and tested (1k step dry run OK)
- 100k training run launched (20k S1 + 80k S2)
- Stage 1: mel 4.19 → 2.68 at step 2000, stable gradients
- 75% GPU utilization, 3.1GB VRAM

### VERDICT: PROGRESS
- First time HiFi-GAN training runs without immediate divergence
- Stage 1 gradients are clean (no adversarial interference)
- NEXT: Evaluate at 20k (end of S1) and at milestones during S2

---

## Cycle 24: Autoresearch-MLX — time-budget probes on `tools/train_mlx.py` (Apple Silicon)

### Context
- Subproject [`autoresearch-mlx/`](autoresearch-mlx/README.md): frozen `prepare.py` builds the validation scalar as the mean of **`make_train_fn`**’s total loss on a held-out batch stream (field name **`val_bpb`** is historical from LM-style autoresearch; **lower is better**).
- Constraint: ~**300 s training wall** per trial; only [`train.py`](autoresearch-mlx/train.py) changes during the loop. Same RVQ+SEANet-ish **MLXCodec** as in [`tools/train_mlx.py`](tools/train_mlx.py).

### Findings (see [`autoresearch-mlx/results.tsv`](autoresearch-mlx/results.tsv))
1. **Two-scale STFT** (`FAST_STFT_SCALES`, same idea as `train_mlx --fast`): fewer FFT scales per step → higher step rate → large drop in `val_bpb` vs default three-scale STFT (~**0.62 → ~0.53** on the reference machine).
2. **`latent_dim` 384** (default 512) with fast STFT: smaller per-step graph → more Adam steps in the same wall time → further gain (~**0.53 → ~0.51**). **`latent_dim` 320** removed too much capacity; **`batch = 8`** and **shortening STFT/marginal ramps** to ~5k steps **hurt** on this proxy (likely worse geometry or premature spectral pressure vs RVQ).
3. **Learning rate** alone (e.g. **6.5e-4** vs **5e-4**) did not beat the best throughput-shaped config under the same budget.

### Relation to cycles above
- Cycles 1–23 focus on **bitrate vs perceptual quality** (PESQ, architecture families). This cycle is **compute vs training objective proxy** on MLX: useful for **ranking cheap knobs** (STFT layout, width, batch) before long GPU/PyTorch runs.
- **Not** a claim about PESQ or final codec ranking — only about the **combined reconstruction loss** used in `train_mlx`.

### VERDICT: auxiliary frontier
- Best logged row: **fast STFT + `latent_dim` 384 + `lambda_stft` 0.30** (`aa2e494` in `results.tsv`; prior best without STFT tweak was `af40b90`). **`latent_dim` 448** under the same recipe was worse on the proxy.
- **NEXT**: Optional — try **`lambda_stft` ∈ {0.28, 0.32}** around 0.30; long runs on real `data/` with spectrograms / listening, not only `val_bpb`.
