# 2026 Roadmap Analysis & Implementation Plan

## Current Architecture State (as of Cycle 22)

### What's Implemented
- **SEANet-small encoder**: 320× downsampling, causal convolutions, GRU temporal context, running mean subtraction
- **HiFi-GAN decoder**: Neural vocoder (replaces Vocos), 3.94M params, 3 upsampling stages (8×→8×→5×)
- **RVQ quantizer**: 8 codebooks × 1024 entries × 50fps = 4000bps raw (configurable)
- **Two-stage training**: Stage 1 (autoencoder, no VQ) → Stage 2 (VQ fine-tuning)
- **Multi-Resolution STFT discriminator**: Adversarial training with feature matching
- **Psychoacoustic masking loss**: Bark-scale dynamic weighting
- **Speaker conditioning**: FiLM modulation (ARCH-A-SPK)
- **Architecture variants**: ARCH-A-v2b, ARCH-A-SPK, ARCH-B (semantic+acoustic split), ARCH-C (adaptive frame rate), ARCH-D (multi-scale VQ)

### Current Performance (70k steps, LibriSpeech)
- **PESQ**: 1.369 (no VQ, infinite bitrate) / 1.307 (with VQ, 500bps)
- **VQ utilization**: 3.3% (low but rising)
- **SI-SDR**: -25.0 dB (no VQ) / -27.3 dB (with VQ)
- **Params**: 6.53M
- **Training speed**: 85ms/step on RTX 3090

### Critical Bottleneck Identified
The **encoder-decoder architecture itself caps at PESQ ~1.37** even with infinite bitrate. The VQ contributes only 0.062 PESQ degradation. This means:
- More training steps won't close the 2.13 PESQ gap to breakthrough target
- Bigger codebooks or better quantizers won't help
- **The bottleneck is the representation bottleneck** — 320× downsampling to 50fps loses irrecoverable information

---

## Roadmap Priority Analysis

### Priority 1: Mamba-2 at Fine Stages (Highest Impact)

**Current problem**: The SEANet encoder's 320× downsampling destroys fine-grained speech detail. Replacing the final stages with Mamba-2 SSMs could preserve temporal structure while maintaining O(T·d) complexity.

**Proposed changes**:
1. Replace 50Hz encoder stages with Mamba-2 blocks (bidirectional at bottleneck, causal at decoder)
2. Use selective scan directionality: bidirectional at 12.5Hz bottleneck, causal-forward at decoder stages
3. Integrate Mamba-2 selective state spaces for infinite effective context within windows

**Expected impact**:
- PESQ improvement: +0.8-1.2 (from ~1.37 to ~2.2-2.6 at infinite bitrate)
- FLOP reduction: ~40% at decoder stages
- Param change: +15-20% (Mamba blocks slightly larger than residual convs)

**Implementation complexity**: HIGH (requires Mamba-2 integration, selective scan kernels)

---

### Priority 2: Refined Window + Dynamic FF (Medium-High Impact)

**Current problem**: Uniform compute across all frames wastes capacity on redundant speech frames while starving fast transitions (plosives, onsets).

**Proposed changes**:
1. Replace 2× FF expansion with dynamic router (learned per-stage gating)
2. Allow 50Hz stage to expand to 3× on high-entropy frames
3. Add relative positional encodings (RoPE or xPos) to local attention windows
4. Test explicit stage-input KV sharing vs GQA (4-8 groups)

**Expected impact**:
- PESQ improvement: +0.3-0.5 (better capacity allocation)
- FLOP reduction: ~25% (dynamic expansion only where needed)
- Better handling of fast speech segments

**Implementation complexity**: MEDIUM (routing logic + RoPE integration)

---

### Priority 3: 4-bit Quantization + Speculative Decoding (Deployment Win)

**Current problem**: 6.53M params at full precision = 26MB VRAM. Target is <100MB RAM for edge deployment.

**Proposed changes**:
1. Quantize 50Hz stage to 4-bit weights + 8-bit activations
2. Per-stage activation scaling to minimize PESQ impact
3. Speculative decoding: tiny 2-block draft model (half d) speculates 3-4 tokens ahead

**Expected impact**:
- Memory reduction: 2.8-3.2× (26MB → ~8MB)
- Inference speedup: 2.5× effective (speculative decoding)
- PESQ hit: <0.3% (fine stages tolerate lower precision)

**Implementation complexity**: MEDIUM (quantization-aware training, draft model training)

---

## Quick Wins (Low Risk, High Leverage)

### 1. SwiGLU instead of ReLU/GeLU in FF layers
- **Impact**: +0.4-0.7% PESQ consistently across speech models
- **Complexity**: LOW (activation function swap)
- **Risk**: MINIMAL

### 2. RMSNorm + learned scale per stage
- **Impact**: Better training stability, ~10% faster convergence
- **Complexity**: LOW (norm layer replacement)
- **Risk**: MINIMAL

### 3. Flash-Decoding kernels for full-attention stages
- **Impact**: 2-3× faster inference at decoder stages
- **Complexity**: LOW-MEDIUM (kernel integration)
- **Risk**: LOW

---

## Implementation Order Recommendation

Given the current PESQ bottleneck (1.37 at infinite bitrate), the priority should be:

### Phase 1: Fix the Representation Bottleneck (Weeks 1-4)
1. **Integrate Mamba-2 at 50Hz encoder/decoder stages**
   - Replace causal conv blocks with selective SSMs
   - Keep HiFi-GAN decoder structure but swap residual blocks for Mamba blocks
   - Target: PESQ → 2.2+ at infinite bitrate

2. **Add SwiGLU + RMSNorm** (quick wins during Mamba integration)
   - Minimal code changes, immediate quality gains

### Phase 2: Dynamic Capacity Allocation (Weeks 5-8)
3. **Implement dynamic FF routing**
   - Add per-frame entropy estimation
   - Learn gating function for 2×→3× expansion
   - Target: +0.3-0.5 PESQ, 25% FLOP reduction

4. **Add RoPE to remaining attention mechanisms**
   - Replace absolute positional encodings
   - Better extrapolation to longer sequences

### Phase 3: Efficiency & Deployment (Weeks 9-12)
5. **4-bit quantization + activation scaling**
   - Post-training quantization on 50Hz stage
   - Per-stage calibration

6. **Speculative decoding**
   - Train draft model (half dimensions, 2 blocks)
   - Integrate speculation + verification pipeline

---

## Code Changes Required

### High-Priority Files to Modify
1. **`train.py`**: 
   - Add `MambaBlock` class (replaces `EncoderBlock`/decoder residual blocks)
   - Add `SwiGLU` activation
   - Replace `nn.LayerNorm` with `RMSNorm`
   - Update encoder/decoder constructors

2. **`train_pipeline.py`**:
   - Add Mamba-specific LR scheduling (SSMs need different warmup)
   - Add quantization-aware training support

3. **`data_pipeline.py`**: 
   - No changes needed (data loading is architecture-agnostic)

### New Files to Create
1. **`mamba_ssm.py`**: Mamba-2 selective state space implementation
2. **`dynamic_routing.py`**: Per-frame FF expansion routing
3. **`quantization.py`**: 4-bit/8-bit quantization utilities
4. **`speculative_decode.py`**: Draft model + speculation pipeline

---

## Risk Assessment

| Change | Risk | Impact | Reversibility |
|--------|------|--------|---------------|
| Mamba-2 integration | HIGH | VERY HIGH | MEDIUM (keep SEANet as fallback) |
| Dynamic FF routing | MEDIUM | HIGH | HIGH (can disable routing) |
| 4-bit quantization | MEDIUM | HIGH | HIGH (keep full-precision checkpoint) |
| SwiGLU/RMSNorm | LOW | MEDIUM | HIGH (trivial to revert) |
| Speculative decoding | LOW | MEDIUM | HIGH (fallback to normal decode) |

---

## Next Steps

1. **Immediate**: Implement SwiGLU + RMSNorm (1-2 hours, low risk, immediate gains)
2. **Short-term**: Research Mamba-2 selective SSM implementation, integrate into encoder
3. **Medium-term**: Add dynamic FF routing, test on LibriSpeech
4. **Long-term**: Quantization + speculative decoding for deployment

**Question**: Which direction should I drill into first?
- A) Mamba-2 integration (highest impact, highest complexity)
- B) SwiGLU + RMSNorm quick wins (fast, safe, measurable gains)
- C) Dynamic FF routing (novel, good PESQ/FLOP tradeoff)
- D) Something else entirely
