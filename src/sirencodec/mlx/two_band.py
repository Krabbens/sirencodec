"""Joint low/high-band MLX codec with a fixed complementary FIR split."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn

from ..config import Config, encoder_time_stride
from .codec import Encoder, MLXCodec, ResidualVectorQuantizer


DEFAULT_HIGH_CHANNELS: tuple[int, ...] = (12, 16, 24, 32, 48, 64, 96, 128)


@mx.custom_function
def _guard_high_latent_gradient(x: mx.array) -> mx.array:
    return x


@_guard_high_latent_gradient.vjp
def _guard_high_latent_gradient_vjp(primals, cotangent, output):
    del primals, output
    grad = mx.nan_to_num(cotangent, nan=0.0, posinf=0.1, neginf=-0.1)
    rms = mx.sqrt(mx.mean(grad * grad) + 1e-8)
    scale = mx.minimum(mx.array(1.0, dtype=grad.dtype), 0.1 / rms)
    return (mx.clip(grad * scale, -0.5, 0.5),)


@mx.custom_function
def _guard_high_wave_gradient(x: mx.array) -> mx.array:
    return x


@_guard_high_wave_gradient.vjp
def _guard_high_wave_gradient_vjp(primals, cotangent, output):
    del primals, output
    grad = mx.nan_to_num(cotangent, nan=0.0, posinf=0.01, neginf=-0.01)
    rms = mx.sqrt(mx.mean(grad * grad) + 1e-10)
    scale = mx.minimum(mx.array(1.0, dtype=grad.dtype), 0.01 / rms)
    return (mx.clip(grad * scale, -0.05, 0.05),)


def design_lowpass_fir(
    sample_rate: int = 16_000,
    cutoff_hz: float = 3_500.0,
    num_taps: int = 127,
) -> tuple[float, ...]:
    """Return an odd-length Hann-windowed sinc low-pass filter."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not (0.0 < cutoff_hz < sample_rate / 2.0):
        raise ValueError("cutoff_hz must be between 0 and Nyquist")
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps must be an odd integer >= 3")

    fc = float(cutoff_hz) / float(sample_rate)
    center = (num_taps - 1) / 2.0
    taps: list[float] = []
    for i in range(num_taps):
        n = float(i) - center
        if n == 0.0:
            sinc = 2.0 * fc
        else:
            sinc = math.sin(2.0 * math.pi * fc * n) / (math.pi * n)
        window = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / float(num_taps - 1))
        taps.append(sinc * window)
    scale = sum(taps)
    if abs(scale) < 1e-12:
        raise ValueError("designed FIR has zero DC gain")
    return tuple(float(x / scale) for x in taps)


def _reflect_pad_time(x: mx.array, pad: int) -> mx.array:
    """Reflection-pad NLC audio along time, including support for pad >= length."""
    if pad <= 0:
        return x
    length = int(x.shape[1])
    if length <= 1:
        return mx.pad(x, [(0, 0), (pad, pad), (0, 0)], mode="edge")
    period = 2 * (length - 1)
    positions = mx.arange(-pad, length + pad, dtype=mx.int32)
    folded = mx.remainder(positions, period)
    indices = mx.where(folded < length, folded, period - folded).astype(mx.int32)
    return mx.take(x, indices, axis=1)


class FixedComplementaryFIR:
    """Non-trainable low/high split with exact complementary analysis."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        cutoff_hz: float = 3_500.0,
        num_taps: int = 127,
    ):
        self.sample_rate = int(sample_rate)
        self.cutoff_hz = float(cutoff_hz)
        self.num_taps = int(num_taps)
        self._taps = design_lowpass_fir(self.sample_rate, self.cutoff_hz, self.num_taps)

    @property
    def group_delay_samples(self) -> int:
        return (self.num_taps - 1) // 2

    def lowpass(self, x: mx.array) -> mx.array:
        if len(x.shape) != 3 or int(x.shape[-1]) != 1:
            raise ValueError(f"expected mono NLC tensor [B,T,1], got {tuple(x.shape)}")
        pad = self.group_delay_samples
        padded = _reflect_pad_time(x, pad)
        weight = mx.reshape(mx.array(self._taps, dtype=x.dtype), (1, self.num_taps, 1))
        return mx.conv1d(padded, weight)

    def split(self, x: mx.array) -> tuple[mx.array, mx.array]:
        low = self.lowpass(x)
        return low, x - low

    def synthesis(self, low: mx.array, high: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if tuple(low.shape) != tuple(high.shape):
            raise ValueError(f"band shapes differ: low={tuple(low.shape)} high={tuple(high.shape)}")
        low_band = self.lowpass(low)
        high_band = high - self.lowpass(high)
        return low_band + high_band, low_band, high_band


@dataclass(frozen=True)
class TwoBandCodecConfig:
    """Architecture settings not represented by the single-band Config."""

    split_hz: float = 3_500.0
    fir_taps: int = 127
    high_channels: tuple[int, ...] = DEFAULT_HIGH_CHANNELS
    high_latent_dim: int = 256
    high_input_gain: float = 1.0
    high_loss_floor: float = 0.1
    low_loss_weight: float = 0.5
    high_loss_weight: float = 0.1
    low_vq_weight: float = 5.0
    high_vq_weight: float = 0.0
    low_marginal_weight: float = 1.0
    high_marginal_weight: float = 0.0
    low_codebooks: int = 2
    high_codebooks: int = 1
    codebook_size: int = 32

    def validate(self, base: Config) -> None:
        if int(base.sample_rate) != 16_000:
            raise ValueError("TwoBandCodec currently requires a 16 kHz base Config")
        if len(self.high_channels) != len(base.enc_channels):
            raise ValueError(
                "high_channels must have the same number of stride-2 stages as base.enc_channels"
            )
        if self.high_latent_dim < 1:
            raise ValueError("high_latent_dim must be positive")
        if self.high_input_gain <= 0:
            raise ValueError("high_input_gain must be positive")
        if self.high_loss_floor <= 0:
            raise ValueError("high_loss_floor must be positive")
        if self.low_loss_weight < 0 or self.high_loss_weight < 0:
            raise ValueError("band loss weights must be non-negative")
        if self.low_vq_weight < 0 or self.high_vq_weight < 0:
            raise ValueError("VQ loss weights must be non-negative")
        if self.low_marginal_weight < 0 or self.high_marginal_weight < 0:
            raise ValueError("marginal loss weights must be non-negative")
        if self.low_codebooks != 2 or self.high_codebooks != 1 or self.codebook_size != 32:
            raise ValueError("the 937.5 bit/s layout requires low=2xK32 and high=1xK32")


def compact_high_channels(base: Config) -> tuple[int, ...]:
    """Derive a small high-band width schedule for non-default smoke configs."""
    if len(base.enc_channels) == len(DEFAULT_HIGH_CHANNELS):
        return DEFAULT_HIGH_CHANNELS
    return tuple(max(4, int(ch) // 2) for ch in base.enc_channels)


def branch_configs(
    base: Config,
    two_band: TwoBandCodecConfig | None = None,
) -> tuple[Config, Config, TwoBandCodecConfig]:
    """Build compatible low/high branch configs from one training config."""
    tb = two_band or TwoBandCodecConfig(high_channels=compact_high_channels(base))
    tb.validate(base)
    low_cfg = replace(
        base,
        quantizer="rvq",
        decoder_backend="waveform",
        n_codebooks=tb.low_codebooks,
        codebook_size=tb.codebook_size,
        codebook_sizes=None,
        vq_cosine=False,
        ae_only=False,
    )
    high_cfg = replace(
        base,
        quantizer="rvq",
        decoder_backend="waveform",
        enc_channels=tb.high_channels,
        latent_dim=tb.high_latent_dim,
        n_codebooks=tb.high_codebooks,
        codebook_size=tb.codebook_size,
        codebook_sizes=None,
        vq_cosine=False,
        ae_only=False,
        pre_vq_layernorm=True,
        self_attention_depth=0,
        self_attention_post_depth=0,
        decoder_refine_depth=0,
        decoder_band_heads=1,
        post_lavasr_depth=0,
        speech_control_depth=0,
        harmonic_source=False,
    )
    return low_cfg, high_cfg, tb


def nominal_two_band_bitrate_bps(
    base: Config,
    two_band: TwoBandCodecConfig | None = None,
) -> float:
    """Nominal index rate for the configured low/high RVQ streams."""
    _, _, tb = branch_configs(base, two_band)
    frame_rate = base.sample_rate / float(encoder_time_stride(base))
    bits_per_frame = (tb.low_codebooks + tb.high_codebooks) * math.log2(
        float(tb.codebook_size)
    )
    return frame_rate * bits_per_frame


def _decode_branch(codec: MLXCodec, z_q: mx.array, target_len: int) -> mx.array:
    z = z_q
    if codec.latent_post is not None:
        z = codec.latent_post(z)
    if codec.self_attn_post is not None:
        z = codec.self_attn_post(z)
    return codec.decoder(z, target_len)


def quantized_from_indices(
    rvq: ResidualVectorQuantizer,
    indices: list[mx.array],
) -> mx.array:
    """Reconstruct an Euclidean RVQ latent using only transmitted indices."""
    if len(indices) != rvq.n_q:
        raise ValueError(f"expected {rvq.n_q} RVQ stages, got {len(indices)}")
    quantized: mx.array | None = None
    expected_shape: tuple[int, ...] | None = None
    for i, idx in enumerate(indices):
        if len(idx.shape) != 2:
            raise ValueError(f"indices[{i}] must have shape [B,T], got {tuple(idx.shape)}")
        shape = tuple(int(x) for x in idx.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"indices[{i}] shape {shape} differs from {expected_shape}")
        stage = getattr(rvq, f"q{i}")
        if bool(stage.cfg.vq_cosine):
            raise ValueError("index-only decoding requires vq_cosine=False")
        z_low = stage.embedding(idx.astype(mx.int32))
        z_stage = stage.out_proj(z_low) if stage.out_proj is not None else z_low
        quantized = z_stage if quantized is None else quantized + z_stage
    if quantized is None:
        raise ValueError("indices cannot be empty")
    return quantized


def stack_indices(indices: list[mx.array]) -> mx.array:
    """Convert a per-stage list of [B,T] arrays into [B,T,Q]."""
    if not indices:
        raise ValueError("indices cannot be empty")
    return mx.stack(indices, axis=-1)


@dataclass
class TwoBandOutput:
    reconstruction: mx.array
    low_reconstruction: mx.array
    high_reconstruction: mx.array
    low_target: mx.array
    high_target: mx.array
    low_vq_loss: mx.array
    high_vq_loss: mx.array
    low_entropy: mx.array
    high_entropy: mx.array
    low_marginal_entropy: mx.array
    high_marginal_entropy: mx.array
    low_indices: list[mx.array]
    high_indices: list[mx.array]
    low_latent: mx.array
    high_latent: mx.array


class TwoBandCodec(nn.Module):
    """Two separately quantized waveform branches with low-band conditioning."""

    architecture_name = "two_band_conditioned_v1"

    def __init__(self, base_cfg: Config, two_band_cfg: TwoBandCodecConfig | None = None):
        super().__init__()
        low_cfg, high_cfg, tb = branch_configs(base_cfg, two_band_cfg)
        self.base_cfg = base_cfg
        self.low_cfg = low_cfg
        self.high_cfg = high_cfg
        self.two_band_cfg = tb
        self.filterbank = FixedComplementaryFIR(
            sample_rate=base_cfg.sample_rate,
            cutoff_hz=tb.split_hz,
            num_taps=tb.fir_taps,
        )
        self.low_codec = MLXCodec(low_cfg)
        self.high_codec = MLXCodec(high_cfg)
        self.low_wave_conditioner = Encoder(high_cfg)
        self.low_latent_projection = nn.Linear(low_cfg.latent_dim, high_cfg.latent_dim)
        self.condition_fusion = nn.Linear(3 * high_cfg.latent_dim, high_cfg.latent_dim)
        self.condition_norm = nn.LayerNorm(high_cfg.latent_dim)

    @property
    def nominal_bitrate_bps(self) -> float:
        return nominal_two_band_bitrate_bps(self.base_cfg, self.two_band_cfg)

    def _condition_high(
        self,
        high_z_q: mx.array,
        low_z_q: mx.array,
        low_reconstruction: mx.array,
    ) -> mx.array:
        low_latent = self.low_latent_projection(mx.stop_gradient(low_z_q))
        low_wave = self.low_wave_conditioner(mx.stop_gradient(low_reconstruction))
        frames = min(
            int(high_z_q.shape[1]),
            int(low_latent.shape[1]),
            int(low_wave.shape[1]),
        )
        high_z_q = high_z_q[:, :frames, :]
        low_latent = low_latent[:, :frames, :]
        low_wave = low_wave[:, :frames, :]
        fused = mx.concatenate([high_z_q, low_latent, low_wave], axis=-1)
        return self.condition_norm(self.condition_fusion(fused))

    def __call__(self, x: mx.array) -> TwoBandOutput:
        target_len = int(x.shape[1])
        low_target, high_target = self.filterbank.split(x)

        low_z = self.low_codec.latent_before_rvq(low_target)
        (
            low_z_q,
            low_vq,
            low_entropy,
            low_marginal,
            low_indices,
        ) = self.low_codec.rvq(low_z)
        low_raw = _decode_branch(self.low_codec, low_z_q, target_len)
        low_reconstruction = self.filterbank.lowpass(low_raw)

        high_z = self.high_codec.latent_before_rvq(
            high_target * float(self.two_band_cfg.high_input_gain)
        )
        high_z = _guard_high_latent_gradient(high_z)
        (
            high_z_q,
            high_vq,
            high_entropy,
            high_marginal,
            high_indices,
        ) = self.high_codec.rvq(high_z)
        high_conditioned = self._condition_high(high_z_q, low_z_q, low_reconstruction)
        high_raw = _decode_branch(self.high_codec, high_conditioned, target_len)
        high_raw = _guard_high_wave_gradient(high_raw)
        reconstruction, low_reconstruction, high_reconstruction = self.filterbank.synthesis(
            low_raw,
            high_raw,
        )

        return TwoBandOutput(
            reconstruction=reconstruction,
            low_reconstruction=low_reconstruction,
            high_reconstruction=high_reconstruction,
            low_target=low_target,
            high_target=high_target,
            low_vq_loss=low_vq,
            high_vq_loss=high_vq,
            low_entropy=low_entropy,
            high_entropy=high_entropy,
            low_marginal_entropy=low_marginal,
            high_marginal_entropy=high_marginal,
            low_indices=low_indices,
            high_indices=high_indices,
            low_latent=low_z_q,
            high_latent=high_z_q,
        )

    def decode_indices(
        self,
        low_indices: list[mx.array],
        high_indices: list[mx.array],
        target_len: int,
    ) -> mx.array:
        """Decode a self-contained pair of low/high RVQ index streams."""
        low_z_q = quantized_from_indices(self.low_codec.rvq, low_indices)
        low_raw = _decode_branch(self.low_codec, low_z_q, int(target_len))
        low_reconstruction = self.filterbank.lowpass(low_raw)
        high_z_q = quantized_from_indices(self.high_codec.rvq, high_indices)
        high_conditioned = self._condition_high(high_z_q, low_z_q, low_reconstruction)
        high_raw = _decode_branch(self.high_codec, high_conditioned, int(target_len))
        reconstruction, _, _ = self.filterbank.synthesis(low_raw, high_raw)
        return reconstruction
