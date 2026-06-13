"""Encoder, RVQ, decoder, and full codec (MLX)."""
from __future__ import annotations

import math
from statistics import NormalDist

import mlx.core as mx
import mlx.nn as nn

from ..config import Config, effective_codebook_sizes
from .. import streaming as streaming_mod
from .losses import entropy_from_logits, marginal_code_entropy_from_dist


def _argmin_lower_int32(dist: mx.array) -> mx.array:
    """Deterministic argmin: exact ties resolve to the lowest code index."""
    k = int(dist.shape[-1])
    inf = mx.array(float("inf"), dtype=dist.dtype)
    dist = mx.where(mx.isfinite(dist), dist, inf)
    ranks = mx.arange(k, dtype=mx.int32)
    sentinel = mx.full((k,), k, dtype=mx.int32)
    winners = mx.where(dist == mx.min(dist, axis=-1, keepdims=True), ranks, sentinel)
    return mx.min(winners, axis=-1).astype(mx.int32)


def _enc_conv(cfg: Config, cin: int, cout: int, stride: int) -> nn.Module:
    if cfg.causal:
        return streaming_mod.CausalConv1d(cin, cout, 7, stride=stride)
    return nn.Conv1d(cin, cout, 7, stride=stride, padding=3)


def _same_conv(cfg: Config, cin: int, cout: int, kernel: int, *, dilation: int = 1) -> nn.Module:
    if cfg.causal:
        return streaming_mod.CausalConv1d(cin, cout, kernel, stride=1)
    return nn.Conv1d(cin, cout, kernel, stride=1, padding=(kernel // 2) * dilation, dilation=dilation)


def _act_module(cfg: Config, ch: int, activation: str | None = None) -> nn.Module:
    a = (activation or cfg.activation or "gelu").strip().lower()
    if a in ("snake", "snake_beta"):
        return SnakeBeta(ch)
    return nn.GELU()


def _decoder_activation(cfg: Config) -> str:
    return (getattr(cfg, "decoder_activation", None) or cfg.activation or "gelu").strip().lower()


def _sigmoid(x: mx.array) -> mx.array:
    return 1.0 / (1.0 + mx.exp(-x))


class SnakeBeta(nn.Module):
    """DAC-style periodic activation: ``x + (1/a)·sin²(a·x)`` with learnable ``a`` per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = max(1, int(channels))
        self.log_alpha = mx.zeros((self.channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        a = mx.exp(mx.clip(self.log_alpha, -10.0, 10.0))[None, None, :]
        return x + (1.0 / (a + 1e-8)) * (mx.sin(a * x) ** 2)


class UpsampleRepeatConv(nn.Module):
    """2× time via sample repetition + smoothing Conv1d (NLC)."""

    def __init__(self, cin: int, cout: int, k: int = 7):
        super().__init__()
        pad = max(0, k // 2)
        self.conv = nn.Conv1d(cin, cout, k, stride=1, padding=pad)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(x, 2, axis=1)
        return self.conv(x)


class Encoder(nn.Module):
    """Downsample by 2^len(strides) along time. Optional causal convs + Snake/GELU."""

    def __init__(self, cfg: Config):
        super().__init__()
        ch = (1,) + cfg.enc_channels
        n1 = max(0, int(cfg.stride1_blocks_per_scale))
        self.layers = []
        for i in range(len(cfg.enc_channels)):
            for _ in range(n1):
                self.layers.append(_enc_conv(cfg, ch[i], ch[i], 1))
                self.layers.append(_act_module(cfg, ch[i]))
            self.layers.append(_enc_conv(cfg, ch[i], ch[i + 1], 2))
            self.layers.append(_act_module(cfg, ch[i + 1]))
        self.out = (
            streaming_mod.CausalConv1d(cfg.enc_channels[-1], cfg.latent_dim, 3, stride=1)
            if cfg.causal
            else nn.Conv1d(cfg.enc_channels[-1], cfg.latent_dim, 3, padding=1)
        )
        for i, layer in enumerate(self.layers):
            setattr(self, f"_b{i}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


class VectorQuantizerStage(nn.Module):
    """One RVQ stage: optional factorized D→d→D projection + cosine or Euclidean VQ."""

    def __init__(self, cfg: Config, num_embeddings: int):
        super().__init__()
        self.cfg = cfg
        self.num_embeddings = int(num_embeddings)
        d = int(cfg.latent_dim)
        cd = int(cfg.rvq_code_dim) if int(getattr(cfg, "rvq_code_dim", 0) or 0) > 0 else d
        self.use_factor = cd < d
        if self.use_factor:
            self.in_proj = nn.Linear(d, cd)
            self.out_proj = nn.Linear(cd, d)
            emb_dim = cd
        else:
            self.in_proj = None
            self.out_proj = None
            emb_dim = d
        self.embedding = nn.Embedding(self.num_embeddings, emb_dim)
        self.beta = cfg.vq_commitment

    def __call__(self, residual: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if self.in_proj is not None:
            r = self.in_proj(residual)
        else:
            r = residual
        cb = self.embedding["weight"]
        if self.cfg.vq_cosine:
            z2 = mx.sum(r * r, axis=-1, keepdims=True)
            e2 = mx.sum(cb * cb, axis=-1, keepdims=True)
            zn = r / mx.sqrt(z2 + 1e-8)
            eb = cb / mx.sqrt(e2 + 1e-8)
            dots = zn @ mx.transpose(eb)
            dist = 2.0 - 2.0 * dots
            idx = _argmin_lower_int32(dist)
            rn = mx.sqrt(z2 + 1e-8)
            flat = mx.reshape(idx, (-1,))
            z_q_dir = mx.reshape(mx.take(eb, flat, axis=0), r.shape)
            z_q_low = z_q_dir * rn
        else:
            z2 = mx.sum(r * r, axis=-1, keepdims=True)
            e2 = mx.sum(cb * cb, axis=-1)
            dots = r @ mx.transpose(cb)
            dist = z2 + e2 - 2.0 * dots
            idx = _argmin_lower_int32(dist)
            z_q_low = self.embedding(idx)
        if self.out_proj is not None:
            z_q = self.out_proj(z_q_low)
        else:
            z_q = z_q_low
        z_st = residual + mx.stop_gradient(z_q - residual)
        commit = mx.mean((mx.stop_gradient(z_q) - residual) ** 2)
        codebook = mx.mean((z_q - mx.stop_gradient(residual)) ** 2)
        vq_loss = self.beta * commit + codebook
        if self.num_embeddings == 2 and float(getattr(self.cfg, "lambda_binary_usage", 0.0) or 0.0) > 0:
            tau = max(float(getattr(self.cfg, "binary_usage_tau", 0.04) or 0.04), 1e-6)
            soft = mx.softmax(-dist / tau, axis=-1)
            codes = mx.reshape(mx.arange(self.num_embeddings, dtype=mx.int32), (1, 1, self.num_embeddings))
            hard = (mx.expand_dims(idx, -1) == codes).astype(soft.dtype)
            usage = mx.reshape(hard + soft - mx.stop_gradient(soft), (-1, self.num_embeddings))
            probs = mx.mean(usage, axis=0)
            target = mx.ones((self.num_embeddings,), dtype=soft.dtype) / float(self.num_embeddings)
            usage_loss = mx.sum((probs - target) ** 2)
            vq_loss = vq_loss + float(self.cfg.lambda_binary_usage) * usage_loss
        return z_st, vq_loss, dist, idx


class ResidualVectorQuantizer(nn.Module):
    """Residual sum of ``n_codebooks`` vector quantizers (SoundStream / EnCodec style)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_q = cfg.n_codebooks
        sizes = effective_codebook_sizes(cfg)
        for i in range(self.n_q):
            setattr(self, f"q{i}", VectorQuantizerStage(cfg, sizes[i]))

    def __call__(
        self, z: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array]]:
        quantized = mx.zeros_like(z)
        total_vq = mx.array(0.0)
        total_ent = mx.array(0.0)
        total_marg = mx.array(0.0)
        indices: list[mx.array] = []
        for i in range(self.n_q):
            stage: VectorQuantizerStage = getattr(self, f"q{i}")
            residual = z - quantized
            z_i, lvq, dist, idx = stage(residual)
            quantized = quantized + z_i
            total_vq = total_vq + lvq
            indices.append(idx)
            if self.cfg.lambda_entropy > 0:
                total_ent = total_ent + entropy_from_logits(-dist)
            if self.cfg.lambda_marginal > 0:
                total_marg = total_marg + marginal_code_entropy_from_dist(dist, self.cfg.marginal_tau)
        if self.cfg.lambda_entropy > 0 and self.n_q > 0:
            total_ent = total_ent / float(self.n_q)
        if self.cfg.lambda_marginal > 0 and self.n_q > 0:
            total_marg = total_marg / float(self.n_q)
        return quantized, total_vq, total_ent, total_marg, indices


class TurboQuantBottleneck(nn.Module):
    """Fixed TurboQuant-style scalar latent quantizer with optional D→d→D projection."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.bits = int(getattr(cfg, "turboquant_bits", 2))
        if self.bits not in (2, 4):
            raise ValueError(f"turboquant_bits must be 2 or 4, got {self.bits}")
        d = int(cfg.latent_dim)
        cd = int(getattr(cfg, "turboquant_code_dim", 0) or 0)
        if cd <= 0:
            cd = d
        if cd > d:
            raise ValueError(f"turboquant_code_dim must be <= latent_dim ({d}), got {cd}")
        self.code_dim = int(cd)
        self.use_factor = self.code_dim < d
        if self.use_factor:
            self.in_proj = nn.Linear(d, self.code_dim)
            self.out_proj = nn.Linear(self.code_dim, d)
        else:
            self.in_proj = None
            self.out_proj = None
        self.beta = float(cfg.vq_commitment)
        self.num_levels = 1 << self.bits
        nd = NormalDist()
        inv_std = math.sqrt(float(max(1, self.code_dim)))
        # Quantile-spaced normal centroids approximate the high-dimensional unit-sphere marginal.
        cents = [nd.inv_cdf((i + 0.5) / self.num_levels) / inv_std for i in range(self.num_levels)]
        bnds = [nd.inv_cdf(i / self.num_levels) / inv_std for i in range(1, self.num_levels)]
        self.centroids = tuple(float(x) for x in cents)
        self.boundaries = tuple(float(x) for x in bnds)

    def __call__(
        self, z: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array]]:
        r = self.in_proj(z) if self.in_proj is not None else z
        norm = mx.sqrt(mx.sum(r * r, axis=-1, keepdims=True) + 1e-8)
        unit = r / norm
        idx = mx.zeros(unit.shape, dtype=mx.int32)
        for b in self.boundaries:
            idx = idx + (unit > float(b)).astype(mx.int32)
        centroids = mx.array(self.centroids, dtype=unit.dtype)
        flat = mx.reshape(idx, (-1,))
        q_unit = mx.reshape(mx.take(centroids, flat, axis=0), unit.shape)
        q_norm = mx.sqrt(mx.sum(q_unit * q_unit, axis=-1, keepdims=True) + 1e-8)
        q_low = q_unit * (norm / q_norm)
        q = self.out_proj(q_low) if self.out_proj is not None else q_low
        z_st = z + mx.stop_gradient(q - z)
        vq_loss = self.beta * mx.mean((mx.stop_gradient(q) - z) ** 2)

        levels = mx.reshape(centroids, (1, 1, 1, self.num_levels))
        dist = (mx.expand_dims(unit, -1) - levels) ** 2
        ent = mx.array(0.0, dtype=mx.float32)
        marg = mx.array(0.0, dtype=mx.float32)
        if self.cfg.lambda_entropy > 0:
            ent = entropy_from_logits(-dist)
        if self.cfg.lambda_marginal > 0:
            tau = max(float(getattr(self.cfg, "marginal_tau", 0.04) or 0.04), 1e-6)
            logits = mx.clip(-dist / tau, -50.0, 50.0)
            m = mx.max(logits, axis=-1, keepdims=True)
            ex = mx.exp(logits - m)
            p = ex / (mx.sum(ex, axis=-1, keepdims=True) + 1e-8)
            avg = mx.mean(p, axis=(0, 1))
            h = -mx.sum(avg * mx.log(avg + 1e-8), axis=-1)
            marg = mx.mean(h)
        return z_st, vq_loss, ent, marg, [idx.astype(mx.int32)]


class DecoderRefineStack(nn.Module):
    """Small residual temporal post-filter on final decoder features."""

    def __init__(self, cfg: Config, channels: int, depth: int, gain: float):
        super().__init__()
        self.depth = max(0, int(depth))
        self.gain = float(gain)
        self.layers = []
        act = _decoder_activation(cfg)
        for i in range(self.depth):
            dilation = 2 ** (i % 4)
            self.layers.append(_same_conv(cfg, channels, channels, 7, dilation=dilation))
            self.layers.append(_act_module(cfg, channels, act))
        for i, layer in enumerate(self.layers):
            setattr(self, f"_refine{i}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        if self.depth <= 0 or self.gain == 0.0:
            return x
        h = x
        for layer in self.layers:
            h = layer(h)
        return x + self.gain * h


class DecoderBandHead(nn.Module):
    """Residual waveform head with a distinct kernel size for multi-band synthesis pressure."""

    def __init__(self, cfg: Config, channels: int, depth: int, kernel: int):
        super().__init__()
        self.layers = []
        act = _decoder_activation(cfg)
        for i in range(max(0, int(depth))):
            self.layers.append(_same_conv(cfg, channels, channels, kernel))
            self.layers.append(_act_module(cfg, channels, act))
        self.out = _same_conv(cfg, channels, 1, kernel)
        for i, layer in enumerate(self.layers):
            setattr(self, f"_band{i}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.out(h)


class HarmonicSource(nn.Module):
    """Learned differentiable harmonic excitation from decoder features."""

    def __init__(self, cfg: Config, channels: int):
        super().__init__()
        self.sample_rate = float(cfg.sample_rate)
        self.harmonics = max(1, int(cfg.harmonic_harmonics))
        self.f0_min = float(cfg.harmonic_f0_min)
        self.f0_max = float(cfg.harmonic_f0_max)
        self.f0_head = _same_conv(cfg, channels, 1, 7)
        self.amp_head = _same_conv(cfg, channels, self.harmonics, 7)

    def __call__(self, x: mx.array) -> mx.array:
        f0_unit = _sigmoid(self.f0_head(x))
        f0 = self.f0_min + (self.f0_max - self.f0_min) * f0_unit
        phase = mx.cumsum(f0 * (2.0 * math.pi / self.sample_rate), axis=1)
        harmonics = mx.reshape(mx.arange(1, self.harmonics + 1, dtype=x.dtype), (1, 1, self.harmonics))
        amps = _sigmoid(self.amp_head(x)) / float(self.harmonics)
        return mx.sum(amps * mx.sin(phase * harmonics), axis=-1, keepdims=True)


def _lava_highpass_residual(x: mx.array) -> mx.array:
    """Cheap differentiable high-pass residual for LavaSR-style low/high-band merge."""
    left = mx.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    right = mx.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    return x - (0.5 * (left + right))


def _repeat_time_to(x: mx.array, target_len: int) -> mx.array:
    """Repeat low-rate controls on the time axis to match a decoder waveform grid."""
    t = int(x.shape[1])
    n = max(1, int(target_len))
    if t >= n:
        return x[:, :n, :]
    repeats = max(1, (n + t - 1) // t)
    return mx.repeat(x, repeats, axis=1)[:, :n, :]


class PostLavaSRRefiner(nn.Module):
    """Small waveform refiner applied after the decoder, inspired by LavaSR's LR merge."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.depth = max(0, int(cfg.post_lavasr_depth))
        self.gain = float(cfg.post_lavasr_gain)
        self.highpass = bool(cfg.post_lavasr_highpass)
        ch = max(4, int(cfg.post_lavasr_channels))
        k = max(3, int(cfg.post_lavasr_kernel))
        if k % 2 == 0:
            k += 1
        act = _decoder_activation(cfg)
        self.in_proj = _same_conv(cfg, 1, ch, k)
        self.layers = []
        for i in range(self.depth):
            dilation = 2 ** (i % 4)
            self.layers.append(_same_conv(cfg, ch, ch, k, dilation=dilation))
            self.layers.append(_act_module(cfg, ch, act))
        self.out = _same_conv(cfg, ch, 1, k)
        for i, layer in enumerate(self.layers):
            setattr(self, f"_post_lava{i}", layer)

    def __call__(self, y: mx.array) -> mx.array:
        h = self.in_proj(y)
        for layer in self.layers:
            h = layer(h)
        residual = self.out(h)
        if self.highpass:
            residual = _lava_highpass_residual(residual)
        return mx.tanh(y + self.gain * residual)


class SpeechControlHead(nn.Module):
    """Predict bounded speech gates for a conservative decoder residual path."""

    names = ("energy", "voicing", "harmonicity", "hf_gate", "noise_gate")

    def __init__(self, cfg: Config, channels: int):
        super().__init__()
        self.depth = max(0, int(cfg.speech_control_depth))
        hidden = max(4, int(cfg.speech_control_channels))
        act = _decoder_activation(cfg)
        self.in_proj = _same_conv(cfg, channels, hidden, 7)
        self.layers = []
        for i in range(self.depth):
            dilation = 2 ** (i % 4)
            self.layers.append(_same_conv(cfg, hidden, hidden, 7, dilation=dilation))
            self.layers.append(_act_module(cfg, hidden, act))
        self.out = _same_conv(cfg, hidden, len(self.names), 7)
        if hasattr(self.out, "bias"):
            bias = mx.zeros((len(self.names),), dtype=mx.float32)
            # Start new checkpoints conservative: HF/noise residual gates are nearly closed.
            bias = bias.at[3].add(-4.0)
            bias = bias.at[4].add(-2.0)
            self.out.bias = bias
        for i, layer in enumerate(self.layers):
            setattr(self, f"_speech_ctl{i}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h)
        return _sigmoid(self.out(h))


class Decoder(nn.Module):
    """Upsample toward full length: ConvTranspose or repeat×2+Conv (anti-aliased ZOH)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        c = list(reversed(cfg.enc_channels))
        d = cfg.latent_dim
        n1 = max(0, int(cfg.stride1_blocks_per_scale))
        use_repeat = (cfg.decoder_upsample or "transpose").lower() == "repeat_conv" or bool(cfg.causal)
        self.layers = []
        act = _decoder_activation(cfg)
        if use_repeat:
            self.layers.append(UpsampleRepeatConv(d, c[0]))
        else:
            self.layers.append(nn.ConvTranspose1d(d, c[0], 7, stride=2, padding=3))
        self.layers.append(_act_module(cfg, c[0], act))
        for _ in range(n1):
            self.layers.append(
                streaming_mod.CausalConv1d(c[0], c[0], 7, stride=1)
                if cfg.causal
                else nn.Conv1d(c[0], c[0], 7, stride=1, padding=3)
            )
            self.layers.append(_act_module(cfg, c[0], act))
        for i in range(len(c) - 1):
            if use_repeat:
                self.layers.append(UpsampleRepeatConv(c[i], c[i + 1]))
            else:
                self.layers.append(nn.ConvTranspose1d(c[i], c[i + 1], 7, stride=2, padding=3))
            self.layers.append(_act_module(cfg, c[i + 1], act))
            for _ in range(n1):
                self.layers.append(
                    streaming_mod.CausalConv1d(c[i + 1], c[i + 1], 7, stride=1)
                    if cfg.causal
                    else nn.Conv1d(c[i + 1], c[i + 1], 7, stride=1, padding=3)
                )
                self.layers.append(_act_module(cfg, c[i + 1], act))
        self.out = (
            streaming_mod.CausalConv1d(c[-1], 1, 7, stride=1)
            if cfg.causal
            else nn.Conv1d(c[-1], 1, 7, padding=3)
        )
        self.refine = (
            DecoderRefineStack(cfg, c[-1], cfg.decoder_refine_depth, cfg.decoder_refine_gain)
            if int(cfg.decoder_refine_depth) > 0
            else None
        )
        self.band_gain = float(cfg.decoder_band_gain)
        self.band_heads = []
        for i in range(max(0, int(cfg.decoder_band_heads) - 1)):
            kernel = 7 + 8 * (i + 1)
            self.band_heads.append(DecoderBandHead(cfg, c[-1], cfg.decoder_band_depth, kernel))
        for i, head in enumerate(self.band_heads):
            setattr(self, f"_band_head{i}", head)
        self.harmonic_gain = float(cfg.harmonic_amp)
        self.harmonic_source = HarmonicSource(cfg, c[-1]) if bool(cfg.harmonic_source) else None
        self.speech_control_gain = float(cfg.speech_control_gain)
        self.speech_residual_gain = float(cfg.speech_residual_gain)
        self.speech_hf_gate_floor = float(cfg.speech_hf_gate_floor)
        self.speech_control = SpeechControlHead(cfg, cfg.latent_dim) if int(cfg.speech_control_depth) > 0 else None
        self.speech_residual = _same_conv(cfg, c[-1], 1, 7) if self.speech_control is not None else None
        if self.speech_residual is not None:
            self.speech_residual.weight = mx.zeros_like(self.speech_residual.weight)
            if hasattr(self.speech_residual, "bias"):
                self.speech_residual.bias = mx.zeros_like(self.speech_residual.bias)
        object.__setattr__(self, "_last_speech_control_stats", {})
        self.post_lavasr = PostLavaSRRefiner(cfg) if int(cfg.post_lavasr_depth) > 0 else None
        for i, layer in enumerate(self.layers):
            setattr(self, f"_d{i}", layer)

    def speech_control_stats(self) -> dict[str, mx.array]:
        return dict(getattr(self, "_last_speech_control_stats", {}))

    def __call__(self, z: mx.array, target_len: int) -> mx.array:
        control_src = self.speech_control(z) if self.speech_control is not None else None
        x = z
        for layer in self.layers:
            x = layer(x)
        features = self.refine(x) if self.refine is not None else x
        y = self.out(features)
        if control_src is not None:
            controls = _repeat_time_to(control_src, int(y.shape[1]))
            energy = controls[:, :, 0:1]
            voicing = controls[:, :, 1:2]
            harmonicity = controls[:, :, 2:3]
            hf_gate = controls[:, :, 3:4]
            noise_gate = controls[:, :, 4:5]
            floor = max(0.0, min(1.0, self.speech_hf_gate_floor))
            gate = floor + (1.0 - floor) * hf_gate
            speech_presence = 0.25 + 0.75 * mx.maximum(voicing, harmonicity)
            residual_gate = mx.clip(self.speech_control_gain * gate * speech_presence, 0.0, 1.0)
            if self.speech_residual is not None and self.speech_residual_gain != 0.0:
                residual = _lava_highpass_residual(self.speech_residual(features))
                residual_gate = residual_gate * (0.5 + 0.5 * noise_gate)
                y = y + self.speech_residual_gain * residual_gate * residual
            object.__setattr__(
                self,
                "_last_speech_control_stats",
                {
                    "energy": mx.stop_gradient(mx.mean(energy)),
                    "voicing": mx.stop_gradient(mx.mean(voicing)),
                    "harmonicity": mx.stop_gradient(mx.mean(harmonicity)),
                    "hf_gate": mx.stop_gradient(mx.mean(hf_gate)),
                    "noise_gate": mx.stop_gradient(mx.mean(noise_gate)),
                    "residual_gate": mx.stop_gradient(mx.mean(residual_gate)),
                },
            )
        else:
            object.__setattr__(self, "_last_speech_control_stats", {})
        if self.band_heads and self.band_gain != 0.0:
            extra = mx.zeros_like(y)
            for head in self.band_heads:
                extra = extra + head(features)
            y = y + self.band_gain * (extra / float(len(self.band_heads)))
        if self.harmonic_source is not None and self.harmonic_gain != 0.0:
            y = y + self.harmonic_gain * self.harmonic_source(features)
        t = y.shape[1]
        if t >= target_len:
            y = y[:, :target_len, :]
        else:
            pad = target_len - t
            y = mx.pad(y, [(0, 0), (0, pad), (0, 0)])
        y = mx.tanh(y)
        if self.post_lavasr is not None:
            y = self.post_lavasr(y)
        return y


class LuxVocosFeatureDecoder(nn.Module):
    """Decode bottleneck latents to LuxTTS/Vocos log-mel frames instead of waveform samples."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.depth = max(0, int(cfg.latent_temporal_post_depth))
        self.feature_dim = max(1, int(getattr(cfg, "lux_vocos_feature_dim", 100)))
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.layers = []
        act = _decoder_activation(cfg)
        for i in range(max(0, int(cfg.decoder_refine_depth))):
            dilation = 2 ** (i % 4)
            self.layers.append(_same_conv(cfg, cfg.latent_dim, cfg.latent_dim, 3, dilation=dilation))
            self.layers.append(_act_module(cfg, cfg.latent_dim, act))
        self.out = nn.Linear(cfg.latent_dim, self.feature_dim)
        for i, layer in enumerate(self.layers):
            setattr(self, f"_lux_feat{i}", layer)

    def speech_control_stats(self) -> dict[str, mx.array]:
        return {}

    def __call__(self, z: mx.array, target_len: int) -> mx.array:
        h = self.in_proj(z)
        for layer in self.layers:
            h = layer(h)
        y = self.out(h)
        t = int(y.shape[1])
        n = max(1, int(target_len))
        if t >= n:
            return y[:, :n, :]
        return mx.pad(y, [(0, 0), (0, n - t), (0, 0)])


class LatentTemporalStack(nn.Module):
    """Residual dilated Conv1d stack on [B, L, C] latents (temporal context)."""

    def __init__(self, dim: int, depth: int, *, activation: str = "snake_beta"):
        super().__init__()
        self.dim = int(dim)
        self.depth = max(0, int(depth))
        self.layers = []
        act = (activation or "gelu").strip().lower()
        for i in range(self.depth):
            d = 2 ** (i % 5)  # 1,2,4,8,16,...
            # k=3, stride=1: L_out = L + 2*pad - d*(k-1) = L + 2*pad - 2*d → pad=d preserves length
            pad = d
            self.layers.append(nn.Conv1d(self.dim, self.dim, 3, stride=1, padding=pad, dilation=d))
            self.layers.append(SnakeBeta(self.dim) if act in ("snake", "snake_beta") else nn.GELU())
        for i, layer in enumerate(self.layers):
            setattr(self, f"_lt{i}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        if self.depth <= 0:
            return x
        h = x
        i = 0
        while i < len(self.layers):
            h = self.layers[i](h)
            i += 1
            if i < len(self.layers):
                h = self.layers[i](h)
                i += 1
        return x + h


class LatentSelfAttentionStack(nn.Module):
    """Transformer encoder blocks on [B, L, C] latents."""

    def __init__(self, dim: int, depth: int, heads: int, *, causal: bool = False):
        super().__init__()
        self.depth = max(0, int(depth))
        self.causal = bool(causal)
        self.layers = [
            nn.TransformerEncoderLayer(dim, heads, mlp_dims=max(dim, dim * 2), dropout=0.0)
            for _ in range(self.depth)
        ]
        for i, layer in enumerate(self.layers):
            setattr(self, f"_sa{i}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        if self.depth <= 0:
            return x
        mask = None
        if self.causal:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MLXCodec(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        quantizer = (getattr(cfg, "quantizer", "rvq") or "rvq").strip().lower()
        if quantizer == "turboquant":
            self.rvq = TurboQuantBottleneck(cfg)
        elif quantizer == "rvq":
            self.rvq = ResidualVectorQuantizer(cfg)
        else:
            raise ValueError(f"unknown quantizer {cfg.quantizer!r}")
        backend = (getattr(cfg, "decoder_backend", "waveform") or "waveform").strip().lower()
        if backend == "lux_vocos":
            self.decoder = LuxVocosFeatureDecoder(cfg)
        elif backend == "waveform":
            self.decoder = Decoder(cfg)
        else:
            raise ValueError(f"unknown decoder_backend {cfg.decoder_backend!r}")
        self.pre_vq_ln = nn.LayerNorm(cfg.latent_dim) if cfg.pre_vq_layernorm else None
        self.latent_pre = (
            LatentTemporalStack(cfg.latent_dim, cfg.latent_temporal_depth, activation=cfg.activation)
            if cfg.latent_temporal_depth > 0
            else None
        )
        self.self_attn_pre = (
            LatentSelfAttentionStack(
                cfg.latent_dim,
                cfg.self_attention_depth,
                cfg.self_attention_heads,
                causal=cfg.causal,
            )
            if cfg.self_attention_depth > 0
            else None
        )
        self.latent_post = (
            LatentTemporalStack(cfg.latent_dim, cfg.latent_temporal_post_depth, activation=cfg.activation)
            if cfg.latent_temporal_post_depth > 0
            else None
        )
        self.self_attn_post = (
            LatentSelfAttentionStack(
                cfg.latent_dim,
                cfg.self_attention_post_depth,
                cfg.self_attention_heads,
                causal=cfg.causal,
            )
            if cfg.self_attention_post_depth > 0
            else None
        )
        self.semantic_head_norm = (
            nn.LayerNorm(cfg.latent_dim) if float(getattr(cfg, "lambda_latent_semantic", 0.0)) > 0 else None
        )
        self.semantic_head = (
            nn.Linear(cfg.latent_dim, int(cfg.latent_semantic_dim))
            if float(getattr(cfg, "lambda_latent_semantic", 0.0)) > 0
            else None
        )

    def latent_before_rvq(self, x: mx.array) -> mx.array:
        """Encoder output after optional LayerNorm + pre-RVQ temporal stack (matches RVQ input)."""
        z = self.encoder(x)
        if self.pre_vq_ln is not None:
            z = self.pre_vq_ln(z)
        if self.latent_pre is not None:
            z = self.latent_pre(z)
        if self.self_attn_pre is not None:
            z = self.self_attn_pre(z)
        return z

    def forward_full(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None]:
        """Returns (recon, vq_loss, ent_pos, marginal_ent, indices or None)."""
        y, vq_loss, ent_pos, marg_ent, indices, _ = self.forward_full_with_latent(x)
        return y, vq_loss, ent_pos, marg_ent, indices

    def forward_full_with_latent(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None, mx.array]:
        """Returns reconstruction plus the raw bottleneck latent used before decoder post-processing."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        backend = (getattr(self.cfg, "decoder_backend", "waveform") or "waveform").strip().lower()
        out_len = int(z.shape[1]) if backend == "lux_vocos" else int(tlen)
        if self.cfg.ae_only:
            y = self.decoder(z, out_len)
            return y, mx.array(0.0), mx.array(0.0), mx.array(0.0), None, z
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        z_bottleneck = z_q
        z_dec = z_q
        if self.latent_post is not None:
            z_dec = self.latent_post(z_dec)
        if self.self_attn_post is not None:
            z_dec = self.self_attn_post(z_dec)
        y = self.decoder(z_dec, out_len)
        return y, vq_loss, ent_pos, marg_ent, indices, z_bottleneck

    def semantic_from_latent(self, z_q: mx.array) -> mx.array:
        """Auxiliary HuBERT/ContentVec prediction head for bottleneck supervision."""
        if self.semantic_head is None:
            raise RuntimeError("semantic latent head is disabled; set lambda_latent_semantic > 0")
        h = self.semantic_head_norm(z_q) if self.semantic_head_norm is not None else z_q
        return self.semantic_head(h)

    def speech_control_stats(self) -> dict[str, mx.array]:
        return self.decoder.speech_control_stats()

    def forward_reconstruction_only(self, x: mx.array) -> mx.array:
        """Reconstruction ``ŷ`` only (same graph as ``forward_full``)."""
        return self.forward_full(x)[0]

    def encode_decode_streaming_stub(self, x: mx.array) -> mx.array:
        """Chunked/streaming API stub: non-stateful full-chunk pass (causal = left-pad only within chunk)."""
        return self.forward_reconstruction_only(x)
