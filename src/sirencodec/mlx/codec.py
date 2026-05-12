"""Encoder, RVQ, decoder, and full codec (MLX)."""
from __future__ import annotations

import math
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from ..config import Config, effective_codebook_sizes
from .. import streaming as streaming_mod
from .losses import apply_band_split_filterbank, entropy_from_logits, marginal_code_entropy_from_dist

_TURBO_ROT_CACHE: dict[tuple[int, int, int], mx.array] = {}


def _turbo_rotation(seed: int, stage_id: int, dim: int) -> mx.array:
    """Deterministic orthogonal mixing matrix used by the TurboQuant-compatible RVQ path."""
    key = (int(seed), int(stage_id), int(dim))
    rot = _TURBO_ROT_CACHE.get(key)
    if rot is not None:
        return rot
    import numpy as np

    rng = np.random.default_rng(int(seed) + 1009 * int(stage_id) + 9176 * int(dim))
    a = rng.standard_normal((int(dim), int(dim))).astype(np.float32)
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r)).astype(np.float32)
    signs[signs == 0] = 1.0
    q = (q * signs[None, :]).astype(np.float32, copy=False)
    rot = mx.array(q)
    _TURBO_ROT_CACHE[key] = rot
    return rot


def _enc_conv(cfg: Config, cin: int, cout: int, stride: int) -> nn.Module:
    if cfg.causal:
        return streaming_mod.CausalConv1d(cin, cout, 7, stride=stride)
    return nn.Conv1d(cin, cout, 7, stride=stride, padding=3)


def _act_from_string(act: str, ch: int) -> nn.Module:
    a = (act or "gelu").strip().lower()
    if a in ("snake", "snake_beta"):
        return SnakeBeta(ch)
    if a == "harmonic_beta":
        return HarmonicBeta(ch)
    if a == "periodic_gelu":
        return PeriodicGELU(ch)
    return nn.GELU()


def _act_module(cfg: Config, ch: int) -> nn.Module:
    return _act_from_string(cfg.activation, ch)


class SnakeBeta(nn.Module):
    """DAC-style periodic activation: ``x + (1/a)·sin²(a·x)`` with learnable ``a`` per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = max(1, int(channels))
        self.log_alpha = mx.zeros((self.channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        a = mx.exp(mx.clip(self.log_alpha, -10.0, 10.0))[None, None, :]
        return x + (1.0 / (a + 1e-8)) * (mx.sin(a * x) ** 2)


class HarmonicBeta(nn.Module):
    """Two-scale periodic residual: ``x + sin²(ax)/a + w·sin²(bx)/b`` with ``b = a·exp(log_ratio)``.

    Extends single-frequency sin² (DAC/Snake-style) with a learnable second ripple scale and mix,
    which fits multi-harmonic audio content better than one global frequency per channel.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = max(1, int(channels))
        self.log_alpha = mx.zeros((self.channels,), dtype=mx.float32)
        self.log_ratio = mx.full((self.channels,), math.log(2.0), dtype=mx.float32)
        self.log_mix = mx.zeros((self.channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        a = mx.exp(mx.clip(self.log_alpha, -10.0, 10.0))[None, None, :]
        r = mx.exp(mx.clip(self.log_ratio, -3.0, 3.0))[None, None, :]
        b = a * r
        w = mx.sigmoid(self.log_mix)[None, None, :]
        t1 = (1.0 / (a + 1e-8)) * (mx.sin(a * x) ** 2)
        t2 = (1.0 / (b + 1e-8)) * (mx.sin(b * x) ** 2)
        return x + t1 + w * t2


class PeriodicGELU(nn.Module):
    """``GELU(x) + sin²(ax)/a`` with learnable ``a`` per channel.

    Unlike DAC SnakeBeta ``x + sin²(ax)/a``, the nonlinear envelope comes from GELU while the
    sin² term keeps a dedicated periodic degree of freedom—often less spectral muffling than
    stacking both effects on the identity path, especially through quantized latents.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = max(1, int(channels))
        self.log_alpha = mx.zeros((self.channels,), dtype=mx.float32)
        self._gelu = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        a = mx.exp(mx.clip(self.log_alpha, -10.0, 10.0))[None, None, :]
        return self._gelu(x) + (1.0 / (a + 1e-8)) * (mx.sin(a * x) ** 2)


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


def _vq_match_loss(diff: mx.array, cfg: Config, ref: mx.array | None = None) -> mx.array:
    if bool(getattr(cfg, "vq_loss_normalize", False)) and ref is not None:
        eps = max(1e-8, float(getattr(cfg, "vq_loss_norm_eps", 1e-4)))
        scale = mx.sqrt(mx.mean(mx.stop_gradient(ref) * mx.stop_gradient(ref), axis=-1, keepdims=True) + eps)
        diff = diff / scale
    mode = str(getattr(cfg, "vq_loss", "mse") or "mse").lower()
    if mode == "huber":
        delta = max(1e-6, float(getattr(cfg, "vq_huber_delta", 1.0)))
        ad = mx.abs(diff)
        quad = mx.minimum(ad, delta)
        lin = ad - quad
        return mx.mean(0.5 * quad * quad / delta + lin)
    return mx.mean(diff * diff)


class VectorQuantizerStage(nn.Module):
    """One RVQ stage: optional factorized D→d→D projection + cosine or Euclidean VQ."""

    def __init__(self, cfg: Config, num_embeddings: int, dim: int | None = None, stage_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.num_embeddings = int(num_embeddings)
        self.stage_id = int(stage_id)
        d = int(dim) if dim is not None else int(cfg.latent_dim)
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
        if self.cfg.turboquant:
            rot = _turbo_rotation(self.cfg.turboquant_seed, self.stage_id, int(r.shape[-1]))
            r_vq = r @ rot
        else:
            rot = None
            r_vq = r
        cb = self.embedding["weight"]
        if self.cfg.vq_cosine:
            z2 = mx.sum(r_vq * r_vq, axis=-1, keepdims=True)
            e2 = mx.sum(cb * cb, axis=-1, keepdims=True)
            zn = r_vq / mx.sqrt(z2 + 1e-8)
            eb = cb / mx.sqrt(e2 + 1e-8)
            dots = zn @ mx.transpose(eb)
            dist = 2.0 - 2.0 * dots
            idx = mx.argmin(dist, axis=-1)
            rn = mx.sqrt(z2 + 1e-8)
            flat = mx.reshape(idx, (-1,))
            z_q_dir = mx.reshape(mx.take(eb, flat, axis=0), r.shape)
            z_q_low = z_q_dir * rn
        else:
            z2 = mx.sum(r_vq * r_vq, axis=-1, keepdims=True)
            e2 = mx.sum(cb * cb, axis=-1)
            dots = r_vq @ mx.transpose(cb)
            dist = z2 + e2 - 2.0 * dots
            idx = mx.argmin(dist, axis=-1)
            z_q_low = self.embedding(idx)
        if rot is not None:
            z_q_low = z_q_low @ mx.transpose(rot)
        if self.out_proj is not None:
            z_q = self.out_proj(z_q_low)
        else:
            z_q = z_q_low
        z_st = residual + mx.stop_gradient(z_q - residual)
        commit = _vq_match_loss(mx.stop_gradient(z_q) - residual, self.cfg, residual)
        codebook = _vq_match_loss(z_q - mx.stop_gradient(residual), self.cfg, residual)
        vq_loss = self.beta * commit + codebook
        return z_st, vq_loss, dist, idx


class ResidualVectorQuantizer(nn.Module):
    """Residual sum of ``n_codebooks`` vector quantizers (SoundStream / EnCodec style)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_q = cfg.n_codebooks
        sizes = effective_codebook_sizes(cfg)
        splits = getattr(cfg, "rvq_band_splits", None)
        self.group_splits = tuple(int(x) for x in splits) if splits else None
        self.group_dims: tuple[int, ...] | None = None
        if self.group_splits:
            n_groups = len(self.group_splits)
            base = int(cfg.latent_dim) // n_groups
            rem = int(cfg.latent_dim) % n_groups
            self.group_dims = tuple(base + (1 if i < rem else 0) for i in range(n_groups))
            stage_i = 0
            for g, (n_stages, dim) in enumerate(zip(self.group_splits, self.group_dims, strict=True)):
                for j in range(n_stages):
                    setattr(self, f"q{g}_{j}", VectorQuantizerStage(cfg, sizes[stage_i], dim=dim, stage_id=stage_i))
                    stage_i += 1
        else:
            for i in range(self.n_q):
                setattr(self, f"q{i}", VectorQuantizerStage(cfg, sizes[i], stage_id=i))

    def _run_chain(
        self, z: mx.array, stages: list[VectorQuantizerStage]
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array]]:
        quantized = mx.zeros_like(z)
        total_vq = mx.array(0.0)
        total_ent = mx.array(0.0)
        total_marg = mx.array(0.0)
        indices: list[mx.array] = []
        for stage in stages:
            residual = z - quantized
            z_i, lvq, dist, idx = stage(residual)
            quantized = quantized + z_i
            total_vq = total_vq + lvq
            indices.append(idx)
            if self.cfg.lambda_entropy > 0:
                total_ent = total_ent + entropy_from_logits(-dist)
            if self.cfg.lambda_marginal > 0:
                total_marg = total_marg + marginal_code_entropy_from_dist(dist, self.cfg.marginal_tau)
        return quantized, total_vq, total_ent, total_marg, indices

    def __call__(
        self, z: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array]]:
        if self.group_splits and self.group_dims:
            pieces = []
            total_vq = mx.array(0.0)
            total_ent = mx.array(0.0)
            total_marg = mx.array(0.0)
            indices: list[mx.array] = []
            c0 = 0
            for g, (n_stages, dim) in enumerate(zip(self.group_splits, self.group_dims, strict=True)):
                c1 = c0 + dim
                stages = [getattr(self, f"q{g}_{j}") for j in range(n_stages)]
                q_g, vq_g, ent_g, marg_g, idx_g = self._run_chain(z[:, :, c0:c1], stages)
                pieces.append(q_g)
                total_vq = total_vq + vq_g
                total_ent = total_ent + ent_g
                total_marg = total_marg + marg_g
                indices.extend(idx_g)
                c0 = c1
            if self.cfg.lambda_entropy > 0 and self.n_q > 0:
                total_ent = total_ent / float(self.n_q)
            if self.cfg.lambda_marginal > 0 and self.n_q > 0:
                total_marg = total_marg / float(self.n_q)
            return mx.concatenate(pieces, axis=2), total_vq, total_ent, total_marg, indices

        stages = [getattr(self, f"q{i}") for i in range(self.n_q)]
        quantized, total_vq, total_ent, total_marg, indices = self._run_chain(z, stages)
        if self.cfg.lambda_entropy > 0 and self.n_q > 0:
            total_ent = total_ent / float(self.n_q)
        if self.cfg.lambda_marginal > 0 and self.n_q > 0:
            total_marg = total_marg / float(self.n_q)
        return quantized, total_vq, total_ent, total_marg, indices

    def _legacy_call(
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


class Decoder(nn.Module):
    """Upsample toward full length: ConvTranspose or repeat×2+Conv (anti-aliased ZOH)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        c = list(reversed(cfg.enc_channels))
        d = cfg.latent_dim
        n1 = max(0, int(cfg.stride1_blocks_per_scale))
        n_refine = max(0, int(getattr(cfg, "decoder_refine_blocks_per_scale", 0)))
        use_repeat = (cfg.decoder_upsample or "transpose").lower() == "repeat_conv" or bool(cfg.causal)
        kt = max(2, int(getattr(cfg, "decoder_transpose_kernel", 7)))
        tp = (kt - 2) // 2 if kt % 2 == 0 else kt // 2
        self.band_split = bool(getattr(cfg, "decoder_band_split", False))
        self.harmonic_source = bool(getattr(cfg, "decoder_harmonic_source", False))
        self.harmonic_mode = str(getattr(cfg, "decoder_harmonic_mode", "additive") or "additive").strip().lower()
        self.band_head_depth = max(0, int(getattr(cfg, "decoder_band_head_depth", 0) or 0))
        self.layers = []
        if use_repeat:
            self.layers.append(UpsampleRepeatConv(d, c[0]))
        else:
            self.layers.append(nn.ConvTranspose1d(d, c[0], kt, stride=2, padding=tp))
        self.layers.append(_act_module(cfg, c[0]))
        for _ in range(n1 + n_refine):
            self.layers.append(
                streaming_mod.CausalConv1d(c[0], c[0], 7, stride=1)
                if cfg.causal
                else nn.Conv1d(c[0], c[0], 7, stride=1, padding=3)
            )
            self.layers.append(_act_module(cfg, c[0]))
        for i in range(len(c) - 1):
            if use_repeat:
                self.layers.append(UpsampleRepeatConv(c[i], c[i + 1]))
            else:
                self.layers.append(nn.ConvTranspose1d(c[i], c[i + 1], kt, stride=2, padding=tp))
            self.layers.append(_act_module(cfg, c[i + 1]))
            for _ in range(n1 + n_refine):
                self.layers.append(
                    streaming_mod.CausalConv1d(c[i + 1], c[i + 1], 7, stride=1)
                    if cfg.causal
                    else nn.Conv1d(c[i + 1], c[i + 1], 7, stride=1, padding=3)
                )
                self.layers.append(_act_module(cfg, c[i + 1]))
        harm_groups = max(0, int(getattr(cfg, "decoder_harmonic_groups", 0)))
        if self.harmonic_source and self.band_split:
            out_ch = (7 + harm_groups) if self.harmonic_mode == "source_filter" else (5 + harm_groups)
        else:
            out_ch = (3 + harm_groups) if self.harmonic_source else (3 if self.band_split else 1)
        self.use_band_heads = self.band_split and not self.harmonic_source and self.band_head_depth > 0
        if self.use_band_heads:
            self.band_heads = []
            for b in range(3):
                head = []
                for j in range(self.band_head_depth):
                    head.append(
                        streaming_mod.CausalConv1d(c[-1], c[-1], 5, stride=1)
                        if cfg.causal
                        else nn.Conv1d(c[-1], c[-1], 5, padding=2)
                    )
                    head.append(_act_module(cfg, c[-1]))
                head.append(
                    streaming_mod.CausalConv1d(c[-1], 1, 7, stride=1)
                    if cfg.causal
                    else nn.Conv1d(c[-1], 1, 7, padding=3)
                )
                self.band_heads.append(head)
                for j, layer in enumerate(head):
                    setattr(self, f"_band_head{b}_{j}", layer)
            self.out = None
        else:
            self.out = (
                streaming_mod.CausalConv1d(c[-1], out_ch, 7, stride=1)
                if cfg.causal
                else nn.Conv1d(c[-1], out_ch, 7, padding=3)
            )
        for i, layer in enumerate(self.layers):
            setattr(self, f"_d{i}", layer)

    def _decode_to_length(self, z: mx.array, target_len: int) -> mx.array:
        x = z
        for layer in self.layers:
            x = layer(x)
        if self.use_band_heads:
            outs = []
            for head in self.band_heads:
                h = x
                for layer in head:
                    h = layer(h)
                outs.append(h)
            x = mx.concatenate(outs, axis=2)
        else:
            x = self.out(x)
        t = x.shape[1]
        if t >= target_len:
            y = x[:, :target_len, :]
        else:
            pad = target_len - t
            y = mx.pad(x, [(0, 0), (0, pad), (0, 0)])
        return y

    def _smooth_harmonic_control(self, x: mx.array) -> mx.array:
        taps = int(getattr(self.cfg, "decoder_harmonic_control_smooth", 0) or 0)
        if taps <= 1:
            return x
        if taps % 2 == 0:
            taps += 1
        if int(x.shape[-1]) == 1:
            kernel = mx.ones((1, taps, 1), dtype=x.dtype) / float(taps)
            return mx.conv1d(x, kernel, padding=taps // 2)
        ch = int(x.shape[-1])
        kernel = mx.ones((ch, taps, 1), dtype=x.dtype) / float(taps)
        return mx.conv1d(x, kernel, padding=taps // 2, groups=ch)

    def _harmonic_excitation_components(
        self, f_raw: mx.array, amp_raw: mx.array, group_raw: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        fmin = max(1.0, float(getattr(self.cfg, "decoder_harmonic_fmin_hz", 60.0)))
        fmax = max(fmin + 1.0, float(getattr(self.cfg, "decoder_harmonic_fmax_hz", 450.0)))
        nyq = 0.5 * float(self.cfg.sample_rate)
        fmax = min(fmax, nyq * 0.95)
        freq = fmin + (fmax - fmin) * mx.sigmoid(f_raw)
        freq = mx.clip(self._smooth_harmonic_control(freq), fmin, fmax)
        phase_inc = (2.0 * math.pi / float(self.cfg.sample_rate)) * freq
        phase = mx.cumsum(phase_inc, axis=1)
        partials = max(1, int(getattr(self.cfg, "decoder_harmonic_partials", 6)))
        groups = max(0, int(getattr(self.cfg, "decoder_harmonic_groups", 0)))
        rolloff = max(0.1, float(getattr(self.cfg, "decoder_harmonic_rolloff", 1.0)))
        group_gate = None
        if groups > 0:
            group_bias = float(getattr(self.cfg, "decoder_harmonic_group_bias", -1.5))
            group_gate = mx.sigmoid(group_raw[:, :, :groups] + group_bias)
        exc = mx.zeros_like(freq)
        norm = 0.0
        for k in range(1, partials + 1):
            wk = 1.0 / (float(k) ** rolloff)
            if group_gate is not None:
                gi = min(groups - 1, int((k - 1) * groups / partials))
                gk = group_gate[:, :, gi : gi + 1]
                exc = exc + (wk * gk) * mx.sin(float(k) * phase)
            else:
                exc = exc + wk * mx.sin(float(k) * phase)
            norm += wk
        exc = exc / max(norm, 1e-6)
        amp_bias = float(getattr(self.cfg, "decoder_harmonic_amp_bias", -2.0))
        amp = mx.sigmoid(amp_raw + amp_bias)
        amp = mx.clip(self._smooth_harmonic_control(amp), 0.0, 1.0)
        gain = max(0.0, float(getattr(self.cfg, "decoder_harmonic_gain", 0.25)))
        return gain * amp * exc, freq, amp

    def _harmonic_source_components(self, y: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        base = y[:, :, 0:1]
        harm, freq, amp = self._harmonic_excitation_components(y[:, :, 1:2], y[:, :, 2:3], y[:, :, 3:])
        return mx.tanh(base + harm), freq, amp

    def _band_split_harmonic_components(self, y: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        branch_raw = y[:, :, :3]
        bands = apply_band_split_filterbank(
            branch_raw,
            sample_rate=self.cfg.sample_rate,
            cutoffs_hz=self.cfg.band_split_cutoffs_hz,
            taps=self.cfg.band_split_taps,
        )
        if self.harmonic_mode == "source_filter":
            residual_weights = mx.array(
                getattr(self.cfg, "decoder_harmonic_residual_band_weights", (1.0, 0.25, 0.05)),
                dtype=bands.dtype,
            )
            bands = bands * mx.reshape(residual_weights, (1, 1, 3))
            harm, freq, amp = self._harmonic_excitation_components(y[:, :, 3:4], y[:, :, 4:5], y[:, :, 7:])
            env_bias = float(getattr(self.cfg, "decoder_harmonic_env_bias", 0.0))
            env_mid_high = mx.sigmoid(y[:, :, 5:7] + env_bias)
            env_mid_high = self._smooth_harmonic_control(env_mid_high)
            env = mx.concatenate([mx.zeros_like(env_mid_high[:, :, :1]), env_mid_high], axis=2)
        else:
            harm, freq, amp = self._harmonic_excitation_components(y[:, :, 3:4], y[:, :, 4:5], y[:, :, 5:])
            env = None
        harm_bands = apply_band_split_filterbank(
            harm,
            sample_rate=self.cfg.sample_rate,
            cutoffs_hz=self.cfg.band_split_cutoffs_hz,
            taps=self.cfg.band_split_taps,
        )
        weights = mx.array(getattr(self.cfg, "decoder_harmonic_band_weights", (0.0, 0.35, 1.0)), dtype=harm_bands.dtype)
        harm_scale = mx.reshape(weights, (1, 1, 3))
        if env is not None:
            harm_scale = harm_scale * env
        bands = bands + harm_bands * harm_scale
        return mx.tanh(mx.sum(bands, axis=2, keepdims=True)), bands, freq, amp

    def _apply_harmonic_source(self, y: mx.array) -> mx.array:
        return self._harmonic_source_components(y)[0]

    def forward_with_harmonic_aux(self, z: mx.array, target_len: int) -> tuple[mx.array, mx.array | None, mx.array | None]:
        y = self._decode_to_length(z, target_len)
        if self.harmonic_source and self.band_split:
            y_hat, _, freq, amp = self._band_split_harmonic_components(y)
            return y_hat, freq, amp
        if self.harmonic_source:
            return self._harmonic_source_components(y)
        return mx.tanh(y), None, None

    def forward_with_bands(self, z: mx.array, target_len: int) -> tuple[mx.array, mx.array | None]:
        y = self._decode_to_length(z, target_len)
        if self.harmonic_source and self.band_split:
            y_hat, bands, _, _ = self._band_split_harmonic_components(y)
            return y_hat, bands
        if self.harmonic_source:
            return self._apply_harmonic_source(y), None
        if self.band_split:
            bands = apply_band_split_filterbank(
                y,
                sample_rate=self.cfg.sample_rate,
                cutoffs_hz=self.cfg.band_split_cutoffs_hz,
                taps=self.cfg.band_split_taps,
            )
            return mx.tanh(mx.sum(bands, axis=2, keepdims=True)), bands
        return mx.tanh(y), None

    def forward_with_bands_and_harmonic_aux(
        self, z: mx.array, target_len: int
    ) -> tuple[mx.array, mx.array | None, mx.array | None, mx.array | None]:
        y = self._decode_to_length(z, target_len)
        if self.harmonic_source and self.band_split:
            y_hat, bands, freq, amp = self._band_split_harmonic_components(y)
            return y_hat, bands, freq, amp
        if self.harmonic_source:
            y_hat, freq, amp = self._harmonic_source_components(y)
            return y_hat, None, freq, amp
        if self.band_split:
            y_hat, bands = self.forward_with_bands(z, target_len)
            return y_hat, bands, None, None
        return mx.tanh(y), None, None, None

    def __call__(self, z: mx.array, target_len: int) -> mx.array:
        return self.forward_with_bands(z, target_len)[0]
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
            self.layers.append(_act_from_string(act, self.dim))
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


class SelfAttention1D(nn.Module):
    """Residual multi-head self-attention over NLC latent sequences."""

    def __init__(self, dim: int, heads: int = 8, *, causal: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.causal = bool(causal)
        if self.heads < 1:
            raise ValueError("self_attention_heads must be >= 1")
        if self.dim % self.heads != 0:
            raise ValueError(f"latent_dim={self.dim} must be divisible by self_attention_heads={self.heads}")
        self.head_dim = self.dim // self.heads
        self.norm = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.out = nn.Linear(self.dim, self.dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, t, c = x.shape
        h = self.norm(x)
        qkv = mx.reshape(self.qkv(h), (b, t, 3, self.heads, self.head_dim))
        q = mx.transpose(qkv[:, :, 0, :, :], (0, 2, 1, 3))
        k = mx.transpose(qkv[:, :, 1, :, :], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2, :, :], (0, 2, 1, 3))
        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.head_dim ** -0.5,
            mask="causal" if self.causal else None,
        )
        y = mx.transpose(y, (0, 2, 1, 3))
        y = mx.reshape(y, (b, t, c))
        return x + self.out(y)


class LatentSelfAttentionStack(nn.Module):
    def __init__(self, dim: int, depth: int, *, heads: int = 8, causal: bool = False):
        super().__init__()
        self.depth = max(0, int(depth))
        self.layers = []
        for i in range(self.depth):
            layer = SelfAttention1D(dim, heads=heads, causal=causal)
            self.layers.append(layer)
            setattr(self, f"_sa{i}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class StateSpace1D(nn.Module):
    """Residual diagonal SSM over NLC latent sequences using causal FFT convolution."""

    def __init__(
        self,
        dim: int,
        *,
        state_dim: int = 16,
        expand: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.state_dim = max(1, int(state_dim))
        self.expand = max(1, int(expand))
        self.inner = self.dim * self.expand
        self.bidirectional = bool(bidirectional)
        self.norm = nn.LayerNorm(self.dim)
        self.in_proj = nn.Linear(self.dim, 2 * self.inner)
        self.out_proj = nn.Linear(self.inner, self.dim)
        self.out_proj.weight = self.out_proj.weight * 0.02
        if self.out_proj.bias is not None:
            self.out_proj.bias = mx.zeros_like(self.out_proj.bias)

        # Log-spaced stable poles: short to long memory, then lightly jittered per channel.
        lo = math.log(0.20 / 0.80)
        hi = math.log(0.995 / 0.005)
        base = mx.linspace(lo, hi, self.state_dim, dtype=mx.float32)
        base = mx.broadcast_to(base[None, :], (self.inner, self.state_dim))
        self.logit_decay = base + 0.01 * mx.random.normal(shape=(self.inner, self.state_dim))
        self.amp = 0.02 * mx.random.normal(shape=(self.inner, self.state_dim)) / math.sqrt(float(self.state_dim))
        if self.bidirectional:
            self.logit_decay_b = base + 0.01 * mx.random.normal(shape=(self.inner, self.state_dim))
            self.amp_b = 0.02 * mx.random.normal(shape=(self.inner, self.state_dim)) / math.sqrt(float(self.state_dim))
        else:
            self.logit_decay_b = None
            self.amp_b = None

    def _kernel(self, length: int, amp: mx.array, logit_decay: mx.array) -> mx.array:
        t = mx.arange(length, dtype=mx.float32)
        decay = mx.sigmoid(mx.clip(logit_decay, -12.0, 12.0))
        decay = mx.clip(decay, 1e-4, 1.0 - 1e-5)
        k = amp[None, :, :] * mx.exp(t[:, None, None] * mx.log(decay[None, :, :]))
        return mx.sum(k, axis=-1)

    def _causal_fft_conv(self, u: mx.array, k: mx.array) -> mx.array:
        t = int(u.shape[1])
        if t <= 0:
            return u
        fft_len = 1 << max(1, (2 * t - 1).bit_length())
        u_f = mx.fft.rfft(u, n=fft_len, axis=1)
        k_f = mx.fft.rfft(k, n=fft_len, axis=0)
        y = mx.fft.irfft(u_f * k_f[None, :, :], n=fft_len, axis=1)
        return y[:, :t, :]

    def __call__(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        h = self.in_proj(self.norm(x))
        u, gate = mx.split(h, 2, axis=-1)
        u = u * gate * mx.sigmoid(gate)
        y = self._causal_fft_conv(u, self._kernel(int(t), self.amp, self.logit_decay))
        if self.bidirectional:
            assert self.amp_b is not None and self.logit_decay_b is not None
            u_b = u[:, ::-1, :]
            k_b = self._kernel(int(t), self.amp_b, self.logit_decay_b)
            y_b = self._causal_fft_conv(u_b, k_b)[:, ::-1, :]
            y = 0.5 * (y + y_b)
        return x + self.out_proj(y)


class LatentStateSpaceStack(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        state_dim: int = 16,
        expand: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.depth = max(0, int(depth))
        self.layers = []
        for i in range(self.depth):
            layer = StateSpace1D(
                dim,
                state_dim=state_dim,
                expand=expand,
                bidirectional=bidirectional,
            )
            self.layers.append(layer)
            setattr(self, f"_ssm{i}", layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class MLXCodec(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        pre_attn_depth = int(getattr(cfg, "self_attention_depth", 0))
        post_attn_depth = int(getattr(cfg, "self_attention_post_depth", 0))
        pre_ssm_depth = int(getattr(cfg, "state_space_depth", 0))
        post_ssm_depth = int(getattr(cfg, "state_space_post_depth", 0))
        if pre_attn_depth > 0 and pre_ssm_depth > 0:
            raise ValueError("self_attention_depth and state_space_depth are mutually exclusive")
        if post_attn_depth > 0 and post_ssm_depth > 0:
            raise ValueError("self_attention_post_depth and state_space_post_depth are mutually exclusive")
        if bool(getattr(cfg, "causal", False)) and bool(getattr(cfg, "state_space_bidirectional", True)) and (
            pre_ssm_depth > 0 or post_ssm_depth > 0
        ):
            raise ValueError("state_space_bidirectional is incompatible with causal state-space blocks")
        self.encoder = Encoder(cfg)
        self.rvq = ResidualVectorQuantizer(cfg)
        self.decoder_band_latent_split = bool(getattr(cfg, "decoder_band_latent_split", False))
        self.decoder = None if self.decoder_band_latent_split else Decoder(cfg)
        self.band_latent_dims: tuple[int, ...] | None = None
        if self.decoder_band_latent_split:
            splits = getattr(cfg, "rvq_band_splits", None)
            if not (bool(getattr(cfg, "decoder_band_split", False)) and splits and len(splits) == 3):
                raise ValueError("decoder_band_latent_split requires decoder_band_split and three rvq_band_splits")
            base = int(cfg.latent_dim) // 3
            rem = int(cfg.latent_dim) % 3
            self.band_latent_dims = tuple(base + (1 if i < rem else 0) for i in range(3))
            self.band_decoders = []
            for i, dim in enumerate(self.band_latent_dims):
                dcfg = replace(
                    cfg,
                    latent_dim=int(dim),
                    decoder_band_split=False,
                    decoder_band_head_depth=0,
                    decoder_band_latent_split=False,
                    decoder_harmonic_source=False,
                )
                dec = Decoder(dcfg)
                self.band_decoders.append(dec)
                setattr(self, f"_band_decoder{i}", dec)
        self.pre_vq_ln = nn.LayerNorm(cfg.latent_dim) if cfg.pre_vq_layernorm else None
        self.latent_pre = (
            LatentTemporalStack(cfg.latent_dim, cfg.latent_temporal_depth, activation=cfg.activation)
            if cfg.latent_temporal_depth > 0
            else None
        )
        self.latent_attn = (
            LatentSelfAttentionStack(
                cfg.latent_dim,
                pre_attn_depth,
                heads=cfg.self_attention_heads,
                causal=cfg.causal,
            )
            if pre_attn_depth > 0
            else (
                LatentStateSpaceStack(
                    cfg.latent_dim,
                    pre_ssm_depth,
                    state_dim=cfg.state_space_state_dim,
                    expand=cfg.state_space_expand,
                    bidirectional=cfg.state_space_bidirectional,
                )
                if pre_ssm_depth > 0
                else None
            )
        )
        self.latent_post = (
            LatentTemporalStack(cfg.latent_dim, cfg.latent_temporal_post_depth, activation=cfg.activation)
            if cfg.latent_temporal_post_depth > 0
            else None
        )
        self.latent_attn_post = (
            LatentSelfAttentionStack(
                cfg.latent_dim,
                post_attn_depth,
                heads=cfg.self_attention_heads,
                causal=cfg.causal,
            )
            if post_attn_depth > 0
            else (
                LatentStateSpaceStack(
                    cfg.latent_dim,
                    post_ssm_depth,
                    state_dim=cfg.state_space_state_dim,
                    expand=cfg.state_space_expand,
                    bidirectional=cfg.state_space_bidirectional,
                )
                if post_ssm_depth > 0
                else None
            )
        )

    def _decode_from_latent(self, z_q: mx.array, tlen: int) -> mx.array:
        if not self.decoder_band_latent_split:
            assert self.decoder is not None
            return self.decoder(z_q, tlen)
        assert self.band_latent_dims is not None
        raws = []
        c0 = 0
        for dim, dec in zip(self.band_latent_dims, self.band_decoders, strict=True):
            c1 = c0 + dim
            raws.append(dec._decode_to_length(z_q[:, :, c0:c1], tlen))
            c0 = c1
        raw = mx.concatenate(raws, axis=2)
        bands = apply_band_split_filterbank(
            raw,
            sample_rate=self.cfg.sample_rate,
            cutoffs_hz=self.cfg.band_split_cutoffs_hz,
            taps=self.cfg.band_split_taps,
        )
        return mx.tanh(mx.sum(bands, axis=2, keepdims=True))

    def _decode_from_latent_with_bands(self, z_q: mx.array, tlen: int) -> tuple[mx.array, mx.array | None]:
        if not self.decoder_band_latent_split:
            assert self.decoder is not None
            return self.decoder.forward_with_bands(z_q, tlen)
        assert self.band_latent_dims is not None
        raws = []
        c0 = 0
        for dim, dec in zip(self.band_latent_dims, self.band_decoders, strict=True):
            c1 = c0 + dim
            raws.append(dec._decode_to_length(z_q[:, :, c0:c1], tlen))
            c0 = c1
        raw = mx.concatenate(raws, axis=2)
        bands = apply_band_split_filterbank(
            raw,
            sample_rate=self.cfg.sample_rate,
            cutoffs_hz=self.cfg.band_split_cutoffs_hz,
            taps=self.cfg.band_split_taps,
        )
        return mx.tanh(mx.sum(bands, axis=2, keepdims=True)), bands

    def latent_before_rvq(self, x: mx.array) -> mx.array:
        """Encoder output after optional LayerNorm + pre-RVQ temporal stack (matches RVQ input)."""
        z = self.encoder(x)
        if self.pre_vq_ln is not None:
            z = self.pre_vq_ln(z)
        if self.latent_pre is not None:
            z = self.latent_pre(z)
        if self.latent_attn is not None:
            z = self.latent_attn(z)
        return z

    def forward_full(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None]:
        """Returns (recon, vq_loss, ent_pos, marginal_ent, indices or None)."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        if self.cfg.ae_only:
            y = self._decode_from_latent(z, tlen)
            return y, mx.array(0.0), mx.array(0.0), mx.array(0.0), None
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        if self.latent_attn_post is not None:
            z_q = self.latent_attn_post(z_q)
        y = self._decode_from_latent(z_q, tlen)
        return y, vq_loss, ent_pos, marg_ent, indices

    def forward_full_with_bands(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None, mx.array | None]:
        """Hard-RVQ reconstruction plus optional decoder low/mid/high branch bands."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        if self.cfg.ae_only:
            y, bands = self._decode_from_latent_with_bands(z, tlen)
            return y, mx.array(0.0), mx.array(0.0), mx.array(0.0), None, bands
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        if self.latent_attn_post is not None:
            z_q = self.latent_attn_post(z_q)
        y, bands = self._decode_from_latent_with_bands(z_q, tlen)
        return y, vq_loss, ent_pos, marg_ent, indices, bands

    def forward_full_with_harmonic_aux(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None, mx.array | None, mx.array | None]:
        """Hard-RVQ reconstruction plus harmonic-source ``freq`` and ``amp`` when enabled."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        if self.cfg.ae_only:
            y, freq, amp = self.decoder.forward_with_harmonic_aux(z, tlen)
            return y, mx.array(0.0), mx.array(0.0), mx.array(0.0), None, freq, amp
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        if self.latent_attn_post is not None:
            z_q = self.latent_attn_post(z_q)
        y, freq, amp = self.decoder.forward_with_harmonic_aux(z_q, tlen)
        return y, vq_loss, ent_pos, marg_ent, indices, freq, amp

    def forward_full_with_bands_and_harmonic_aux(
        self, x: mx.array
    ) -> tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        list[mx.array] | None,
        mx.array | None,
        mx.array | None,
        mx.array | None,
    ]:
        """Hard-RVQ reconstruction plus decoder bands and harmonic aux tensors."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        if self.cfg.ae_only:
            y, bands, freq, amp = self.decoder.forward_with_bands_and_harmonic_aux(z, tlen)
            return y, mx.array(0.0), mx.array(0.0), mx.array(0.0), None, bands, freq, amp
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        if self.latent_attn_post is not None:
            z_q = self.latent_attn_post(z_q)
        y, bands, freq, amp = self.decoder.forward_with_bands_and_harmonic_aux(z_q, tlen)
        return y, vq_loss, ent_pos, marg_ent, indices, bands, freq, amp

    def forward_full_with_continuous(
        self, x: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None, mx.array]:
        """Hard-RVQ reconstruction plus continuous z→decoder anchor from the same encoder pass."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        y_cont = self._decode_from_latent(z, tlen)
        if self.cfg.ae_only:
            return y_cont, mx.array(0.0), mx.array(0.0), mx.array(0.0), None, y_cont
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        if self.latent_attn_post is not None:
            z_q = self.latent_attn_post(z_q)
        y = self._decode_from_latent(z_q, tlen)
        return y, vq_loss, ent_pos, marg_ent, indices, y_cont

    def forward_full_with_continuous_and_harmonic_aux(
        self, x: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None, mx.array, mx.array | None, mx.array | None]:
        """Hard-RVQ reconstruction, continuous anchor, and harmonic-source aux tensors."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        y_cont = self._decode_from_latent(z, tlen)
        if self.cfg.ae_only:
            y, freq, amp = self.decoder.forward_with_harmonic_aux(z, tlen)
            return y, mx.array(0.0), mx.array(0.0), mx.array(0.0), None, y_cont, freq, amp
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        if self.latent_attn_post is not None:
            z_q = self.latent_attn_post(z_q)
        y, freq, amp = self.decoder.forward_with_harmonic_aux(z_q, tlen)
        return y, vq_loss, ent_pos, marg_ent, indices, y_cont, freq, amp

    def forward_full_with_continuous_and_bands(
        self, x: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None, mx.array, mx.array | None]:
        """Hard-RVQ reconstruction, continuous anchor, and optional decoder branch bands."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        y_cont = self.decoder(z, tlen)
        if self.cfg.ae_only:
            y, bands = self._decode_from_latent_with_bands(z, tlen)
            return y, mx.array(0.0), mx.array(0.0), mx.array(0.0), None, y_cont, bands
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        if self.latent_attn_post is not None:
            z_q = self.latent_attn_post(z_q)
        y, bands = self._decode_from_latent_with_bands(z_q, tlen)
        return y, vq_loss, ent_pos, marg_ent, indices, y_cont, bands

    def forward_reconstruction_only(self, x: mx.array) -> mx.array:
        """Reconstruction ``ŷ`` only (same graph as ``forward_full``)."""
        return self.forward_full(x)[0]

    def encode_decode_streaming_stub(self, x: mx.array) -> mx.array:
        """Chunked/streaming API stub: non-stateful full-chunk pass (causal = left-pad only within chunk)."""
        return self.forward_reconstruction_only(x)
