"""Encoder, RVQ, decoder, and full codec (MLX)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..config import Config, effective_codebook_sizes
from .. import streaming as streaming_mod
from .losses import entropy_from_logits, marginal_code_entropy_from_dist

def _enc_conv(cfg: Config, cin: int, cout: int, stride: int) -> nn.Module:
    if cfg.causal:
        return streaming_mod.CausalConv1d(cin, cout, 7, stride=stride)
    return nn.Conv1d(cin, cout, 7, stride=stride, padding=3)


def _act_module(cfg: Config, ch: int) -> nn.Module:
    a = (cfg.activation or "gelu").strip().lower()
    if a in ("snake", "snake_beta"):
        return SnakeBeta(ch)
    return nn.GELU()


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
            idx = mx.argmin(dist, axis=-1)
            rn = mx.sqrt(z2 + 1e-8)
            flat = mx.reshape(idx, (-1,))
            z_q_dir = mx.reshape(mx.take(eb, flat, axis=0), r.shape)
            z_q_low = z_q_dir * rn
        else:
            z2 = mx.sum(r * r, axis=-1, keepdims=True)
            e2 = mx.sum(cb * cb, axis=-1)
            dots = r @ mx.transpose(cb)
            dist = z2 + e2 - 2.0 * dots
            idx = mx.argmin(dist, axis=-1)
            z_q_low = self.embedding(idx)
        if self.out_proj is not None:
            z_q = self.out_proj(z_q_low)
        else:
            z_q = z_q_low
        z_st = residual + mx.stop_gradient(z_q - residual)
        commit = mx.mean((mx.stop_gradient(z_q) - residual) ** 2)
        codebook = mx.mean((z_q - mx.stop_gradient(residual)) ** 2)
        vq_loss = self.beta * commit + codebook
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
        if use_repeat:
            self.layers.append(UpsampleRepeatConv(d, c[0]))
        else:
            self.layers.append(nn.ConvTranspose1d(d, c[0], 7, stride=2, padding=3))
        self.layers.append(_act_module(cfg, c[0]))
        for _ in range(n1):
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
                self.layers.append(nn.ConvTranspose1d(c[i], c[i + 1], 7, stride=2, padding=3))
            self.layers.append(_act_module(cfg, c[i + 1]))
            for _ in range(n1):
                self.layers.append(
                    streaming_mod.CausalConv1d(c[i + 1], c[i + 1], 7, stride=1)
                    if cfg.causal
                    else nn.Conv1d(c[i + 1], c[i + 1], 7, stride=1, padding=3)
                )
                self.layers.append(_act_module(cfg, c[i + 1]))
        self.out = (
            streaming_mod.CausalConv1d(c[-1], 1, 7, stride=1)
            if cfg.causal
            else nn.Conv1d(c[-1], 1, 7, padding=3)
        )
        for i, layer in enumerate(self.layers):
            setattr(self, f"_d{i}", layer)

    def __call__(self, z: mx.array, target_len: int) -> mx.array:
        x = z
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        t = x.shape[1]
        if t >= target_len:
            y = x[:, :target_len, :]
        else:
            pad = target_len - t
            y = mx.pad(x, [(0, 0), (0, pad), (0, 0)])
        return mx.tanh(y)
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


class MLXCodec(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.rvq = ResidualVectorQuantizer(cfg)
        self.decoder = Decoder(cfg)
        self.pre_vq_ln = nn.LayerNorm(cfg.latent_dim) if cfg.pre_vq_layernorm else None
        self.latent_pre = (
            LatentTemporalStack(cfg.latent_dim, cfg.latent_temporal_depth, activation=cfg.activation)
            if cfg.latent_temporal_depth > 0
            else None
        )
        self.latent_post = (
            LatentTemporalStack(cfg.latent_dim, cfg.latent_temporal_post_depth, activation=cfg.activation)
            if cfg.latent_temporal_post_depth > 0
            else None
        )

    def latent_before_rvq(self, x: mx.array) -> mx.array:
        """Encoder output after optional LayerNorm + pre-RVQ temporal stack (matches RVQ input)."""
        z = self.encoder(x)
        if self.pre_vq_ln is not None:
            z = self.pre_vq_ln(z)
        if self.latent_pre is not None:
            z = self.latent_pre(z)
        return z

    def forward_full(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, list[mx.array] | None]:
        """Returns (recon, vq_loss, ent_pos, marginal_ent, indices or None)."""
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        if self.cfg.ae_only:
            y = self.decoder(z, tlen)
            return y, mx.array(0.0), mx.array(0.0), mx.array(0.0), None
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        y = self.decoder(z_q, tlen)
        return y, vq_loss, ent_pos, marg_ent, indices

    def forward_reconstruction_only(self, x: mx.array) -> mx.array:
        """Reconstruction ``ŷ`` only (same graph as ``forward_full``)."""
        return self.forward_full(x)[0]

    def encode_decode_streaming_stub(self, x: mx.array) -> mx.array:
        """Chunked/streaming API stub: non-stateful full-chunk pass (causal = left-pad only within chunk)."""
        return self.forward_reconstruction_only(x)

