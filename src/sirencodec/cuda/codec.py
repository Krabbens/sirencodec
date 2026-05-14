"""PyTorch/CUDA codec backend for the former MLX trainer CLI."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config, effective_codebook_sizes
from .losses import entropy_from_logits, marginal_code_entropy_from_dist


class NLCConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, *, stride: int = 1, padding: int = 0, dilation: int = 1, causal: bool = False):
        super().__init__()
        self.causal = bool(causal)
        self.kernel = int(kernel)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            cin,
            cout,
            kernel,
            stride=stride,
            padding=0 if self.causal else padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        if self.causal:
            x = F.pad(x, (self.dilation * (self.kernel - 1), 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class NLCConvTranspose1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, *, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, kernel, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2).contiguous()).transpose(1, 2)


class UpsampleRepeatConv(nn.Module):
    """2x time via sample repetition + smoothing Conv1d, NLC layout."""

    def __init__(self, cin: int, cout: int, k: int = 7):
        super().__init__()
        self.conv = NLCConv1d(cin, cout, k, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.repeat_interleave(2, dim=1))


class SnakeBeta(nn.Module):
    """DAC-style periodic activation: x + (1/a) * sin(a*x)^2."""

    def __init__(self, channels: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(int(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.exp(torch.clamp(self.log_alpha, -10.0, 10.0)).view(1, 1, -1)
        return x + (torch.sin(a * x) ** 2) / (a + 1e-8)


def _act_module(cfg: Config, ch: int) -> nn.Module:
    a = (cfg.activation or "gelu").strip().lower()
    if a in ("snake", "snake_beta"):
        return SnakeBeta(ch)
    return nn.GELU()


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        ch = (1,) + cfg.enc_channels
        n1 = max(0, int(cfg.stride1_blocks_per_scale))
        layers: list[nn.Module] = []
        for i in range(len(cfg.enc_channels)):
            for _ in range(n1):
                layers.append(NLCConv1d(ch[i], ch[i], 7, stride=1, padding=3, causal=cfg.causal))
                layers.append(_act_module(cfg, ch[i]))
            layers.append(NLCConv1d(ch[i], ch[i + 1], 7, stride=2, padding=3, causal=cfg.causal))
            layers.append(_act_module(cfg, ch[i + 1]))
        layers.append(NLCConv1d(cfg.enc_channels[-1], cfg.latent_dim, 3, padding=1, causal=cfg.causal))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VectorQuantizerStage(nn.Module):
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
        self.beta = float(cfg.vq_commitment)

    def project_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return self.in_proj(residual) if self.in_proj is not None else residual

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.project_residual(residual)
        cb = self.embedding.weight
        if self.cfg.vq_cosine:
            zn = F.normalize(r, dim=-1, eps=1e-8)
            eb = F.normalize(cb, dim=-1, eps=1e-8)
            dist = 2.0 - 2.0 * torch.matmul(zn, eb.t())
            idx = torch.argmin(dist, dim=-1)
            z_q_low = F.embedding(idx, eb) * torch.linalg.vector_norm(r, dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            z2 = torch.sum(r * r, dim=-1, keepdim=True)
            e2 = torch.sum(cb * cb, dim=-1)
            dist = z2 + e2 - 2.0 * torch.matmul(r, cb.t())
            idx = torch.argmin(dist, dim=-1)
            z_q_low = self.embedding(idx)
        z_q = self.out_proj(z_q_low) if self.out_proj is not None else z_q_low
        z_st = residual + (z_q - residual).detach()
        commit = torch.mean((z_q.detach() - residual) ** 2)
        codebook = torch.mean((z_q - residual.detach()) ** 2)
        return z_st, self.beta * commit + codebook, dist, idx


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.stages = nn.ModuleList(VectorQuantizerStage(cfg, k) for k in effective_codebook_sizes(cfg))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        quantized = torch.zeros_like(z)
        total_vq = z.new_zeros(())
        total_ent = z.new_zeros(())
        total_marg = z.new_zeros(())
        indices: list[torch.Tensor] = []
        for stage in self.stages:
            # Previous straight-through stages would otherwise cancel the residual gradient.
            residual = z - quantized.detach()
            z_i, lvq, dist, idx = stage(residual)
            quantized = quantized + z_i
            total_vq = total_vq + lvq
            indices.append(idx)
            if self.cfg.lambda_entropy > 0:
                total_ent = total_ent + entropy_from_logits(-dist)
            if self.cfg.lambda_marginal > 0:
                total_marg = total_marg + marginal_code_entropy_from_dist(dist, self.cfg.marginal_tau)
        if self.cfg.lambda_entropy > 0 and len(self.stages) > 0:
            total_ent = total_ent / float(len(self.stages))
        if self.cfg.lambda_marginal > 0 and len(self.stages) > 0:
            total_marg = total_marg / float(len(self.stages))
        return quantized, total_vq, total_ent, total_marg, indices


class DecoderRefineStack(nn.Module):
    """Small residual temporal post-filter."""

    def __init__(self, cfg: Config, channels: int, depth: int, gain: float):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(max(0, int(depth))):
            dilation = 2 ** (i % 4)
            layers.append(NLCConv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation, causal=cfg.causal))
            layers.append(_act_module(cfg, channels))
        self.layers = nn.Sequential(*layers)
        self.gain = float(gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.layers) == 0 or self.gain == 0.0:
            return x
        return x + self.gain * self.layers(x)


class DecoderBandHead(nn.Module):
    """Residual waveform head with a distinct temporal kernel."""

    def __init__(self, cfg: Config, channels: int, depth: int, kernel: int):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(max(0, int(depth))):
            layers.append(NLCConv1d(channels, channels, kernel, padding=kernel // 2, causal=cfg.causal))
            layers.append(_act_module(cfg, channels))
        layers.append(NLCConv1d(channels, 1, kernel, padding=kernel // 2, causal=cfg.causal))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HarmonicSource(nn.Module):
    """Learned differentiable harmonic excitation."""

    def __init__(self, cfg: Config, channels: int):
        super().__init__()
        self.sample_rate = float(cfg.sample_rate)
        self.harmonics = max(1, int(cfg.harmonic_harmonics))
        self.f0_min = float(cfg.harmonic_f0_min)
        self.f0_max = float(cfg.harmonic_f0_max)
        self.f0_head = NLCConv1d(channels, 1, 7, padding=3, causal=cfg.causal)
        self.amp_head = NLCConv1d(channels, self.harmonics, 7, padding=3, causal=cfg.causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0_unit = torch.sigmoid(self.f0_head(x))
        f0 = self.f0_min + (self.f0_max - self.f0_min) * f0_unit
        phase = torch.cumsum(f0 * (2.0 * math.pi / self.sample_rate), dim=1)
        harmonics = torch.arange(1, self.harmonics + 1, dtype=x.dtype, device=x.device).view(1, 1, -1)
        amps = torch.sigmoid(self.amp_head(x)) / float(self.harmonics)
        return torch.sum(amps * torch.sin(phase * harmonics), dim=-1, keepdim=True)


def _lava_highpass_residual(x: torch.Tensor) -> torch.Tensor:
    """Cheap differentiable high-pass residual for LavaSR-style low/high-band merge."""
    left = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
    right = torch.cat([x[:, 1:, :], x[:, -1:, :]], dim=1)
    return x - (0.5 * (left + right))


class PostLavaSRRefiner(nn.Module):
    """Small waveform refiner applied after the decoder, inspired by LavaSR's LR merge."""

    def __init__(self, cfg: Config):
        super().__init__()
        depth = max(0, int(cfg.post_lavasr_depth))
        ch = max(4, int(cfg.post_lavasr_channels))
        kernel = max(3, int(cfg.post_lavasr_kernel))
        if kernel % 2 == 0:
            kernel += 1
        layers: list[nn.Module] = [NLCConv1d(1, ch, kernel, padding=kernel // 2, causal=cfg.causal)]
        for i in range(depth):
            dilation = 2 ** (i % 4)
            layers.append(
                NLCConv1d(
                    ch,
                    ch,
                    kernel,
                    padding=(kernel // 2) * dilation,
                    dilation=dilation,
                    causal=cfg.causal,
                )
            )
            layers.append(_act_module(cfg, ch))
        layers.append(NLCConv1d(ch, 1, kernel, padding=kernel // 2, causal=cfg.causal))
        self.layers = nn.Sequential(*layers)
        self.gain = float(cfg.post_lavasr_gain)
        self.highpass = bool(cfg.post_lavasr_highpass)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        residual = self.layers(y)
        if self.highpass:
            residual = _lava_highpass_residual(residual)
        return torch.tanh(y + self.gain * residual)


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c = list(reversed(cfg.enc_channels))
        d = cfg.latent_dim
        n1 = max(0, int(cfg.stride1_blocks_per_scale))
        use_repeat = (cfg.decoder_upsample or "transpose").lower() == "repeat_conv" or bool(cfg.causal)
        layers: list[nn.Module] = []
        layers.append(UpsampleRepeatConv(d, c[0]) if use_repeat else NLCConvTranspose1d(d, c[0], 7, stride=2, padding=3))
        layers.append(_act_module(cfg, c[0]))
        for _ in range(n1):
            layers.append(NLCConv1d(c[0], c[0], 7, padding=3, causal=cfg.causal))
            layers.append(_act_module(cfg, c[0]))
        for i in range(len(c) - 1):
            layers.append(UpsampleRepeatConv(c[i], c[i + 1]) if use_repeat else NLCConvTranspose1d(c[i], c[i + 1], 7, stride=2, padding=3))
            layers.append(_act_module(cfg, c[i + 1]))
            for _ in range(n1):
                layers.append(NLCConv1d(c[i + 1], c[i + 1], 7, padding=3, causal=cfg.causal))
                layers.append(_act_module(cfg, c[i + 1]))
        self.layers = nn.Sequential(*layers)
        self.out = NLCConv1d(c[-1], 1, 7, padding=3, causal=cfg.causal)
        self.refine = (
            DecoderRefineStack(cfg, c[-1], cfg.decoder_refine_depth, cfg.decoder_refine_gain)
            if int(cfg.decoder_refine_depth) > 0
            else None
        )
        self.band_gain = float(cfg.decoder_band_gain)
        self.band_heads = nn.ModuleList(
            DecoderBandHead(cfg, c[-1], cfg.decoder_band_depth, 7 + 8 * (i + 1))
            for i in range(max(0, int(cfg.decoder_band_heads) - 1))
        )
        self.harmonic_gain = float(cfg.harmonic_amp)
        self.harmonic_source = HarmonicSource(cfg, c[-1]) if bool(cfg.harmonic_source) else None
        self.post_lavasr = PostLavaSRRefiner(cfg) if int(cfg.post_lavasr_depth) > 0 else None

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        features = self.layers(z)
        if self.refine is not None:
            features = self.refine(features)
        y = self.out(features)
        if len(self.band_heads) > 0 and self.band_gain != 0.0:
            extra = torch.zeros_like(y)
            for head in self.band_heads:
                extra = extra + head(features)
            y = y + self.band_gain * (extra / float(len(self.band_heads)))
        if self.harmonic_source is not None and self.harmonic_gain != 0.0:
            y = y + self.harmonic_gain * self.harmonic_source(features)
        t = y.shape[1]
        if t >= target_len:
            y = y[:, :target_len, :]
        else:
            y = F.pad(y, (0, 0, 0, target_len - t))
        y = torch.tanh(y)
        if self.post_lavasr is not None:
            y = self.post_lavasr(y)
        return y


class LatentTemporalStack(nn.Module):
    def __init__(self, dim: int, depth: int, *, activation: str = "snake_beta"):
        super().__init__()
        layers: list[nn.Module] = []
        act = (activation or "gelu").strip().lower()
        for i in range(max(0, int(depth))):
            d = 2 ** (i % 5)
            layers.append(NLCConv1d(dim, dim, 3, stride=1, padding=d, dilation=d))
            layers.append(SnakeBeta(dim) if act in ("snake", "snake_beta") else nn.GELU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.layers) == 0:
            return x
        return x + self.layers(x)


class LatentSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, *, causal: bool = False):
        super().__init__()
        self.causal = bool(causal)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = max(dim, dim * 2)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        mask = None
        if self.causal:
            t = h.shape[1]
            mask = torch.ones((t, t), dtype=torch.bool, device=h.device).triu(1)
        y, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + y
        return x + self.ff(self.norm2(x))


class LatentSelfAttentionStack(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, *, causal: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            *(LatentSelfAttentionBlock(dim, heads, causal=causal) for _ in range(max(0, int(depth))))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class WaveDiscriminator(nn.Module):
    """Lightweight 1D discriminator for waveform hinge GAN training."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        ch = int(base_channels)
        widths = (ch, ch * 2, ch * 4, ch * 8, ch * 16)
        layers: list[nn.Module] = [
            nn.Conv1d(1, widths[0], kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(widths[0], widths[1], kernel_size=41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(widths[1], widths[2], kernel_size=41, stride=4, padding=20, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(widths[2], widths[3], kernel_size=41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(widths[3], widths[4], kernel_size=11, stride=1, padding=5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(widths[4], 1, kernel_size=3, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        y = x.transpose(1, 2)
        feats: list[torch.Tensor] = []
        for layer in self.net:
            y = layer(y)
            if isinstance(layer, nn.Conv1d):
                feats.append(y)
        return y.flatten(1), feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[0]


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN style period discriminator; focuses on pitch-period structure."""

    def __init__(self, period: int, base_channels: int = 16):
        super().__init__()
        self.period = int(period)
        ch = int(base_channels)
        widths = (ch, ch * 2, ch * 4, ch * 8, ch * 16)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, widths[0], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(widths[0], widths[1], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(widths[1], widths[2], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(widths[2], widths[3], kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(widths[3], widths[4], kernel_size=(5, 1), stride=(1, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(widths[4], 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            ]
        )

    def _reshape_period(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        t = int(y.shape[-1])
        pad = (self.period - (t % self.period)) % self.period
        if pad > 0:
            y = F.pad(y, (0, pad), mode="reflect")
            t += pad
        return y.reshape(y.shape[0], 1, t // self.period, self.period)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        y = self._reshape_period(x)
        feats: list[torch.Tensor] = []
        for layer in self.layers:
            y = layer(y)
            if isinstance(layer, nn.Conv2d):
                feats.append(y)
        return y.flatten(1), feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[0]


class MultiScaleWaveDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3, base_channels: int = 32):
        super().__init__()
        self.discriminators = nn.ModuleList(WaveDiscriminator(base_channels=base_channels) for _ in range(max(1, int(n_scales))))
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

    def forward_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        cur = x
        outs: list[torch.Tensor] = []
        feats: list[list[torch.Tensor]] = []
        for i, disc in enumerate(self.discriminators):
            score, fm = disc.forward_features(cur)
            outs.append(score)
            feats.append(fm)
            if i + 1 < len(self.discriminators):
                cur_nct = cur.transpose(1, 2)
                cur = self.downsample(cur_nct).transpose(1, 2)
        return outs, feats

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(x)[0]


class MultiPeriodWaveDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), base_channels: int = 16):
        super().__init__()
        self.discriminators = nn.ModuleList(
            PeriodDiscriminator(period=int(period), base_channels=base_channels) for period in periods
        )

    def forward_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs: list[torch.Tensor] = []
        feats: list[list[torch.Tensor]] = []
        for disc in self.discriminators:
            score, fm = disc.forward_features(x)
            outs.append(score)
            feats.append(fm)
        return outs, feats

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(x)[0]


class MultiScaleMultiPeriodWaveDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        n_scales: int = 3,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        base_channels: int = 16,
    ):
        super().__init__()
        self.msd = MultiScaleWaveDiscriminator(n_scales=n_scales, base_channels=base_channels)
        self.mpd = MultiPeriodWaveDiscriminator(periods=periods, base_channels=base_channels)

    def forward_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        ms_scores, ms_feats = self.msd.forward_features(x)
        mp_scores, mp_feats = self.mpd.forward_features(x)
        return ms_scores + mp_scores, ms_feats + mp_feats

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(x)[0]


def build_wave_discriminator(cfg: Config) -> nn.Module:
    kind = str(getattr(cfg, "disc_type", "msd") or "msd").strip().lower()
    base = int(getattr(cfg, "disc_base_channels", 32))
    scales = int(getattr(cfg, "disc_scales", 3))
    periods = tuple(int(p) for p in getattr(cfg, "disc_periods", (2, 3, 5, 7, 11)))
    if kind == "msd":
        return MultiScaleWaveDiscriminator(n_scales=scales, base_channels=base)
    if kind == "mpd":
        return MultiPeriodWaveDiscriminator(periods=periods, base_channels=base)
    if kind == "msmpd":
        return MultiScaleMultiPeriodWaveDiscriminator(n_scales=scales, periods=periods, base_channels=base)
    raise ValueError(f"unknown disc_type={kind!r}")


class CUDACodec(nn.Module):
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

    def latent_before_rvq(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.pre_vq_ln is not None:
            z = self.pre_vq_ln(z)
        if self.latent_pre is not None:
            z = self.latent_pre(z)
        if self.self_attn_pre is not None:
            z = self.self_attn_pre(z)
        return z

    def forward_full(
        self,
        x: torch.Tensor,
        *,
        quantize_blend: float = 1.0,
        return_continuous: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor] | None] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor] | None,
        torch.Tensor,
    ]:
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        if self.cfg.ae_only:
            z0 = z.new_zeros(())
            y_cont = self.decoder(z, tlen)
            if return_continuous:
                return y_cont, z0, z0, z0, None, y_cont
            return y_cont, z0, z0, z0, None
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        if self.self_attn_post is not None:
            z_q = self.self_attn_post(z_q)
        q = min(1.0, max(0.0, float(quantize_blend)))
        if q < 1.0:
            z_q = z + q * (z_q - z)
        y_hat = self.decoder(z_q, tlen)
        if return_continuous:
            return y_hat, vq_loss, ent_pos, marg_ent, indices, self.decoder(z, tlen)
        return y_hat, vq_loss, ent_pos, marg_ent, indices

    def forward_reconstruction_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_full(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_reconstruction_only(x)


MLXCodec = CUDACodec
