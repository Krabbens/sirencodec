"""PyTorch/CUDA codec backend for the former MLX trainer CLI."""
from __future__ import annotations

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
        x = x.transpose(1, 2)
        if self.causal:
            x = F.pad(x, (self.dilation * (self.kernel - 1), 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class NLCConvTranspose1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, *, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, kernel, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


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
            residual = z - quantized
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
        layers.append(NLCConv1d(c[-1], 1, 7, padding=3, causal=cfg.causal))
        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        x = self.layers(z)
        t = x.shape[1]
        if t >= target_len:
            y = x[:, :target_len, :]
        else:
            y = F.pad(x, (0, 0, 0, target_len - t))
        return torch.tanh(y)


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
        self.latent_post = (
            LatentTemporalStack(cfg.latent_dim, cfg.latent_temporal_post_depth, activation=cfg.activation)
            if cfg.latent_temporal_post_depth > 0
            else None
        )

    def latent_before_rvq(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.pre_vq_ln is not None:
            z = self.pre_vq_ln(z)
        if self.latent_pre is not None:
            z = self.latent_pre(z)
        return z

    def forward_full(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        tlen = x.shape[1]
        z = self.latent_before_rvq(x)
        if self.cfg.ae_only:
            z0 = z.new_zeros(())
            return self.decoder(z, tlen), z0, z0, z0, None
        z_q, vq_loss, ent_pos, marg_ent, indices = self.rvq(z)
        if self.latent_post is not None:
            z_q = self.latent_post(z_q)
        return self.decoder(z_q, tlen), vq_loss, ent_pos, marg_ent, indices

    def forward_reconstruction_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_full(x)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_reconstruction_only(x)


MLXCodec = CUDACodec
