"""Minimal conditional DDPM on log-mel (experiment / precursor to codec decoder).

Trains noise prediction ε_θ(x_t, t, cond) with cond = Conv1d features of clean mel.
Inference: reverse diffusion from Gaussian noise (DDIM optional shortcut).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """timesteps: [B] int64 → [B, dim] float."""
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(0, half, device=device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConvBlock(nn.Module):
    def __init__(self, ch: int, tdim: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(tdim, ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x [B, C, T], t_emb [B, tdim]
        h = self.norm(x)
        h = self.act(h)
        h = self.conv(h)
        h = h + self.t_proj(t_emb).unsqueeze(-1)
        return x + h


class MelCondEpsilonNet(nn.Module):
    """Predict ε for noisy log-mel x_t; conditioned on clean-mel features (same T)."""

    def __init__(self, n_mels: int = 80, base: int = 64, tdim: int = 128, cond_width: int = 64, n_blocks: int = 8):
        super().__init__()
        self.n_mels = n_mels
        self.cond_enc = nn.Sequential(
            nn.Conv1d(n_mels, cond_width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(cond_width, cond_width, kernel_size=3, padding=1),
        )
        in_ch = n_mels + cond_width
        self.t_mlp = nn.Sequential(
            nn.Linear(128, tdim * 4),
            nn.SiLU(),
            nn.Linear(tdim * 4, tdim * 4),
        )
        self.in_conv = nn.Conv1d(in_ch, base, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ConvBlock(base, tdim * 4) for _ in range(n_blocks)])
        self.out = nn.Conv1d(base, n_mels, kernel_size=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, mel_clean: torch.Tensor) -> torch.Tensor:
        """
        x_t:     [B, n_mels, T] noisy log-mel
        t:       [B] int64 timestep index
        mel_clean: [B, n_mels, T] for conditioning (teacher)
        """
        t_emb = _sinusoidal_embedding(t, 128)
        t_emb = self.t_mlp(t_emb)
        cond = self.cond_enc(mel_clean)
        h = torch.cat([x_t, cond], dim=1)
        h = self.in_conv(h)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out(h)


class DDPMMel(nn.Module):
    """DDPM utilities + epsilon net."""

    def __init__(self, n_mels: int = 80, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.n_mels = n_mels
        self.T = timesteps
        self.eps_net = MelCondEpsilonNet(n_mels=n_mels)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(min=1e-20)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("sqrt_acp", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_om_acp", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        # x_t = sqrt(ā)*x0 + sqrt(1-ā)*ε
        s1 = self.sqrt_acp[t].view(-1, 1, 1)
        s2 = self.sqrt_om_acp[t].view(-1, 1, 1)
        return s1 * x0 + s2 * noise

    def loss(self, mel_log: torch.Tensor) -> torch.Tensor:
        """mel_log: [B, n_mels, T] clean log-mel (e.g. clamped log)."""
        b = mel_log.size(0)
        device = mel_log.device
        t = torch.randint(0, self.T, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(mel_log)
        x_t = self.q_sample(mel_log, t, noise)
        pred = self.eps_net(x_t, t, mel_log)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, mel_cond: torch.Tensor) -> torch.Tensor:
        """DDPM ancestral sample. mel_cond: [B, n_mels, T] (conditioning path; can be zeros)."""
        self.eval()
        device = mel_cond.device
        b, _, time = mel_cond.shape
        x = torch.randn(b, self.n_mels, time, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            beta = self.betas[i]
            alpha = 1.0 - beta
            alpha_bar = self.alphas_cumprod[i]
            eps = self.eps_net(x, t, mel_cond)
            coef = 1.0 / torch.sqrt(alpha)
            mean = coef * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * eps)
            if i > 0:
                pv = self.posterior_variance[i].clamp(min=1e-20)
                x = mean + torch.sqrt(pv) * torch.randn_like(x)
            else:
                x = mean
        return x
