"""
Conditional mel refiner: maps coarse VQ mels toward ground-truth mels.

Residual dilated CNN (~0.5–1.5M params): delta = net(coarse), refined = coarse + delta.
"""
import torch
import torch.nn as nn


class MelRefinerNet(nn.Module):
    """Dilated residual stack on coarse log-mel [B, n_mels, T]."""

    def __init__(self, n_mels: int = 100, hidden: int = 128, n_layers: int = 10):
        super().__init__()
        self.n_mels = n_mels
        self.in_proj = nn.Conv1d(n_mels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            d = 2 ** (i % 4)
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d),
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                    nn.Conv1d(hidden, hidden, 1),
                    nn.GroupNorm(8, hidden),
                )
            )
        self.out = nn.Conv1d(hidden, n_mels, kernel_size=1)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        """Returns residual delta; refined = coarse + delta."""
        h = self.in_proj(coarse)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)

    def refine(self, coarse: torch.Tensor) -> torch.Tensor:
        return coarse + self.forward(coarse)


# Alias for thesis docs
MelRefinerUNet = MelRefinerNet
