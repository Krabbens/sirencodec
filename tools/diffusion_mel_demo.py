#!/usr/bin/env python3
"""Train a small conditional DDPM on log-mels (proof-of-concept; extend toward codec decoder).

  python run.py diffusion_mel_demo --steps 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torchaudio

from sirencodec.diffusion_mel import DDPMMel


def pick_device(prefer: str | None) -> torch.device:
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mel DDPM demo (synthetic batches)")
    ap.add_argument("--steps", type=int, default=400, help="optimizer steps")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--ddpm-steps", type=int, default=100, help="diffusion timesteps T")
    ap.add_argument("--sample-every", type=int, default=200)
    args = ap.parse_args()

    device = pick_device(None if args.device == "auto" else args.device)
    print(f"device={device}")

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000,
        n_fft=1024,
        hop_length=320,
        n_mels=80,
    ).to(device)

    def synthetic_batch() -> torch.Tensor:
        w = torch.randn(args.batch_size, 16_000, device=device) * 0.15
        m = mel_tf(w).clamp(min=1e-5).log()
        return m

    model = DDPMMel(n_mels=80, timesteps=args.ddpm_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DDPMMel params: {n_params:.2f}M  |  T={args.ddpm_steps}")

    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        m = synthetic_batch()
        loss = model.loss(m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 50 == 0 or i == args.steps - 1:
            print(f"  step {i:5d}  loss={loss.item():.5f}")

        if args.sample_every > 0 and i > 0 and i % args.sample_every == 0:
            with torch.no_grad():
                m = synthetic_batch()
                x_synth = model.sample(torch.zeros_like(m))
                print(f"  [sample] mels shape={tuple(x_synth.shape)}  std={x_synth.std().item():.4f}")

    with torch.no_grad():
        m = synthetic_batch()
        x_synth = model.sample(torch.zeros_like(m))
    print(f"final sample: shape={tuple(x_synth.shape)}  mean={x_synth.mean().item():.4f}")


if __name__ == "__main__":
    main()
