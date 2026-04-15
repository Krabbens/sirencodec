#!/usr/bin/env python3
"""
Train conditional mel refiner on top of a frozen VocosVQCodec checkpoint.

Loss: L1(refined_mel, mel_orig) in log-mel domain (optionally + noise augmentation).
"""
import argparse
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from mel_refiner import MelRefinerNet
from train_vocos_vq import (
    VocosVQCodec,
    VocosVQConfig,
    AudioDataset,
    collate_fn,
)


def load_cfg_from_checkpoint(ckpt_path: str) -> VocosVQConfig:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "cfg" not in ckpt:
        raise ValueError(
            "Checkpoint has no 'cfg' key. Save a new codec checkpoint with train_vocos_vq.py "
            "(checkpoints now include cfg)."
        )
    names = {f.name for f in fields(VocosVQConfig)}
    merged = asdict(VocosVQConfig())
    merged.update(ckpt["cfg"])
    return VocosVQConfig(**{k: merged[k] for k in names})


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg_from_checkpoint(args.codec_checkpoint)
    codec = VocosVQCodec(cfg).to(device)
    ckpt = torch.load(args.codec_checkpoint, map_location=device)
    codec.load_state_dict(ckpt["model"], strict=True)
    codec.eval()
    for p in codec.parameters():
        p.requires_grad = False

    refiner = MelRefinerNet(n_mels=cfg.n_mels, hidden=args.hidden, n_layers=args.layers).to(device)
    opt = optim.AdamW(refiner.parameters(), lr=args.lr, betas=(0.8, 0.99))
    print(f"Refiner params: {sum(p.numel() for p in refiner.parameters()) / 1e6:.3f}M")

    manifest = os.path.join(args.data_dir, "master_manifest.jsonl")
    ds = AudioDataset(manifest, cfg.segment_length)
    n_train = int(len(ds) * 0.9)
    n_dev = len(ds) - n_train
    train_ds, dev_ds = torch.utils.data.random_split(ds, [n_train, n_dev])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    log_path = Path(args.log_tsv)
    log_f = open(log_path, "w")
    log_f.write("step\tloss\tlr\n")

    step = 0
    t0 = time.time()
    while step < args.steps:
        for audio, _ in train_loader:
            if step >= args.steps:
                break
            audio = audio.to(device)
            with torch.no_grad():
                mel_orig, coarse, _, _, _ = codec.encode_to_coarse_mel(audio)
            coarse_d = coarse.detach()
            if args.noise_std > 0:
                coarse_d = coarse_d + torch.randn_like(coarse_d) * args.noise_std
            delta = refiner(coarse_d)
            refined = coarse_d + delta if args.residual else delta
            min_t = min(refined.shape[2], mel_orig.shape[2])
            loss = F.l1_loss(refined[:, :, :min_t], mel_orig[:, :, :min_t])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()

            if step % 200 == 0:
                lr = opt.param_groups[0]["lr"]
                print(f"Step {step}/{args.steps} loss={loss.item():.4f} lr={lr:.6f}")
                log_f.write(f"{step}\t{loss.item():.6f}\t{lr:.6f}\n")
                log_f.flush()
            step += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "refiner": refiner.state_dict(),
        "codec_cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
        "steps": args.steps,
    }, out_dir / "mel_refiner.pt")
    print(f"Saved {out_dir / 'mel_refiner.pt'} in {time.time()-t0:.1f}s")
    log_f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codec-checkpoint", type=str, required=True, help="train_vocos_vq checkpoint with cfg")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise on coarse mel (training)")
    p.add_argument("--no-residual", action="store_true", help="refined = delta only (ablation)")
    p.add_argument("--log-tsv", type=str, default="log_mel_refiner.tsv")
    p.add_argument("--out-dir", type=str, default="checkpoints_mel_refiner")
    args = p.parse_args()
    args.residual = not args.no_residual
    run(args)


if __name__ == "__main__":
    main()
