#!/usr/bin/env python3
"""Secondary trainers: mel refiner + student vocoder distillation."""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from vocos import Vocos
    HAS_VOCOS = True
except ImportError:
    HAS_VOCOS = False

from .core.train_vocos_vq import (
    AudioDataset,
    MelExtractor,
    VocosVQCodec,
    VocosVQConfig,
    collate_fn,
)
from .extras import MelRefinerNet


def load_cfg_from_checkpoint(ckpt_path: str) -> VocosVQConfig:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "cfg" not in ckpt:
        raise ValueError(
            "Checkpoint has no 'cfg' key. Save a codec checkpoint with train_vocos_vq first."
        )
    names = {f.name for f in fields(VocosVQConfig)}
    merged = asdict(VocosVQConfig())
    merged.update(ckpt["cfg"])
    return VocosVQConfig(**{k: merged[k] for k in names})


def run_refiner(args):
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
    train_ds, _dev = torch.utils.data.random_split(ds, [n_train, n_dev])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
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
    torch.save(
        {
            "refiner": refiner.state_dict(),
            "codec_cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
            "steps": args.steps,
        },
        out_dir / "mel_refiner.pt",
    )
    print(f"Saved {out_dir / 'mel_refiner.pt'} in {time.time()-t0:.1f}s")
    log_f.close()


class StudentVocoder(nn.Module):
    """~2–4M params: mel [B, n_mels, T] -> waveform [B, 1, T * 256] (8× stride-2)."""

    def __init__(self, n_mels=100, base=192):
        super().__init__()
        self.in_p = nn.Conv1d(n_mels, base, kernel_size=7, padding=3)
        chs = [base, 128, 96, 64, 48, 32, 24, 16, 1]
        self.blocks = nn.ModuleList()
        for i in range(8):
            cin, cout = chs[i], chs[i + 1]
            self.blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.1) if i < 7 else nn.Identity(),
                )
            )

    def forward(self, mel):
        x = F.leaky_relu(self.in_p(mel), 0.1)
        for b in self.blocks:
            x = b(x)
        return x.unsqueeze(1)


def run_distill(args):
    if not HAS_VOCOS:
        raise SystemExit("pip install vocos")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hop = 24000 // 94
    mel_ext = MelExtractor(n_mels=100, hop_length=hop, sample_rate=24000).to(device)
    teacher = Vocos.from_pretrained(args.vocos_pretrained).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student = StudentVocoder(n_mels=100, base=args.base).to(device)
    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Student params: {n_params:.2f}M (target ~3M; tune --base)")

    opt = optim.AdamW(student.parameters(), lr=args.lr, betas=(0.8, 0.99))

    manifest = os.path.join(args.data_dir, "master_manifest.jsonl")
    ds = AudioDataset(manifest, 24000)
    n_train = int(len(ds) * 0.9)
    dev = len(ds) - n_train
    train_ds, _ = torch.utils.data.random_split(ds, [n_train, dev])
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    step = 0
    data_iter = iter(loader)
    log_f = open(args.log_tsv, "w")
    log_f.write("step\tloss_l1\tloss_hf\n")

    while step < args.steps:
        try:
            audio, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            audio, _ = next(data_iter)
        audio = audio.to(device)
        with torch.no_grad():
            mel = mel_ext(audio)
        with torch.no_grad():
            t_out = teacher.backbone(mel)
            wav_t = teacher.head(t_out)

        s_out = student(mel)
        if wav_t.dim() == 2:
            wav_2d = wav_t
        else:
            wav_2d = wav_t.squeeze(1)
        s_flat = s_out.squeeze(1)
        if s_flat.dim() > 2:
            s_flat = s_flat.reshape(s_flat.shape[0], -1)
        w_flat = wav_2d
        if w_flat.dim() > 2:
            w_flat = w_flat.reshape(w_flat.shape[0], -1)
        if s_flat.shape[1] != w_flat.shape[1]:
            s_flat = F.interpolate(
                s_flat.unsqueeze(1), size=w_flat.shape[1], mode="linear", align_corners=False
            ).squeeze(1)
        L = min(s_flat.shape[1], w_flat.shape[1])
        if L >= 8:
            s_flat = s_flat[:, :L]
            w_flat = w_flat[:, :L]
            loss_l1 = F.l1_loss(s_flat, w_flat)
            d_s = s_flat[:, 1:] - s_flat[:, :-1]
            d_t = w_flat[:, 1:] - w_flat[:, :-1]
            loss_hf = F.l1_loss(d_s, d_t)
            loss = loss_l1 + 0.1 * loss_hf

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            if step % 200 == 0:
                print(f"Step {step} L1={loss_l1.item():.4f} HF={loss_hf.item():.4f}")
                log_f.write(f"{step}\t{loss_l1.item():.6f}\t{loss_hf.item():.6f}\n")
                log_f.flush()
        step += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"student": student.state_dict(), "hop": hop}, out / "student_vocoder.pt")
    print(f"Saved {out / 'student_vocoder.pt'}")
    log_f.close()


def main():
    p = argparse.ArgumentParser(description="Mel refiner or vocoder distillation")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("mel_refiner", help="Train conditional mel refiner on frozen codec")
    r.add_argument("--codec-checkpoint", type=str, required=True)
    r.add_argument("--data-dir", type=str, default="data")
    r.add_argument("--steps", type=int, default=20000)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--hidden", type=int, default=128)
    r.add_argument("--layers", type=int, default=10)
    r.add_argument("--noise-std", type=float, default=0.0)
    r.add_argument("--no-residual", action="store_true")
    r.add_argument("--log-tsv", type=str, default="log_mel_refiner.tsv")
    r.add_argument("--out-dir", type=str, default="checkpoints_mel_refiner")

    d = sub.add_parser("distill", help="Distill small waveform decoder from Vocos")
    d.add_argument("--data-dir", type=str, default="data")
    d.add_argument("--steps", type=int, default=50000)
    d.add_argument("--batch-size", type=int, default=16)
    d.add_argument("--lr", type=float, default=1e-4)
    d.add_argument("--base", type=int, default=384)
    d.add_argument("--vocos-pretrained", type=str, default="charactr/vocos-mel-24khz")
    d.add_argument("--log-tsv", type=str, default="log_vocos_distill.tsv")
    d.add_argument("--out-dir", type=str, default="checkpoints_student_vocoder")

    args = p.parse_args()
    if args.cmd == "mel_refiner":
        args.residual = not args.no_residual
        run_refiner(args)
    else:
        run_distill(args)


if __name__ == "__main__":
    main()
