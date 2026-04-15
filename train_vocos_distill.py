#!/usr/bin/env python3
"""
Knowledge distillation: small student maps log-mel [B,100,T] -> waveform to match frozen Vocos teacher.

Target ~3M params. Loss: L1(student, teacher) + multi-resolution STFT.
"""
import argparse
import os
import time
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

from train_vocos_vq import MelExtractor, AudioDataset, collate_fn




class StudentVocoder(nn.Module):
    """~2–4M params: mel [B, n_mels, T] -> waveform [B, 1, T * 256] (8× stride-2)."""

    def __init__(self, n_mels=100, base=192):
        super().__init__()
        self.in_p = nn.Conv1d(n_mels, base, kernel_size=7, padding=3)
        # 9 channel sizes for 8 upsampling stages (2^8 = 256)
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
        """mel [B, n_mels, T] -> [B, 1, T*256]"""
        x = F.leaky_relu(self.in_p(mel), 0.1)
        for b in self.blocks:
            x = b(x)
        return x.unsqueeze(1)


def run(args):
    if not HAS_VOCOS:
        raise SystemExit("pip install vocos")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hop = 24000 // 94  # match thesis mel rate
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
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
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
            mel = mel_ext(audio)  # [B, n_mels, T]
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
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base", type=int, default=384, help="Channels (~0.3M @192, ~1–3M @384–512)")
    p.add_argument("--vocos-pretrained", type=str, default="charactr/vocos-mel-24khz")
    p.add_argument("--log-tsv", type=str, default="log_vocos_distill.tsv")
    p.add_argument("--out-dir", type=str, default="checkpoints_student_vocoder")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
