#!/usr/bin/env python3
"""Mel spectrogram + optional WAV export from a train_pipeline checkpoint (codec_step*.pt).

Writes PNG (--out) and, by default, *_orig.wav / *_recon.wav (same stem, --no-wav to skip).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sirencodec.core.train import (  # noqa: E402
    AudioCodec,
    AudioCodecASPK,
    AudioCodecB,
    AudioCodecC,
    AudioCodecD,
    CodecConfig,
    compute_abr_depth_heuristic,
    get_curriculum_codebooks,
)


def _strip_compile_prefix(state_dict: dict) -> dict:
    sample = next(iter(state_dict.keys()), "")
    if not sample.startswith("_orig_mod."):
        return state_dict
    return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}


def build_codec(cfg: CodecConfig, device: torch.device):
    arch = cfg.architecture
    if arch == "arch-a-spk":
        return AudioCodecASPK(cfg).to(device)
    if arch == "arch-b-v1":
        return AudioCodecB(cfg).to(device)
    if arch == "arch-c-v1":
        return AudioCodecC(cfg).to(device)
    if arch == "arch-d-v1":
        return AudioCodecD(cfg).to(device)
    return AudioCodec(cfg).to(device)


def load_wave(path: Path, sr: int, seconds: float, device: torch.device):
    # soundfile: avoids torchaudio>=2.9 default path that requires optional torchcodec
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    data = np.mean(data, axis=1) if data.ndim == 2 else np.asarray(data, dtype=np.float32)
    wav = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
    if sample_rate != sr:
        wav = torchaudio.functional.resample(wav, sample_rate, sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    seg = int(sr * seconds)
    if wav.shape[1] > seg:
        start = (wav.shape[1] - seg) // 2
        wav = wav[:, start : start + seg]
    elif wav.shape[1] < seg:
        wav = torch.nn.functional.pad(wav, (0, seg - wav.shape[1]))
    return wav.unsqueeze(0).to(device)


def synthetic_segment(sr: int, seconds: float, device: torch.device):
    t = int(sr * seconds)
    x = torch.linspace(0, 1, t)
    wave = 0.3 * torch.sin(2 * 3.14159 * (200 + 600 * x) * x * 3)
    wave = wave + 0.15 * torch.sin(2 * 3.14159 * 440 * x * 3)
    wave = wave + 0.05 * torch.randn(t)
    return wave.unsqueeze(0).unsqueeze(0).to(device)


def forward_codec(codec, cfg: CodecConfig, x: torch.Tensor, train_step: int):
    arch = cfg.architecture
    if cfg.curriculum_enabled and arch not in ("arch-b-v1",):
        n_cb = get_curriculum_codebooks(train_step, cfg.curriculum_schedule)
    else:
        n_cb = cfg.n_codebooks
    if arch == "arch-c-v1":
        return codec(x, step=train_step)
    if arch in ("arch-a-spk", "arch-b-v1"):
        return codec(x)
    return codec(x, n_codebooks=n_cb)


def write_wav(path: Path, tensor: torch.Tensor, sr: int) -> None:
    """tensor: [1, T] or [T] mono float in ~[-1, 1]."""
    x = tensor.detach().cpu().float().numpy()
    x = np.clip(x, -1.0, 1.0)
    if x.ndim == 2:
        x = x.T
    else:
        x = x[:, np.newaxis]
    sf.write(str(path), np.ascontiguousarray(x), sr, subtype="PCM_16")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/codec_step99999.pt",
        help="train_pipeline checkpoint (codec + config)",
    )
    p.add_argument("--out", type=str, default="spectrogram_pipeline.png")
    p.add_argument("--manifest", type=str, default="data/master_manifest.jsonl")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step", type=int, default=-1, help="curriculum step (default: ckpt step)")
    p.add_argument(
        "--no-wav",
        action="store_true",
        help="Do not write *_orig.wav / *_recon.wav next to the PNG",
    )
    p.add_argument(
        "--no-depth-plot",
        action="store_true",
        help="When ABR is enabled in config, skip the per-frame depth curve panel",
    )
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if not isinstance(cfg, CodecConfig):
        print("Checkpoint config must be CodecConfig (train_pipeline).", file=sys.stderr)
        sys.exit(1)

    codec = build_codec(cfg, device)
    codec.load_state_dict(_strip_compile_prefix(ckpt["codec"]), strict=False)
    codec.eval()

    train_step = args.step if args.step >= 0 else int(ckpt.get("step", 0))
    sr = cfg.sample_rate
    manifest = _ROOT / args.manifest
    if manifest.is_file():
        with open(manifest) as f:
            entries = [json.loads(line) for line in f]
        rng = random.Random(args.seed)
        path = Path(rng.choice(entries)["path"])
        if not path.is_file():
            path = _ROOT / path
        print(f"Audio: {path}")
        x = load_wave(path, sr, args.seconds, device)
    else:
        print("No manifest — synthetic segment")
        x = synthetic_segment(sr, args.seconds, device)

    with torch.no_grad():
        out = forward_codec(codec, cfg, x, train_step)
        recon = out[0]

    depth_latent: np.ndarray | None = None
    if getattr(cfg, "abr_enabled", False) and not args.no_depth_plot and isinstance(codec, AudioCodec):
        with torch.no_grad():
            z = codec.encoder(x)
            if getattr(cfg, "abr_mode", "full") == "learned" and getattr(codec, "depth_policy", None) is not None:
                d, _, _ = codec.depth_policy(z, cfg, hard=True)
                depth_latent = d.float().cpu().numpy()[0]
            elif cfg.abr_mode == "heuristic":
                depth_latent = compute_abr_depth_heuristic(z, cfg).float().cpu().numpy()[0]
            else:
                depth_latent = np.full(z.size(2), float(cfg.n_codebooks), dtype=np.float32)

    orig = x.squeeze(0).cpu()
    deg = recon.squeeze(0).cpu()
    ml = min(orig.shape[-1], deg.shape[-1])
    orig = orig[..., :ml]
    deg = deg[..., :ml]

    n_mels = cfg.n_mels
    n_fft = cfg.n_fft
    hop = cfg.hop_length
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, f_min=50, f_max=sr // 2 - 1
    )
    mo = mel_tf(orig)
    md = mel_tf(deg)
    mel_o = torch.log(mo.squeeze().clamp(min=1e-5)).numpy()
    mel_d = torch.log(md.squeeze().clamp(min=1e-5)).numpy()
    mel_vmin = float(min(mel_o.min(), mel_d.min()))
    mel_vmax = float(max(mel_o.max(), mel_d.max()))

    # Peak-normalize waveforms for comparable Y scale (loudness difference stays in the signal otherwise).
    def peak_norm(w: torch.Tensor) -> torch.Tensor:
        p = w.abs().max().clamp(min=1e-8)
        return w / p

    ow = peak_norm(orig.squeeze()).numpy()
    dw = peak_norm(deg.squeeze()).numpy()

    if depth_latent is not None:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 0.55], hspace=0.35, wspace=0.25)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        axd = fig.add_subplot(gs[2, :])
        axes = np.array([[ax00, ax01], [ax10, ax11]])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axd = None

    axes[0, 0].plot(ow, linewidth=0.5)
    axes[0, 0].set_title("Original (peak-norm)")
    axes[0, 0].set_ylim(-1.05, 1.05)
    axes[0, 1].plot(dw, linewidth=0.5)
    axes[0, 1].set_title("Reconstructed (peak-norm)")
    axes[0, 1].set_ylim(-1.05, 1.05)
    im0 = axes[1, 0].imshow(
        mel_o, aspect="auto", origin="lower", cmap="magma", vmin=mel_vmin, vmax=mel_vmax
    )
    axes[1, 0].set_title("Mel (orig)")
    plt.colorbar(im0, ax=axes[1, 0], fraction=0.046)
    im1 = axes[1, 1].imshow(
        mel_d, aspect="auto", origin="lower", cmap="magma", vmin=mel_vmin, vmax=mel_vmax
    )
    axes[1, 1].set_title("Mel (recon)")
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    if axd is not None and depth_latent is not None:
        axd.plot(np.arange(len(depth_latent)), depth_latent, linewidth=0.8, color="steelblue")
        axd.set_ylabel("RVQ depth")
        axd.set_xlabel("Latent frame")
        axd.set_ylim(0.5, float(cfg.n_codebooks) + 0.5)
        axd.set_title(f"ABR depth ({cfg.abr_mode})")
        axd.grid(True, alpha=0.3)
    step = ckpt.get("step", "?")
    fig.suptitle(f"{ckpt_path.name} | step={step} | arch={cfg.architecture}", fontsize=11)
    plt.tight_layout()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if not args.no_wav:
        wav_orig = out_path.with_name(f"{out_path.stem}_orig.wav")
        wav_recon = out_path.with_name(f"{out_path.stem}_recon.wav")
        write_wav(wav_orig, orig, sr)
        write_wav(wav_recon, deg, sr)
        print(f"Saved: {wav_orig}")
        print(f"Saved: {wav_recon}")


if __name__ == "__main__":
    main()
