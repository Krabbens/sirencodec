#!/usr/bin/env python3
"""Run CUDA/PyTorch inference for a converted SirenCodec checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sirencodec.config import Config, effective_codebook_sizes, encoder_time_stride, nominal_rvq_kbps  # noqa: E402
from sirencodec.cuda.codec import CUDACodec  # noqa: E402
from sirencodec.eval_metrics import quality_metrics_16k  # noqa: E402


def load_audio_mono(path: Path, target_sr: int, max_seconds: float | None) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if sr != target_sr:
        n_new = max(1, int(round(wav.size * target_sr / sr)))
        wav = np.interp(np.linspace(0, wav.size - 1, num=n_new), np.arange(wav.size), wav).astype(np.float32)
    if max_seconds is not None:
        wav = wav[: int(round(float(max_seconds) * target_sr))]
    peak = max(float(np.max(np.abs(wav))) if wav.size else 0.0, 1e-5)
    return (wav / peak).astype(np.float32)


def load_checkpoint(path: Path, device: torch.device) -> tuple[CUDACodec, Config]:
    blob = torch.load(path, map_location=device, weights_only=False)
    if "model" not in blob or "config" not in blob:
        raise RuntimeError(f"checkpoint must contain 'model' and 'config': {path}")
    cfg_blob = dict(blob["config"])
    if isinstance(cfg_blob.get("data_dir"), str):
        cfg_blob["data_dir"] = Path(cfg_blob["data_dir"])
    cfg = Config(**cfg_blob)
    model = CUDACodec(cfg).to(device)
    model.load_state_dict(blob["model"], strict=True)
    model.eval()
    return model, cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="Converted CUDA/PyTorch .pt checkpoint")
    p.add_argument("-i", "--input", type=Path, required=True, help="Input wav/flac/ogg/mp3")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("infer_cuda_out"))
    p.add_argument("--max-seconds", type=float, default=8.0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg = load_checkpoint(args.checkpoint.resolve(), device)
    wav = load_audio_mono(args.input.resolve(), cfg.sample_rate, args.max_seconds)
    x = torch.from_numpy(wav).view(1, -1, 1).to(device)

    with torch.inference_mode():
        out = model.forward_full(x)
        recon_t = out[0]
        indices = out[4]
    recon = recon_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
    recon = recon[: wav.shape[0]]
    metrics = quality_metrics_16k(wav, recon)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    stem = args.input.stem
    orig_path = out_dir / f"{stem}_orig.wav"
    recon_path = out_dir / f"{stem}_recon.wav"
    metrics_path = out_dir / f"{stem}_metrics.json"
    sf.write(str(orig_path), np.clip(wav, -1.0, 1.0), cfg.sample_rate, subtype="PCM_16")
    sf.write(str(recon_path), np.clip(recon, -1.0, 1.0), cfg.sample_rate, subtype="PCM_16")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cb_sizes = effective_codebook_sizes(cfg)
    stride = encoder_time_stride(cfg)
    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"device:     {device}")
    print(f"input:      {args.input.resolve()} ({wav.shape[0] / cfg.sample_rate:.2f}s)")
    print(f"bitrate:    {nominal_rvq_kbps(cfg):.4f} kbps nominal ({sum(np.log2(cb_sizes)):.1f} bits/frame, stride={stride})")
    if indices is not None:
        usage = []
        for q, idx in enumerate(indices):
            u = int(torch.unique(idx.detach().cpu()).numel())
            usage.append(f"q{q}={u}/{cb_sizes[q]}")
        print(f"code usage: {' '.join(usage)}")
    print(
        "metrics:    "
        + " ".join(
            f"{k}={'na' if v is None else f'{float(v):.6f}'}"
            for k, v in metrics.items()
        )
    )
    print(f"wrote:      {orig_path}")
    print(f"            {recon_path}")
    print(f"            {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
