#!/usr/bin/env python3
"""Load a train_vocos_vq checkpoint and save original + reconstructed WAV for listening."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Repo root + src for package imports
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sirencodec.core.train_vocos_vq import (  # noqa: E402
    VocosVQCodec,
    VocosVQConfig,
    _normalize_ckpt_state_dict,
)


def config_from_checkpoint(cfg_dict: dict) -> VocosVQConfig:
    d = asdict(VocosVQConfig())
    for k, v in cfg_dict.items():
        if k not in d:
            continue
        if k == "betas" and isinstance(v, list):
            d[k] = tuple(v)
        else:
            d[k] = v
    if d.get("nl_layers") is None:
        d["nl_layers"] = 1  # matches pre-nl_layers checkpoints (100→h→k MLP)
    return VocosVQConfig(**d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str, help="Path to codec_step*.pt")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: same folder as checkpoint / samples)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seconds", type=float, default=3.0, help="Clip length")
    p.add_argument(
        "--manifest",
        type=str,
        default="data/master_manifest.jsonl",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Filename prefix (default: experiment folder name, e.g. nl4_fsq4_94fps_warm15k_50k)",
    )
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_checkpoint(ckpt["cfg"])
    model = VocosVQCodec(cfg)
    model.load_state_dict(_normalize_ckpt_state_dict(ckpt["model"]), strict=True)
    model.to(device)
    model.eval()

    sr = cfg.sample_rate
    seg_len = int(sr * args.seconds)

    manifest = _ROOT / args.manifest
    with open(manifest) as f:
        entries = [json.loads(line) for line in f]

    rng = random.Random(args.seed)
    entry = rng.choice(entries)
    path = entry["path"]
    if not Path(path).is_file():
        path = str(_ROOT / path)

    wav, sample_rate = torchaudio.load(path, backend="soundfile")
    if sample_rate != sr:
        wav = torchaudio.functional.resample(wav, sample_rate, sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    total = wav.shape[1]
    if total > seg_len:
        start = (total - seg_len) // 2
        wav = wav[:, start : start + seg_len]
    elif total < seg_len:
        wav = torch.nn.functional.pad(wav, (0, seg_len - total))

    audio_in = wav.unsqueeze(0).to(device)
    with torch.no_grad():
        recon, *_ = model(audio_in)

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent.parent / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ckpt_path.stem
    if args.prefix:
        prefix = args.prefix
    elif ckpt_path.parent.name == "checkpoints":
        prefix = ckpt_path.parent.parent.name
    else:
        prefix = ckpt_path.stem
    base = f"{prefix}_{tag}"
    orig_path = out_dir / f"{base}_orig.wav"
    recon_path = out_dir / f"{base}_recon.wav"

    def _write_wav(path: Path, tensor: torch.Tensor) -> None:
        x = tensor.detach().cpu().float().numpy()
        x = np.clip(x, -1.0, 1.0)
        if x.ndim == 2:
            x = x.T  # [C, T] -> [T, C] for soundfile
        else:
            x = np.ascontiguousarray(x)
        sf.write(str(path), x, sr, subtype="PCM_16")

    _write_wav(orig_path, audio_in.squeeze(0))
    _write_wav(recon_path, recon.squeeze(0))

    step = ckpt.get("step", "?")
    print(f"Source: {path}")
    print(f"Checkpoint: {ckpt_path} (step {step})")
    print(f"Wrote: {orig_path}")
    print(f"Wrote: {recon_path}")


if __name__ == "__main__":
    main()
