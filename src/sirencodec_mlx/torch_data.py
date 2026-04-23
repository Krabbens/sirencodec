"""Synthetic and disk audio batches for the PyTorch/CUDA trainer."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .config import Config


_SKIP_PATH_PARTS = frozenset({"venv", ".venv", "site-packages", "node_modules", "__pycache__", ".git", ".eggs", ".tox"})


def _skip_audio_path(p: Path) -> bool:
    if _SKIP_PATH_PARTS.intersection(p.parts):
        return True
    s = str(p).replace("\\", "/").lower()
    return "/site-packages/" in s or "/tests/data/" in s


def collect_audio_paths(root: Path) -> list[Path]:
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    out = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts and not _skip_audio_path(p)]
    out.sort(key=lambda p: str(p).lower())
    return out


def synth_batch(cfg: Config, key: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(key) & 0x7FFFFFFFFFFFFFFF)
    b, t = int(cfg.batch), int(cfg.segment)
    t_ax = torch.arange(t, device=device, dtype=torch.float32).view(1, t, 1) / float(cfg.sample_rate)
    freqs = torch.empty((b, 1, 1), device=device).uniform_(80.0, 2000.0, generator=g)
    phases = torch.empty((b, 1, 1), device=device).uniform_(0.0, 2.0 * math.pi, generator=g)
    x = torch.sin(2.0 * math.pi * freqs * t_ax + phases)
    x = x + 0.05 * torch.randn((b, t, 1), device=device, generator=g)
    return x / (x.abs().amax(dim=1, keepdim=True) + 1e-5)


def _load_audio_row_np(paths: list[Path], n: int, need: int, sample_rate: int, offset: int, row_i: int) -> tuple[int, np.ndarray]:
    import soundfile as sf

    idx = (offset + row_i) % n
    wav = None
    sr = None
    for _ in range(min(n, 128)):
        try:
            wav, sr = sf.read(str(paths[idx]), always_2d=True)
            break
        except Exception:
            idx = (idx + 1) % n
    if wav is None:
        raise RuntimeError("Could not read any audio file. Check --data-dir and audio validity.")
    wav = wav[:, 0].astype(np.float32)
    if wav.size < need:
        wav = np.pad(wav, (0, need - wav.size))
    if sr != sample_rate:
        t_new = np.linspace(0, 1, num=need, endpoint=False)
        wav = np.interp(t_new * wav.size, np.arange(wav.size), wav).astype(np.float32)
    else:
        wav = wav[:need]
    return row_i, (wav[:need] / (float(np.max(np.abs(wav))) + 1e-5)).astype(np.float32)


def load_audio_batch_cpu(cfg: Config, paths: list[Path], offset: int) -> torch.Tensor:
    b, need, n = int(cfg.batch), int(cfg.segment), len(paths)
    th = max(0, int(cfg.load_audio_threads))
    out = np.zeros((b, need, 1), dtype=np.float32)
    if th <= 1 or b <= 1:
        for i in range(b):
            ri, row = _load_audio_row_np(paths, n, need, cfg.sample_rate, offset, i)
            out[ri, :, 0] = row
    else:
        with ThreadPoolExecutor(max_workers=min(th, b, 32)) as ex:
            futs = [ex.submit(_load_audio_row_np, paths, n, need, cfg.sample_rate, offset, i) for i in range(b)]
            for fut in futs:
                ri, row = fut.result()
                out[ri, :, 0] = row
    return torch.from_numpy(out)


def load_audio_batch(cfg: Config, paths: list[Path], offset: int, device: torch.device) -> torch.Tensor:
    return load_audio_batch_cpu(cfg, paths, offset).to(device, non_blocking=device.type == "cuda")


def load_audio_viz_clip(cfg: Config, paths: list[Path], step: int, n_samples: int, device: torch.device) -> torch.Tensor:
    import soundfile as sf

    n = len(paths)
    idx = (step * 7919) % n
    wav = None
    sr = None
    for _ in range(min(n, 128)):
        try:
            wav, sr = sf.read(str(paths[idx]), always_2d=True)
            break
        except Exception:
            idx = (idx + 1) % n
    if wav is None:
        raise RuntimeError("Could not read any file for spectrogram viz.")
    wav = wav[:, 0].astype(np.float32)
    if wav.size < n_samples:
        wav = np.pad(wav, (0, n_samples - wav.size))
    if sr != cfg.sample_rate:
        t_new = np.linspace(0, 1, num=n_samples, endpoint=False)
        wav = np.interp(t_new * wav.size, np.arange(wav.size), wav).astype(np.float32)
    elif wav.size > n_samples:
        start = (step * 11003) % max(1, wav.size - n_samples + 1)
        wav = wav[start : start + n_samples]
    else:
        wav = wav[:n_samples]
    wav = (wav[:n_samples] / (float(np.max(np.abs(wav))) + 1e-5)).astype(np.float32)
    return torch.from_numpy(wav.reshape(1, n_samples, 1)).to(device)


def synth_viz_clip(cfg: Config, key: int, n_samples: int, device: torch.device) -> torch.Tensor:
    old_segment = cfg.segment
    old_batch = cfg.batch
    cfg.segment = int(n_samples)
    cfg.batch = 1
    try:
        return synth_batch(cfg, key, device)
    finally:
        cfg.segment = old_segment
        cfg.batch = old_batch
