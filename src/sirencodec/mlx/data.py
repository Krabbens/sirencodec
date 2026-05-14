"""Synthetic and disk audio batches for MLX training."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx

from ..config import Config

# ═══════════════════════════════════════════════════════════════════════════
# 2. Synthetic + optional WAV data
# ═══════════════════════════════════════════════════════════════════════════
def _synth_f_hi(cfg: Config) -> float:
    limit = 0.45 * float(cfg.sample_rate)
    lowpass_hz = float(getattr(cfg, "audio_lowpass_hz", 0.0) or 0.0)
    if lowpass_hz > 0.0:
        limit = min(limit, lowpass_hz * 0.95)
    return max(90.0, min(2000.0, limit))


def synth_batch(cfg: Config, key: int) -> mx.array:
    """Mixture of sines + noise → [B, T, 1] float32."""
    mx.random.seed(key)
    b, t = cfg.batch, cfg.segment
    t_ax = mx.arange(t, dtype=mx.float32)[None, :, None] / float(cfg.sample_rate)
    f_hi = _synth_f_hi(cfg)
    freqs = mx.random.uniform(low=80.0, high=f_hi, shape=(b, 1, 1))
    phases = mx.random.uniform(low=0.0, high=6.28318, shape=(b, 1, 1))
    x = mx.sin(2.0 * math.pi * freqs * t_ax + phases)
    x = x + 0.05 * mx.random.normal(shape=(b, t, 1))
    # normalize per sample
    m = mx.max(mx.abs(x), axis=1, keepdims=True) + 1e-5
    return x / m


_SKIP_PATH_PARTS = frozenset(
    {
        "venv",
        ".venv",
        "site-packages",
        "node_modules",
        "__pycache__",
        ".git",
        ".eggs",
        ".tox",
    }
)


def _resample_np(wav, sr: int, target_sr: int):
    import numpy as np

    sr = int(sr)
    target_sr = int(target_sr)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if sr == target_sr or wav.size == 0:
        return wav.astype(np.float32, copy=False)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, target_sr)
        return resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    except Exception:
        n_new = max(1, int(round(wav.size * float(target_sr) / float(sr))))
        return np.interp(
            np.linspace(0, wav.size - 1, num=n_new, endpoint=True),
            np.arange(wav.size),
            wav,
        ).astype(np.float32)


def _lowpass_np(wav, sample_rate: int, cutoff_hz: float):
    import numpy as np

    cutoff_hz = float(cutoff_hz or 0.0)
    sample_rate = int(sample_rate)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    nyq = 0.5 * float(sample_rate)
    if cutoff_hz <= 0.0 or cutoff_hz >= nyq or wav.size == 0:
        return wav.astype(np.float32, copy=False)
    try:
        from scipy.signal import butter, sosfilt, sosfiltfilt

        sos = butter(8, cutoff_hz / nyq, btype="lowpass", output="sos")
        if wav.size > 256:
            return sosfiltfilt(sos, wav).astype(np.float32)
        return sosfilt(sos, wav).astype(np.float32)
    except Exception:
        spec = np.fft.rfft(wav)
        freqs = np.fft.rfftfreq(wav.size, d=1.0 / float(sample_rate))
        spec[freqs > cutoff_hz] = 0.0
        return np.fft.irfft(spec, n=wav.size).astype(np.float32)


def _maybe_lowpass_np(wav, cfg: Config):
    return _lowpass_np(wav, cfg.sample_rate, getattr(cfg, "audio_lowpass_hz", 0.0))


def _skip_audio_path(p: Path) -> bool:
    """Ignore venv, pip/scipy test fixtures, and other non-dataset audio under a broad ``data/`` tree."""
    if _SKIP_PATH_PARTS.intersection(p.parts):
        return True
    s = str(p).replace("\\", "/").lower()
    if "/site-packages/" in s or "/tests/data/" in s:
        return True
    return False


def _collect_audio_paths(root: Path) -> list[Path]:
    """Find audio files under ``root`` recursively."""
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if _skip_audio_path(p):
            continue
        out.append(p)
    out.sort(key=lambda p: str(p).lower())
    return out


def _load_audio_row_np(
    paths: list[Path],
    n: int,
    need: int,
    sample_rate: int,
    lowpass_hz: float,
    offset: int,
    row_i: int,
) -> tuple[int, object]:
    """Load and peak-normalize one training row; returns ``(row_index, wav[need] float32)``."""
    import numpy as np
    import soundfile as sf

    idx = (offset + row_i) % n
    wav = None
    sr = None
    for _ in range(min(n, 128)):
        p = paths[idx]
        try:
            wav, sr = sf.read(str(p), always_2d=True)
            break
        except Exception:
            idx = (idx + 1) % n
    if wav is None:
        raise RuntimeError(
            "Could not read any audio file (all failed open/read). "
            "Check --data-dir and that files are valid wav/flac/ogg/mp3."
        )
    if sr != sample_rate:
        wav = _resample_np(wav[:, 0], sr, sample_rate)
    else:
        wav = wav[:, 0].astype(np.float32)
    wav = _lowpass_np(wav, sample_rate, lowpass_hz)
    if wav.size < need:
        wav = np.pad(wav, (0, need - wav.size))
    wav = wav[:need]
    m = float(np.max(np.abs(wav))) + 1e-5
    return row_i, (wav[:need] / m).astype(np.float32)


def _load_audio_batch(cfg: Config, paths: list[Path], offset: int) -> mx.array:
    """Load rotating batches from disk (wav/flac/ogg/mp3); requires soundfile + numpy."""
    import numpy as np

    b, need = cfg.batch, cfg.segment
    n = len(paths)
    th = max(0, int(cfg.load_audio_threads))
    if th <= 1 or b <= 1:
        out = np.zeros((b, need, 1), dtype=np.float32)
        for i in range(b):
            ri, row = _load_audio_row_np(
                paths,
                n,
                need,
                cfg.sample_rate,
                cfg.audio_lowpass_hz,
                offset,
                i,
            )
            out[ri, :, 0] = row
        return mx.array(out)
    wk = min(th, b, 32)
    out = np.zeros((b, need, 1), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=wk) as ex:
        futs = [
            ex.submit(
                _load_audio_row_np,
                paths,
                n,
                need,
                cfg.sample_rate,
                cfg.audio_lowpass_hz,
                offset,
                i,
            )
            for i in range(b)
        ]
        for fut in futs:
            ri, row = fut.result()
            out[ri, :, 0] = row
    return mx.array(out)


def _load_audio_viz_clip(cfg: Config, paths: list[Path], step: int, n_samples: int) -> mx.array:
    """One normalized mono clip [1, n_samples, 1] from disk for spectrogram/wav export."""
    import numpy as np
    import soundfile as sf

    n = len(paths)
    idx = (step * 7919) % n
    wav = None
    sr = None
    for _ in range(min(n, 128)):
        p = paths[idx]
        try:
            wav, sr = sf.read(str(p), always_2d=True)
            break
        except Exception:
            idx = (idx + 1) % n
    if wav is None:
        raise RuntimeError("Could not read any file for spectrogram viz.")
    if sr != cfg.sample_rate:
        wav = _resample_np(wav[:, 0], sr, cfg.sample_rate)
    else:
        wav = wav[:, 0].astype(np.float32)
    wav = _maybe_lowpass_np(wav, cfg)
    if wav.size < n_samples:
        wav = np.pad(wav, (0, n_samples - wav.size))
    if wav.size > n_samples:
        start = (step * 11003) % max(1, wav.size - n_samples + 1)
        wav = wav[start : start + n_samples]
    else:
        wav = wav[:n_samples]
    m = float(np.max(np.abs(wav))) + 1e-5
    wav = (wav[:n_samples] / m).astype(np.float32)
    return mx.array(wav.reshape(1, n_samples, 1))


def synth_viz_clip(cfg: Config, key: int, n_samples: int) -> mx.array:
    """Synthetic [1, n_samples, 1] for viz when no --data-dir."""
    mx.random.seed(key)
    t_ax = mx.arange(n_samples, dtype=mx.float32)[None, :, None] / float(cfg.sample_rate)
    f_hi = _synth_f_hi(cfg)
    freqs = mx.random.uniform(low=80.0, high=f_hi, shape=(1, 1, 1))
    phases = mx.random.uniform(low=0.0, high=6.28318, shape=(1, 1, 1))
    x = mx.sin(2.0 * math.pi * freqs * t_ax + phases)
    x = x + 0.05 * mx.random.normal(shape=(1, n_samples, 1))
    m = mx.max(mx.abs(x), axis=1, keepdims=True) + 1e-5
    return x / m
