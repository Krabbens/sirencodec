#!/usr/bin/env python3
"""
train_mlx.py — Karpathy-style single-file MLX neural audio codec (Apple Silicon).

Inlined stack: SEANet-ish encoder → **residual VQ** (EnCodec-style) → decoder,
time + **multi-scale** log-STFT loss + default **log-mel bin L1** only (``λ_mel_L1=0.12``, ``λ_mel_L2=0``); set ``--lambda-mel-l1 0`` to disable mel, optional WAV batches. No PyTorch / sirencodec.

  uv sync --extra mlx
  uv run python tools/train_mlx.py --steps 500 --data-dir data/mlx_smoke
  uv run python tools/train_mlx.py --steps 200000   # defaults to ``data/cv-corpus/**/*.mp3``
  (``--data-dir`` overrides the default local corpus path)
  PNG spectrograms (orig vs recon) every ``--spectrogram-every`` steps → ``--spectrogram-dir`` (default 1000).
  Waveform **cos** (mean cosine similarity, logged as %). Default ``--lambda-cos 0.15``; for **90%+** also try
  ``--cos-hinge 1.0 --cos-target 0.9`` (longer training helps).
  **Listen:** by default also writes ``step_NNNNNNNN_orig.wav`` / ``_recon.wav`` with each PNG; ``--no-save-audio`` to skip.
  **VQ diversity:** ``--lambda-marginal`` + ``--marginal-tau`` (adds ``λ·(log K − H_marg)``); keep τ small so ``marg_ent`` tracks **hard** collapse (else ``marg_ent≈log K`` while ``u≪K``). Long ``--marginal-boost-steps`` keeps λ_marg strong into mid-training. ``--vq-reset-collapse-frac`` should be **high enough** (e.g. 0.4) that dead rows refill when ``unique/K`` is only moderately low (0.08 only resets at <~10 codes).
  **STFT vs VQ:** default ``--lambda-stft 0.35`` and ``--stft-ramp-steps 12000`` (linear ramp from ``--stft-ramp-start`` to full weight) so early STFT does not wash out RVQ; decoder output is ``tanh``-bounded. Default STFT includes **2048-FFT**; ``--stft-hf-emphasis`` frequency-weights log-mag L1 toward Nyquist (0=flat). Optional ``--stft-scale-weights W1,W2,…`` (same length as ``--stft-scales``) reweights mag/grad/cos STFT terms toward larger FFTs, e.g. ``1024,256;2048,512;4096,1024`` with ``1,1.75,2.5``.
  **Spectral sharpness / prążki harmoniczne:** ``--lambda-stft-grad W`` (default **0.08**) adds ``W·λ_stft_eff`` times a **weighted** mean L1 on ∂/∂f and ∂/∂t of log-mag STFT (default **∂/∂f twice** ∂/∂t via ``--stft-grad-freq-weight`` / ``--stft-grad-time-weight``) — stronger ∂/∂f pushes horizontal harmonic stripes; tune ``W`` in ~0.05–0.12.
  **Spectral cosine:** ``--lambda-stft-cos W`` adds ``W·λ_stft_eff`` times the mean ``(1−cos)`` between flattened log-mag STFTs (per scale, then averaged). Waveform cosine is ``--lambda-cos`` (separate).
  **Spectrogram length:** default ``--spectrogram-seconds 8`` for PNG/WAV (set ``0`` to use the training batch length).
  **Checkpoints:** ``--checkpoint-every 10000`` (default) writes ``codec_step{N}.npz`` under ``--checkpoint-dir``.
  **Resume:** ``--resume`` loads weights; continues from step ``N+1``. Full ``.safetensors`` resume restores optimizer + optional ``lr_plateau`` meta. ``--lr-schedule none`` keeps constant ``--lr``; ``plateau`` reduces LR when EMA train loss stalls; ``cosine`` decays by step.
  **Step time:** single codec forward/backward per micro-batch + multi-scale STFT losses (SC, complex L1, log-mag L1, optional gradient/cosine). **Throughput:** ``--load-audio-threads T`` parallelizes per-sample disk decode (``T≈8–16`` on Libri); ``--prefetch-audio`` overlaps the next micro-batch load (off when ``--grad-accum-steps`` > 1). **Gradient accumulation:** ``--grad-accum-steps K`` + small ``--batch`` = ``K`` backward passes per optimizer step, gradient average = effective batch ``B·K`` (Libri offset matches ``B·K`` samples per step). Mitigations: ``--fast`` / fewer ``--stft-scales``, lower ``--lambda-stft-grad`` or 0, ``--lambda-mel-l1 0`` (or lower), smaller ``--batch``. Heavy: ``--vq-reset-every`` (CPU), long PNG/WAV every ``--spectrogram-every``.
  **Architecture:** 8 stride-2 stages (256× time), default **latent_dim 512**; default **3× residual VQ** with **K=32** → **15 bits/latent-frame** × **62.5 Hz** ≈ **938 b/s** indices (<1 kb/s) @ 16 kHz. Optional ``--codebook-sizes`` / ``--codebook-size``. Optional ``--stride1-blocks-per-scale``.
  **Latent temporal context:** ``--latent-temporal-depth N`` adds N residual dilated Conv1d blocks **before** RVQ; ``--latent-temporal-post-depth M`` adds M blocks **after** RVQ (before decoder).
  **Spectral (GAN-free):** high-quality recipe without a discriminator. Combine log-mag L1 (``--lambda-stft``) with **Spectral Convergence** (``--lambda-sc``; Parallel WaveGAN) and **Complex STFT L1** (``--lambda-complex-stft``; captures phase). Optional sharpness terms: ``--lambda-stft-grad`` (∂/∂f, ∂/∂t log-mag) and ``--lambda-stft-cos`` (spectral cosine). Multi-scale via ``--stft-scales`` / ``--stft-scale-weights``.
  **Per-stage K:** ``--codebook-sizes 256,128,64`` (length = ``--n-codebooks``) for decreasing codebooks; default uniform ``--codebook-size``. Nominal bitrate uses ``sum_i log2(K_i)`` per latent frame.
  **Same nominal bits** when uniform: ``n_q·log2(K)`` per frame; **cosine VQ** (``--vq-cosine``) assigns by **direction** and scales ``z_q`` to ``‖residual‖``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .. import adaptive as adaptive_mod
from .. import entropy_coding as entropy_coding_mod
from .. import eval_metrics as eval_metrics_mod
from .. import streaming as streaming_mod

# ─────────────────────────────────────────────────────────────────────────────
# 0. MLX import (fail fast with install hint)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import Adam, cosine_decay, join_schedules, linear_schedule
    from mlx.utils import tree_flatten, tree_map, tree_reduce
except ImportError:
    print("MLX is not installed. On Apple Silicon: uv sync --extra mlx", file=sys.stderr)
    sys.exit(1)


# Split modules: config / data / losses / codec (see package layout).
from .codec import MLXCodec, VectorQuantizerStage, _turbo_rotation
from ..config import (
    DEFAULT_STFT_SCALES,
    FAST_STFT_SCALES,
    Config,
    argparse_defaults_from_config,
    effective_codebook_sizes,
    encoder_time_stride,
    mean_log_codebook_for_entropy,
    nominal_rvq_kbps,
    parse_codebook_sizes_arg,
    parse_float_list_arg,
    parse_positive_int_list_arg,
    parse_stft_scales_arg,
    parse_stft_scale_weights_arg,
)
from .data import (
    _collect_audio_paths,
    _load_audio_batch,
    _load_audio_viz_clip,
    synth_batch,
    synth_viz_clip,
)
from .losses import (
    band_branch_l1_loss,
    band_l1_loss,
    harmonic_f0_voicing_loss,
    high_frequency_complex_stft_l1,
    high_frequency_stft_terms,
    mel_filterbank_mx,
    mel_log_bin_losses,
    multi_stft_all_terms,
    multi_stft_complex_l1,
    multi_stft_loss,
    multi_stft_mag_l1_linear,
    multi_stft_stationary_line_loss,
    multi_stft_spectral_convergence,
    multi_stft_spectral_terms,
)
from .kernels import batch_mean_cosine_metric


# 5. Train step (closure over batch)
# ═══════════════════════════════════════════════════════════════════════════
def effective_lambda_stft(step: int, cfg: Config) -> float:
    """Scale ``lambda_stft`` early so VQ + L1 shape the latent before full spectrogram pressure."""
    if cfg.stft_ramp_steps <= 0:
        return cfg.lambda_stft
    t = min(1.0, float(step) / float(cfg.stft_ramp_steps))
    m = cfg.stft_ramp_start_frac
    scale = m + (1.0 - m) * t
    return cfg.lambda_stft * scale


def spectral_loss_step_multiplier(step: int, cfg: Config) -> float:
    """Sparse spectral schedule: active full-STFT steps are weighted by their cadence."""
    every = max(1, int(getattr(cfg, "spectral_loss_every", 1)))
    if every <= 1:
        return 1.0
    return float(every) if int(step) % every == 0 else 0.0


def effective_hf_mult(step: int, cfg: Config) -> float:
    """Delay/ramp high-frequency pressure independently from the broad STFT ramp."""
    start = max(0, int(getattr(cfg, "hf_start_step", 0)))
    ramp = max(0, int(getattr(cfg, "hf_ramp_steps", 0)))
    if int(step) < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, float(int(step) - start) / float(ramp))


def _eval_loss_and_grad_tree(loss: mx.array, grads) -> None:
    """Materialize backward: MLX keeps ``grads`` lazy until evaluated; ``opt.update`` needs real values."""
    flat = tree_flatten(grads)
    arrs = [a for _, a in flat]
    if arrs:
        mx.eval(loss, *arrs)
    else:
        mx.eval(loss)


def _eval_loss_grad_and_nonfinite(loss: mx.array, grads) -> tuple[float, bool]:
    """Evaluate loss, gradient tree, and non-finite flag in one MLX sync."""
    flat = tree_flatten(grads)
    arrs = [a for _, a in flat if isinstance(a, mx.array)]
    bad = mx.array(False)
    for a in arrs:
        bad = mx.logical_or(bad, mx.any(mx.logical_or(mx.isnan(a), mx.isinf(a))))
    if arrs:
        mx.eval(loss, bad, *arrs)
    else:
        mx.eval(loss, bad)
    return float(loss.item()), bool(bad.item())


def _eval_grad_tree(grads) -> None:
    flat = tree_flatten(grads)
    arrs = [a for _, a in flat]
    if arrs:
        mx.eval(*arrs)


def _grad_tree_any_nonfinite(grads) -> bool:
    """Single sync: avoid per-tensor ``mx.eval`` + ``.item()`` (was dominating step time)."""
    bad = mx.array(False)
    for _, a in tree_flatten(grads):
        if not isinstance(a, mx.array):
            continue
        bad = mx.logical_or(bad, mx.any(mx.logical_or(mx.isnan(a), mx.isinf(a))))
    mx.eval(bad)
    return bool(bad.item())


def clip_gradients_global_norm(grads, max_norm: float):
    """Scale gradient tree in-place topology so combined L2 norm ≤ ``max_norm`` (MLX lazy arrays)."""
    if max_norm <= 0:
        return grads
    norm_sq = tree_reduce(
        lambda acc, g: acc + mx.sum(g * g),
        grads,
        mx.array(0.0),
        is_leaf=lambda t: isinstance(t, mx.array),
    )
    norm = mx.sqrt(norm_sq + 1e-16)
    coef = mx.minimum(mx.array(1.0, dtype=mx.float32), mx.array(float(max_norm), dtype=mx.float32) / norm)
    return tree_map(
        lambda g: g * coef,
        grads,
        is_leaf=lambda t: isinstance(t, mx.array),
    )


def _grad_tree_scale(grads, scale: float):
    """Return a new tree ``scale * grads`` (only ``mx.array`` leaves)."""
    s = float(scale)
    return tree_map(lambda g: g * s, grads, is_leaf=lambda t: isinstance(t, mx.array))


def _grad_tree_add(a, b):
    """Elementwise ``a + b`` on two gradient trees."""
    return tree_map(lambda x, y: x + y, a, b, is_leaf=lambda t: isinstance(t, mx.array))


def effective_lambda_marginal(step: int, cfg: Config) -> float:
    """Stronger batch-marginal diversity weight at the start (fights fast post-init index collapse)."""
    if cfg.lambda_marginal <= 0 or cfg.marginal_boost_steps <= 0:
        return cfg.lambda_marginal
    m = max(1.0, float(cfg.marginal_boost_mult))
    t = min(1.0, float(step) / float(cfg.marginal_boost_steps))
    scale = m + (1.0 - m) * t  # m → 1.0
    return cfg.lambda_marginal * scale


def batch_mean_cosine(orig: mx.array, recon: mx.array) -> mx.array:
    """Mean cosine similarity in [-1, 1] over batch (each item flattened)."""
    b = orig.shape[0]
    o = orig.reshape(b, -1)
    r = recon.reshape(b, -1)
    dot = mx.sum(o * r, axis=1)
    no = mx.sqrt(mx.sum(o * o, axis=1) + 1e-8)
    nr = mx.sqrt(mx.sum(r * r, axis=1) + 1e-8)
    cos = dot / (no * nr)
    return mx.mean(cos)


def batch_mean_cosine_for_metrics(orig: mx.array, recon: mx.array, cfg: Config) -> mx.array:
    """Metric cosine; custom Metal is used only when cosine is not in the loss."""
    if cfg.lambda_cos > 0 or cfg.cos_hinge > 0:
        return batch_mean_cosine(orig, recon)
    return batch_mean_cosine_metric(orig, recon)


def batch_neg_log_si_sdr(orig: mx.array, recon: mx.array, eps: float = 1e-8) -> mx.array:
    """Differentiable negative log SI-SDR ratio with a polarity gate."""
    b = orig.shape[0]
    o = orig.reshape(b, -1).astype(mx.float32)
    r = recon.reshape(b, -1).astype(mx.float32)
    o = o - mx.mean(o, axis=1, keepdims=True)
    r = r - mx.mean(r, axis=1, keepdims=True)
    dot = mx.sum(r * o, axis=1, keepdims=True)
    target = dot * o / (mx.sum(o * o, axis=1, keepdims=True) + eps)
    noise = r - target
    ratio = (mx.sum(target * target, axis=1) + eps) / (mx.sum(noise * noise, axis=1) + eps)
    dot_cos = mx.sum(o * r, axis=1)
    no = mx.sqrt(mx.sum(o * o, axis=1) + eps)
    nr = mx.sqrt(mx.sum(r * r, axis=1) + eps)
    cos = dot_cos / (no * nr)
    polarity_gate = mx.maximum(mx.sigmoid(8.0 * cos), eps)
    return -mx.mean(mx.log(mx.maximum(ratio, eps)) + mx.log(polarity_gate))


def batch_preemph_l1(orig: mx.array, recon: mx.array, coef: float = 0.97) -> mx.array:
    """L1 after first-order pre-emphasis; cheap high-frequency waveform anchor."""
    if orig.shape[1] < 2 or recon.shape[1] < 2:
        return mx.array(0.0)
    c = float(coef)
    orig_hp = orig[:, 1:, :] - c * orig[:, :-1, :]
    recon_hp = recon[:, 1:, :] - c * recon[:, :-1, :]
    return mx.mean(mx.abs(recon_hp - orig_hp))


def batch_multidelta_l1(orig: mx.array, recon: mx.array, lags: tuple[int, ...] = (1, 2, 4, 8)) -> mx.array:
    """Mean L1 over multiple finite-difference lags; cheap time-domain HF anchor."""
    t = int(orig.shape[1])
    total = mx.array(0.0)
    n = 0
    for lag in lags:
        lag_i = int(lag)
        if lag_i < 1 or lag_i >= t:
            continue
        od = orig[:, lag_i:, :] - orig[:, :-lag_i, :]
        rd = recon[:, lag_i:, :] - recon[:, :-lag_i, :]
        total = total + mx.mean(mx.abs(rd - od))
        n += 1
    if n == 0:
        return mx.array(0.0)
    return total / float(n)


def _select_eval_audio_paths(paths: list[Path], cfg: Config) -> list[Path]:
    """Stable holdout subset; deterministic so eval rows compare checkpoint-to-checkpoint."""
    n = max(1, int(getattr(cfg, "eval_clips", 1) or 1))
    if len(paths) <= n:
        return list(paths)
    rng = random.Random(int(getattr(cfg, "eval_seed", 0) or 0))
    idxs = sorted(rng.sample(range(len(paths)), n))
    return [paths[i] for i in idxs]


def _load_eval_audio_np(path: Path, cfg: Config):
    """Load one full/trimmed normalized mono eval waveform as NumPy float32."""
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if wav.size < 1:
        raise ValueError("empty audio")
    if int(sr) != int(cfg.sample_rate):
        n_out = max(1, int(round(wav.size * float(cfg.sample_rate) / float(sr))))
        if wav.size == 1:
            wav = np.repeat(wav, n_out).astype(np.float32)
        else:
            x_old = np.linspace(0.0, 1.0, num=wav.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            wav = np.interp(x_new, x_old, wav).astype(np.float32)
    eval_seconds = float(getattr(cfg, "eval_seconds", 0.0) or 0.0)
    if eval_seconds > 0:
        need = max(1, int(round(eval_seconds * float(cfg.sample_rate))))
        wav = wav[:need]
    peak = float(np.max(np.abs(wav))) + 1e-5
    return (wav / peak).astype(np.float32, copy=False)


def _infer_eval_waveform_np(model: MLXCodec, cfg: Config, wav):
    """Chunked codec inference for an eval waveform; returns recon and raw RVQ indices."""
    import numpy as np

    seg = max(1, int(cfg.segment))
    stride = max(1, int(encoder_time_stride(cfg)))
    outs = []
    code_rows: list[list[int]] | None = None if cfg.ae_only else [[] for _ in range(int(cfg.n_codebooks))]
    for start in range(0, int(wav.size), seg):
        chunk = wav[start : start + seg]
        valid = int(chunk.size)
        if valid < seg:
            chunk = np.pad(chunk, (0, seg - valid))
        x = mx.array(chunk.reshape(1, seg, 1).astype(np.float32, copy=False))
        y, _, _, _, idx_list = model.forward_full(x)
        to_eval: list[mx.array] = [y]
        if idx_list is not None:
            to_eval.extend(idx_list)
        mx.eval(*to_eval)
        y_np = np.array(y[0, :, 0], dtype=np.float32)
        outs.append(y_np[:valid])
        if idx_list is not None and code_rows is not None:
            valid_frames = max(1, int(math.ceil(float(valid) / float(stride))))
            for si, ix in enumerate(idx_list):
                if si >= len(code_rows):
                    code_rows.append([])
                row = np.array(ix).ravel()
                code_rows[si].extend(int(v) for v in row[:valid_frames].tolist())
    if not outs:
        return np.zeros((0,), dtype=np.float32), code_rows
    return np.concatenate(outs, axis=0).astype(np.float32, copy=False), code_rows


def _finite_mean(vals) -> float | None:
    xs: list[float] = []
    for v in vals:
        if v is None:
            continue
        fv = float(v)
        if math.isfinite(fv):
            xs.append(fv)
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def _fmt_eval_metric(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "na"
    fv = float(v)
    if not math.isfinite(fv):
        return "na"
    return f"{fv:.{digits}f}"


def _rvq_entropy_summary(cfg: Config, code_rows: list[list[int]] | None) -> tuple[float | None, float | None]:
    if not code_rows:
        return None, None
    sizes = effective_codebook_sizes(cfg)
    h_stages: list[float] = []
    for si, row in enumerate(code_rows):
        if not row:
            continue
        kk = sizes[si] if si < len(sizes) else sizes[-1]
        h_stages.append(entropy_coding_mod.empirical_cross_entropy_bits_per_symbol(row, kk))
    if not h_stages:
        return None, None
    h_mean = sum(h_stages) / float(len(h_stages))
    fps = float(cfg.sample_rate) / float(encoder_time_stride(cfg))
    return h_mean, h_mean * fps * float(len(h_stages))


def _holdout_eval_summary(model: MLXCodec, cfg: Config, eval_paths: list[Path]) -> dict[str, float | int | None]:
    rows: list[dict[str, float | None]] = []
    for p in eval_paths:
        try:
            orig = _load_eval_audio_np(p, cfg)
            recon, code_rows = _infer_eval_waveform_np(model, cfg, orig)
            n = min(int(orig.size), int(recon.size))
            if n < 256:
                raise ValueError("too short after load/infer")
            m = eval_metrics_mod.quality_metrics_16k(orig[:n], recon[:n])
            h_mean, idx_bps = _rvq_entropy_summary(cfg, code_rows)
            m["rvq_h_mean_bits_sym"] = h_mean
            m["empirical_index_bps"] = idx_bps
            rows.append(m)
        except Exception as e:
            print(f"  [eval-warn] {p}: {type(e).__name__}: {e}", flush=True)
    keys = (
        "si_sdr_db",
        "pesq_wb",
        "stoi",
        "lsd_db",
        "cos",
        "rvq_h_mean_bits_sym",
        "empirical_index_bps",
    )
    out: dict[str, float | int | None] = {"n": len(rows)}
    for k in keys:
        out[k] = _finite_mean(row.get(k) for row in rows)
    return out


def _append_eval_tsv(path: str, step: int, scope: str, summary: dict[str, float | int | None]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    need_hdr = not p.is_file()
    cols = (
        "step",
        "scope",
        "n",
        "sisdr_db",
        "pesq_wb",
        "stoi",
        "lsd_db",
        "cos",
        "rvq_h_mean_bits_sym",
        "empirical_index_bps",
    )
    with p.open("a") as f:
        if need_hdr:
            f.write("\t".join(cols) + "\n")
        f.write(
            "\t".join(
                [
                    str(int(step)),
                    scope,
                    str(int(summary.get("n") or 0)),
                    _fmt_eval_metric(summary.get("si_sdr_db")),
                    _fmt_eval_metric(summary.get("pesq_wb")),
                    _fmt_eval_metric(summary.get("stoi")),
                    _fmt_eval_metric(summary.get("lsd_db")),
                    _fmt_eval_metric(summary.get("cos")),
                    _fmt_eval_metric(summary.get("rvq_h_mean_bits_sym")),
                    _fmt_eval_metric(summary.get("empirical_index_bps"), digits=1),
                ]
            )
            + "\n"
        )


def _format_eval_summary(prefix: str, summary: dict[str, float | int | None]) -> str:
    return (
        f"  [{prefix}] n={int(summary.get('n') or 0)}  "
        f"SI-SDR={_fmt_eval_metric(summary.get('si_sdr_db'), 2)} dB  "
        f"PESQ_wb={_fmt_eval_metric(summary.get('pesq_wb'))}  "
        f"STOI={_fmt_eval_metric(summary.get('stoi'))}  "
        f"LSD={_fmt_eval_metric(summary.get('lsd_db'))} dB  "
        f"cos={_fmt_eval_metric(summary.get('cos'))}  "
        f"H={_fmt_eval_metric(summary.get('rvq_h_mean_bits_sym'))} b/sym  "
        f"idx_bps={_fmt_eval_metric(summary.get('empirical_index_bps'), 0)}"
    )


_EVAL_METRIC_ALIASES = {
    "sisdr": "si_sdr_db",
    "sisdr_db": "si_sdr_db",
    "si_sdr": "si_sdr_db",
    "si_sdr_db": "si_sdr_db",
    "pesq": "pesq_wb",
    "pesq_wb": "pesq_wb",
    "stoi": "stoi",
    "lsd": "lsd_db",
    "lsd_db": "lsd_db",
    "cos": "cos",
    "rvq_h": "rvq_h_mean_bits_sym",
    "rvq_h_mean_bits_sym": "rvq_h_mean_bits_sym",
    "idx_bps": "empirical_index_bps",
    "empirical_index_bps": "empirical_index_bps",
}


def _canonical_eval_metric_name(name: str) -> str:
    key = (name or "si_sdr_db").strip().lower().replace("-", "_")
    return _EVAL_METRIC_ALIASES.get(key, key)


def _eval_metric_higher_is_better(name: str) -> bool:
    return _canonical_eval_metric_name(name) not in {"lsd_db", "empirical_index_bps"}


def _is_better_eval_value(new: float, old: float | None, metric: str) -> bool:
    if not math.isfinite(new):
        return False
    if old is None or not math.isfinite(old):
        return True
    return new > old if _eval_metric_higher_is_better(metric) else new < old


def _write_best_holdout_marker(
    cfg: Config,
    *,
    step: int,
    metric: str,
    value: float,
    previous: float | None,
    summary: dict[str, float | int | None],
) -> None:
    ck_dir = Path(cfg.checkpoint_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    marker = ck_dir / "best_holdout.json"
    payload = {
        "step": int(step),
        "metric": _canonical_eval_metric_name(metric),
        "value": float(value),
        "previous_value": None if previous is None else float(previous),
        "higher_is_better": _eval_metric_higher_is_better(metric),
        "codec_checkpoint": str(ck_dir / f"codec_step{int(step)}.npz"),
        "full_checkpoint": str(ck_dir / f"ckpt_step{int(step)}.safetensors"),
        "summary": {k: (None if v is None else float(v)) for k, v in summary.items() if k != "n"},
        "n": int(summary.get("n") or 0),
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _spectral_loss_batch(
    pred: mx.array,
    tgt: mx.array,
    *,
    step: int,
    max_items: int,
) -> tuple[mx.array, mx.array]:
    n = int(max_items)
    b = int(pred.shape[0])
    if n <= 0 or n >= b:
        return pred, tgt
    n = max(1, n)
    start = (int(step) * n) % b
    end = start + n
    if end <= b:
        return pred[start:end], tgt[start:end]
    wrap = end - b
    return mx.concatenate([pred[start:], pred[:wrap]], axis=0), mx.concatenate([tgt[start:], tgt[:wrap]], axis=0)


def _active_stft_scales(
    cfg: Config,
    step: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...] | None]:
    scales = tuple(cfg.stft_scales)
    weights = tuple(cfg.stft_scale_weights) if cfg.stft_scale_weights is not None else None
    every = int(getattr(cfg, "stft_large_every", 1))
    min_fft = int(getattr(cfg, "stft_large_min_fft", 0))
    if every <= 1 or min_fft <= 0 or int(step) % every == 0:
        return scales, weights
    keep = [i for i, (n_fft, _) in enumerate(scales) if int(n_fft) < min_fft]
    if not keep:
        return scales, weights
    active_scales = tuple(scales[i] for i in keep)
    active_weights = tuple(weights[i] for i in keep) if weights is not None else None
    return active_scales, active_weights


def make_train_fn(
    model: MLXCodec,
    cfg: Config,
    batch: mx.array,
    step: int,
    mel_fb: mx.array | None = None,
    *,
    compiled_multi_stft: Callable[[mx.array, mx.array], mx.array] | None = None,
):
    """Return a zero-arg loss for nn.value_and_grad(model, loss_fn).

    ``loss_fn.forward_metrics`` is filled each step with arrays from the **same** forward used for
    the loss (so logging does not run a second ``forward_full``). Pure spectral recipe:
    log-mag L1 + SC + complex STFT L1 (+ optional grad / cos / mel / linear-mag), no GAN.
    """

    forward_metrics: dict = {}

    def loss_fn(m: MLXCodec) -> mx.array:
        want_ae_anchor = (
            not cfg.ae_only
            and (float(getattr(cfg, "lambda_ae_anchor_time", 0.0)) > 0.0 or float(getattr(cfg, "lambda_ae_anchor_cos", 0.0)) > 0.0)
        )
        want_band_branches = float(getattr(cfg, "lambda_band_branch_l1", 0.0)) > 0.0
        want_harm_aux = bool(getattr(cfg, "decoder_harmonic_source", False)) and (
            float(getattr(cfg, "lambda_harmonic_f0", 0.0)) > 0.0
            or float(getattr(cfg, "lambda_harmonic_amp", 0.0)) > 0.0
        )
        if want_harm_aux and want_band_branches:
            y_hat, vq_l, ent_pos, marg_ent, idx, y_bands, h_freq, h_amp = m.forward_full_with_bands_and_harmonic_aux(batch)
            y_ae_anchor = None
        elif want_harm_aux and want_ae_anchor:
            y_hat, vq_l, ent_pos, marg_ent, idx, y_ae_anchor, h_freq, h_amp = m.forward_full_with_continuous_and_harmonic_aux(batch)
            y_bands = None
        elif want_harm_aux:
            y_hat, vq_l, ent_pos, marg_ent, idx, h_freq, h_amp = m.forward_full_with_harmonic_aux(batch)
            y_ae_anchor = None
            y_bands = None
        elif want_ae_anchor and want_band_branches:
            y_hat, vq_l, ent_pos, marg_ent, idx, y_ae_anchor, y_bands = m.forward_full_with_continuous_and_bands(batch)
            h_freq = None
            h_amp = None
        elif want_ae_anchor:
            y_hat, vq_l, ent_pos, marg_ent, idx, y_ae_anchor = m.forward_full_with_continuous(batch)
            y_bands = None
            h_freq = None
            h_amp = None
        elif want_band_branches:
            y_hat, vq_l, ent_pos, marg_ent, idx, y_bands = m.forward_full_with_bands(batch)
            y_ae_anchor = None
            h_freq = None
            h_amp = None
        else:
            y_hat, vq_l, ent_pos, marg_ent, idx = m.forward_full(batch)
            y_ae_anchor = None
            y_bands = None
            h_freq = None
            h_amp = None
        lt = mx.mean(mx.abs(y_hat - batch))
        spec_mult = spectral_loss_step_multiplier(step, cfg)
        spec_active = spec_mult > 0.0
        stft_scales, stft_weights = _active_stft_scales(cfg, step)
        if spec_active:
            y_spec, b_spec = _spectral_loss_batch(y_hat, batch, step=step, max_items=cfg.spectral_batch_items)
            if (
                cfg.lambda_stft_grad > 0
                or cfg.lambda_stft_cos > 0
                or cfg.lambda_mag_l1 > 0
                or cfg.lambda_sc > 0
                or cfg.lambda_complex_stft > 0
                or cfg.lambda_stft_excess > 0
                or cfg.lambda_peak_contrast > 0
                or cfg.lambda_peak_mag > 0
                or cfg.lambda_freq_ac > 0
                or cfg.lambda_hf_complex > 0
            ):
                ls, lsg, lsc, l_lin, l_sc, l_cx, l_excess, l_peak, l_peak_mag, l_freq_ac = multi_stft_all_terms(
                    y_spec,
                    b_spec,
                    stft_scales,
                    with_grad=cfg.lambda_stft_grad > 0,
                    with_cos_1m=cfg.lambda_stft_cos > 0,
                    with_linear=cfg.lambda_mag_l1 > 0,
                    with_sc=cfg.lambda_sc > 0,
                    with_complex=cfg.lambda_complex_stft > 0,
                    with_excess=cfg.lambda_stft_excess > 0,
                    with_peak_contrast=cfg.lambda_peak_contrast > 0,
                    with_peak_mag=cfg.lambda_peak_mag > 0,
                    with_freq_ac=cfg.lambda_freq_ac > 0,
                    grad_freq_weight=cfg.stft_grad_freq_weight,
                    grad_time_weight=cfg.stft_grad_time_weight,
                    hf_emphasis=cfg.stft_hf_emphasis,
                    excess_margin=cfg.stft_excess_margin,
                    sample_rate=cfg.sample_rate,
                    peak_min_hz=cfg.hf_min_hz,
                    peak_gate_db=cfg.hf_gate_db,
                    peak_radius=cfg.peak_contrast_radius,
                    freq_ac_min_hz=cfg.freq_ac_min_hz,
                    freq_ac_lags_hz=cfg.freq_ac_lags_hz,
                    scale_weights=stft_weights,
                )
            else:
                if compiled_multi_stft is not None:
                    ls = compiled_multi_stft(y_spec, b_spec)
                else:
                    ls = multi_stft_loss(
                        y_spec,
                        b_spec,
                        stft_scales,
                        hf_emphasis=cfg.stft_hf_emphasis,
                        scale_weights=stft_weights,
                    )
                lsg = mx.array(0.0)
                lsc = mx.array(0.0)
                l_lin = mx.array(0.0)
                l_sc = mx.array(0.0)
                l_cx = mx.array(0.0)
                l_excess = mx.array(0.0)
                l_peak = mx.array(0.0)
                l_peak_mag = mx.array(0.0)
                l_freq_ac = mx.array(0.0)
        else:
            y_spec = y_hat
            b_spec = batch
            ls = mx.array(0.0)
            lsg = mx.array(0.0)
            lsc = mx.array(0.0)
            l_lin = mx.array(0.0)
            l_sc = mx.array(0.0)
            l_cx = mx.array(0.0)
            l_excess = mx.array(0.0)
            l_peak = mx.array(0.0)
            l_peak_mag = mx.array(0.0)
            l_freq_ac = mx.array(0.0)
        cos = batch_mean_cosine_for_metrics(batch, y_hat, cfg)
        l_sisdr = batch_neg_log_si_sdr(batch, y_hat) if cfg.lambda_sisdr > 0 else mx.array(0.0)
        l_preemph = batch_preemph_l1(batch, y_hat, cfg.preemph_coef) if cfg.lambda_preemph > 0 else mx.array(0.0)
        l_multidelta = batch_multidelta_l1(batch, y_hat, cfg.multidelta_lags) if cfg.lambda_multidelta > 0 else mx.array(0.0)
        l_band = (
            band_l1_loss(
                y_hat,
                batch,
                sample_rate=cfg.sample_rate,
                cutoffs_hz=cfg.band_split_cutoffs_hz,
                taps=cfg.band_split_taps,
                weights=cfg.band_l1_weights,
                floor=cfg.band_l1_floor,
            )
            if cfg.lambda_band_l1 > 0
            else mx.array(0.0)
        )
        l_band_branch = (
            band_branch_l1_loss(
                y_bands,
                batch,
                sample_rate=cfg.sample_rate,
                cutoffs_hz=cfg.band_split_cutoffs_hz,
                taps=cfg.band_split_taps,
                weights=cfg.band_branch_weights,
                floor=cfg.band_l1_floor,
            )
            if cfg.lambda_band_branch_l1 > 0 and y_bands is not None
            else mx.array(0.0)
        )
        l_hf_complex = (
            high_frequency_complex_stft_l1(
                y_spec,
                b_spec,
                stft_scales,
                sample_rate=cfg.sample_rate,
                min_hz=cfg.hf_min_hz,
                scale_weights=stft_weights,
            )
            if spec_active and cfg.lambda_hf_complex > 0
            else mx.array(0.0)
        )
        hf_mult = effective_hf_mult(step, cfg)
        if spec_active and hf_mult > 0.0 and (cfg.lambda_hf_under > 0 or cfg.lambda_hf_sc > 0):
            l_hf_under, l_hf_sc = high_frequency_stft_terms(
                y_spec,
                b_spec,
                stft_scales,
                sample_rate=cfg.sample_rate,
                min_hz=cfg.hf_min_hz,
                gate_db=cfg.hf_gate_db,
                under_margin=cfg.hf_under_margin,
                peak_mask=cfg.hf_peak_mask,
                scale_weights=stft_weights,
            )
        else:
            l_hf_under = mx.array(0.0)
            l_hf_sc = mx.array(0.0)
        l_harm_f0, l_harm_amp, harm_voice = (
            harmonic_f0_voicing_loss(
                h_freq,
                h_amp,
                batch,
                sample_rate=cfg.sample_rate,
                frame=cfg.harmonic_f0_frame,
                hop=cfg.harmonic_f0_hop,
                lags=cfg.harmonic_f0_lags,
                voicing_threshold=cfg.harmonic_voicing_threshold,
            )
            if want_harm_aux
            else (mx.array(0.0), mx.array(0.0), mx.array(0.0))
        )
        l_stationary = (
            multi_stft_stationary_line_loss(
                y_spec,
                b_spec,
                stft_scales,
                sample_rate=cfg.sample_rate,
                min_hz=cfg.stationary_line_min_hz,
                radius=cfg.stationary_line_radius,
                margin=cfg.stationary_line_margin,
                scale_weights=stft_weights,
            )
            if spec_active and cfg.lambda_stationary_line > 0
            else mx.array(0.0)
        )
        ls_w = effective_lambda_stft(step, cfg) * spec_mult
        total = cfg.lambda_time * lt + ls_w * ls + cfg.lambda_vq * vq_l
        if cfg.lambda_mag_l1 > 0:
            total = total + ls_w * cfg.lambda_mag_l1 * l_lin
        if cfg.lambda_stft_grad > 0:
            total = total + ls_w * cfg.lambda_stft_grad * lsg
        if cfg.lambda_stft_cos > 0:
            total = total + ls_w * cfg.lambda_stft_cos * lsc
        if cfg.lambda_sc > 0:
            total = total + ls_w * cfg.lambda_sc * l_sc
        if cfg.lambda_complex_stft > 0:
            total = total + ls_w * cfg.lambda_complex_stft * l_cx
        if cfg.lambda_stft_excess > 0:
            total = total + ls_w * cfg.lambda_stft_excess * l_excess
        if cfg.lambda_peak_contrast > 0:
            total = total + ls_w * cfg.lambda_peak_contrast * l_peak
        if cfg.lambda_peak_mag > 0:
            total = total + ls_w * cfg.lambda_peak_mag * l_peak_mag
        if cfg.lambda_freq_ac > 0:
            total = total + ls_w * cfg.lambda_freq_ac * l_freq_ac
        if cfg.lambda_stationary_line > 0:
            total = total + ls_w * cfg.lambda_stationary_line * l_stationary
        if cfg.lambda_hf_under > 0:
            total = total + ls_w * (cfg.lambda_hf_under * hf_mult) * l_hf_under
        if cfg.lambda_hf_sc > 0:
            total = total + ls_w * (cfg.lambda_hf_sc * hf_mult) * l_hf_sc
        if cfg.lambda_hf_complex > 0:
            total = total + ls_w * cfg.lambda_hf_complex * l_hf_complex
        if cfg.lambda_harmonic_f0 > 0:
            total = total + cfg.lambda_harmonic_f0 * l_harm_f0
        if cfg.lambda_harmonic_amp > 0:
            total = total + cfg.lambda_harmonic_amp * l_harm_amp
        # Diversity as non-negative penalty (same ∂/∂θ as −λ·H when H<log K): push H → log K
        log_k = mean_log_codebook_for_entropy(cfg)
        log_k_mx = mx.array(log_k, dtype=mx.float32)
        if cfg.lambda_entropy > 0:
            ent_gap = mx.maximum(mx.array(0.0, dtype=mx.float32), log_k_mx - ent_pos)
            total = total + cfg.lambda_entropy * ent_gap
        if cfg.lambda_marginal > 0:
            marg_gap = mx.maximum(mx.array(0.0, dtype=mx.float32), log_k_mx - marg_ent)
            total = total + effective_lambda_marginal(step, cfg) * marg_gap
        if cfg.lambda_cos > 0:
            total = total + cfg.lambda_cos * (1.0 - cos)
        if cfg.cos_hinge > 0:
            gap = mx.maximum(mx.array(0.0), mx.array(cfg.cos_target, dtype=mx.float32) - cos)
            total = total + cfg.cos_hinge * gap
        if cfg.lambda_sisdr > 0:
            total = total + cfg.lambda_sisdr * l_sisdr
        if cfg.lambda_preemph > 0:
            total = total + cfg.lambda_preemph * l_preemph
        if cfg.lambda_multidelta > 0:
            total = total + cfg.lambda_multidelta * l_multidelta
        if cfg.lambda_band_l1 > 0:
            total = total + cfg.lambda_band_l1 * l_band
        if cfg.lambda_band_branch_l1 > 0:
            total = total + cfg.lambda_band_branch_l1 * l_band_branch
        ae_lt = mx.array(0.0)
        ae_cos = mx.array(1.0)
        ae_anchor = mx.array(0.0)
        if y_ae_anchor is not None:
            ae_lt = mx.mean(mx.abs(y_ae_anchor - batch))
            ae_cos = batch_mean_cosine(batch, y_ae_anchor)
            ae_anchor = cfg.lambda_ae_anchor_time * ae_lt + cfg.lambda_ae_anchor_cos * (1.0 - ae_cos)
            total = total + ae_anchor
        lm1 = mx.array(0.0)
        lm2 = mx.array(0.0)
        if spec_active and mel_fb is not None and (cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0):
            lm1, lm2 = mel_log_bin_losses(
                y_spec,
                b_spec,
                mel_fb,
                cfg.mel_n_fft,
                cfg.mel_hop,
            )
            total = total + ls_w * (cfg.lambda_mel_l1 * lm1 + cfg.lambda_mel_l2 * lm2)
        forward_metrics.clear()
        forward_metrics.update(
            lt=lt,
            ls=ls,
            lsg=lsg,
            lsc=lsc,
            l_sc=l_sc,
            l_cx=l_cx,
            l_excess=l_excess,
            l_peak=l_peak,
            l_peak_mag=l_peak_mag,
            l_freq_ac=l_freq_ac,
            l_stationary=l_stationary,
            l_harm_f0=l_harm_f0,
            l_harm_amp=l_harm_amp,
            harm_voice=harm_voice,
            l_hf_under=l_hf_under,
            l_hf_sc=l_hf_sc,
            l_hf_complex=l_hf_complex,
            hf_mult=mx.array(hf_mult, dtype=mx.float32),
            spec_mult=mx.array(spec_mult, dtype=mx.float32),
            l_lin=l_lin,
            l_sisdr=l_sisdr,
            l_preemph=l_preemph,
            l_multidelta=l_multidelta,
            l_band=l_band,
            l_band_branch=l_band_branch,
            ae_lt=ae_lt,
            ae_cos=ae_cos,
            ae_anchor=ae_anchor,
            lm1=lm1,
            lm2=lm2,
            vq_l=vq_l,
            ent_pos=ent_pos,
            marg_ent=marg_ent,
            cos_m=cos,
            idx=idx,
        )
        return total

    loss_fn.forward_metrics = forward_metrics  # type: ignore[attr-defined]
    return loss_fn


def _kmeans_centroids_numpy(x, k: int, rng, *, max_iter: int = 25):
    """``x`` [n, d] numpy, ``k`` ≤ ``n``. Lloyd with k-means++-style init; returns ``k`` rows [k, d] float32."""
    import numpy as np

    n, d = int(x.shape[0]), int(x.shape[1])
    k = min(int(k), n)
    if k <= 0:
        return np.zeros((0, d), dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # k-means++: first center random, next proportional to D² to existing set
    idx = int(rng.integers(0, n))
    centers = [x[idx].copy()]
    for _ in range(k - 1):
        cstk = np.stack(centers, axis=0)
        dist2 = np.sum((x[:, None, :] - cstk[None, :, :]) ** 2, axis=2)
        dmin = np.min(dist2, axis=1).astype(np.float64)
        p = np.maximum(dmin, 1e-12)
        p = p / float(np.sum(p))
        nxt = int(rng.choice(n, p=p))
        centers.append(x[nxt].copy())
    c = np.stack(centers, axis=0).astype(np.float32)
    for _ in range(max_iter):
        dist2 = np.sum((x[:, None, :] - c[None, :, :]) ** 2, axis=2)
        lab = np.argmin(dist2, axis=1)
        new = np.zeros_like(c)
        for j in range(k):
            m = lab == j
            if np.any(m):
                new[j] = np.mean(x[m], axis=0)
            else:
                new[j] = c[j]
        if np.allclose(new, c, rtol=1e-5, atol=1e-6):
            break
        c = new
    return c


def vq_reset_dead_codes(model: MLXCodec, batch: mx.array, cfg: Config) -> tuple[int, list[int]]:
    """When a stage is collapsed (few distinct codes), fill unused rows from batch residuals (+noise).

    Dead rows are filled either by **K-means centroids** (``vq_reset_kmeans``) or random residual samples.
    If ``n_unique ≤ vq_reset_full_refresh_max_unique`` (e.g. single-code collapse), **all** ``K`` rows are
    replaced with K-means (+ pad) so the dominant row is not left unchanged.

    If collapse is severe (``unique/K`` below ~35% of ``vq_reset_collapse_frac``), optionally permute all
    codebook rows so ``argmin`` is not stuck on index 0 when distances are near-ties.
    """
    if cfg.ae_only or cfg.vq_reset_every <= 0:
        return 0, []
    try:
        import numpy as np
    except ImportError:
        return 0, []

    z = model.latent_before_rvq(batch)
    mx.eval(z)
    quantized = mx.zeros_like(z)
    total = 0
    per_stage: list[int] = []
    sizes = effective_codebook_sizes(cfg)
    thr = cfg.vq_reset_collapse_frac
    # Shuffle all rows when collapse is worse than ~⅓ of the reset threshold (scales with --vq-reset-collapse-frac)
    severe_shuffle_frac = max(0.05, float(thr) * 0.35)

    for i in range(cfg.n_codebooks):
        k_sz = sizes[i]
        stage: VectorQuantizerStage = getattr(model.rvq, f"q{i}")
        residual = z - quantized
        z_i, _, _, idx = stage(residual)
        mx.eval(z_i, idx)
        quantized = quantized + z_i

        idx_np = np.array(idx).astype(np.int64).ravel()
        n_unique = int(len(np.unique(idx_np)))
        if n_unique / float(k_sz) >= thr:
            per_stage.append(0)
            continue

        hist = np.bincount(idx_np, minlength=k_sz)
        dead = np.where(hist == 0)[0]

        d_emb = int(np.array(stage.embedding["weight"]).shape[1])
        res_for_km = _stage_code_space_residual(stage, residual, cfg)
        mx.eval(res_for_km)
        res_np = np.array(res_for_km).reshape(-1, d_emb).astype(np.float32)
        npos = res_np.shape[0]
        if npos < 1:
            per_stage.append(0)
            continue

        rng = np.random.default_rng()

        # Single-code (or near) collapse: dead-row refill never touches the winning embedding → argmin stays tied.
        if n_unique <= int(cfg.vq_reset_full_refresh_max_unique):
            kk = min(k_sz, npos)
            cents = _kmeans_centroids_numpy(res_np, kk, rng)
            w = np.zeros((k_sz, d_emb), dtype=np.float32)
            w[:kk] = cents
            if kk < k_sz:
                pad = rng.integers(0, npos, size=(k_sz - kk,))
                w[kk:] = res_np[pad]
            rs = float(np.std(res_np)) + 1e-5
            if cfg.vq_reset_noise > 0:
                w = w + (cfg.vq_reset_noise * rs) * rng.standard_normal(size=w.shape).astype(np.float32)
            if cfg.vq_reset_shuffle:
                w = w[rng.permutation(k_sz)]
            stage.embedding["weight"] = mx.array(w)
            total += k_sz
            per_stage.append(k_sz)
            quantized = mx.zeros_like(z)
            for j in range(i + 1):
                stj: VectorQuantizerStage = getattr(model.rvq, f"q{j}")
                res_j = z - quantized
                z_j, _, _, _ = stj(res_j)
                mx.eval(z_j)
                quantized = quantized + z_j
            continue

        if dead.size == 0:
            per_stage.append(0)
            continue

        need = int(dead.size)
        if cfg.vq_reset_kmeans and need > 0:
            kk = min(need, npos)
            cents = _kmeans_centroids_numpy(res_np, kk, rng)
            if kk < need:
                pad = rng.integers(0, npos, size=(need - kk,))
                new_rows = np.vstack([cents, res_np[pad].astype(np.float32, copy=True)])
            else:
                new_rows = cents.astype(np.float32, copy=True)
        else:
            pick = rng.integers(0, npos, size=need)
            new_rows = res_np[pick].astype(np.float32, copy=True)
        rs = float(np.std(res_np)) + 1e-5
        if cfg.vq_reset_noise > 0:
            new_rows = new_rows + (cfg.vq_reset_noise * rs) * rng.standard_normal(size=new_rows.shape).astype(
                np.float32
            )

        w = np.array(stage.embedding["weight"], copy=True)
        w[dead] = new_rows
        if cfg.vq_reset_shuffle and (n_unique / float(k_sz) < severe_shuffle_frac):
            perm = rng.permutation(k_sz)
            w = w[perm]
        stage.embedding["weight"] = mx.array(w)
        total += int(dead.size)
        per_stage.append(int(dead.size))

    return total, per_stage


def update_vq_ema_codebooks(model: MLXCodec, batch: mx.array, cfg: Config) -> None:
    """EMA codebook rows toward batch means of assigned projected residuals (code space)."""
    dec = float(cfg.vq_ema_decay)
    if dec <= 0.0 or dec >= 1.0 or cfg.ae_only:
        return
    try:
        import numpy as np
    except ImportError:
        return
    beta = 1.0 - dec
    z = model.latent_before_rvq(batch)
    mx.eval(z)

    def update_chain(z_chain: mx.array, stages: list[VectorQuantizerStage]) -> None:
        quantized = mx.zeros_like(z_chain)
        for stage in stages:
            residual = z_chain - quantized
            z_i, _, _, idx = stage(residual)
            mx.eval(z_i, idx)
            r = _stage_code_space_residual(stage, residual, cfg)
            mx.eval(r)
            idx_np = np.array(idx).astype(np.int64).ravel()
            d_emb = int(np.array(stage.embedding["weight"]).shape[1])
            r_np = np.array(r).reshape(-1, d_emb).astype(np.float32)
            k_sz = int(stage.num_embeddings)
            w = np.array(stage.embedding["weight"], copy=True).astype(np.float32)
            for k in range(k_sz):
                m = idx_np == k
                if np.any(m):
                    mu = np.mean(r_np[m], axis=0)
                    w[k] = dec * w[k] + beta * mu.astype(np.float32)
            stage.embedding["weight"] = mx.array(w)
            z_i2, _, _, _ = stage(residual)
            mx.eval(z_i2)
            quantized = quantized + z_i2

    if model.rvq.group_splits and model.rvq.group_dims:
        c0 = 0
        for g, (n_stages, dim) in enumerate(zip(model.rvq.group_splits, model.rvq.group_dims, strict=True)):
            c1 = c0 + dim
            stages = [getattr(model.rvq, f"q{g}_{j}") for j in range(n_stages)]
            update_chain(z[:, :, c0:c1], stages)
            c0 = c1
        return

    update_chain(z, [getattr(model.rvq, f"q{i}") for i in range(cfg.n_codebooks)])


def _format_vq_util(indices: list[mx.array] | None, sizes: tuple[int, ...]) -> str:
    if not indices:
        return ""
    try:
        import numpy as np
    except ImportError:
        return ""
    parts = []
    for i, idx in enumerate(indices):
        k = sizes[i] if i < len(sizes) else sizes[-1]
        mx.eval(idx)
        a = np.array(idx)
        nu = int(len(np.unique(a)))
        u = float(nu) / float(k)
        parts.append(f"u{i}={nu}/{k}({100.0 * u:.1f}%)")
    return "  " + " ".join(parts)


def _stage_code_space_residual(stage: VectorQuantizerStage, residual: mx.array, cfg: Config) -> mx.array:
    r = stage.in_proj(residual) if stage.in_proj is not None else residual
    if cfg.turboquant:
        rot = _turbo_rotation(cfg.turboquant_seed, stage.stage_id, int(r.shape[-1]))
        r = r @ rot
    return r


def _tree_n_params(tree) -> int:
    if isinstance(tree, dict):
        return sum(_tree_n_params(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(_tree_n_params(v) for v in tree)
    return int(tree.size)


def _log_mag_spectrogram_np(x, sr: int, n_fft: int, hop: int):
    """[n_freq, n_frames] log10 magnitude; clamps so log10 never sees 0 (avoids matplotlib specgram warnings)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    win = np.hanning(n_fft)
    cols = []
    for start in range(0, x.size - n_fft + 1, hop):
        frame = x[start : start + n_fft] * win
        mag = np.abs(np.fft.rfft(frame, n=n_fft))
        cols.append(mag)
    if not cols:
        return np.zeros((n_fft // 2 + 1, 1))
    s = np.stack(cols, axis=1)
    return np.log10(np.maximum(s, 1e-10))


def save_spectrogram_png(
    orig: mx.array,
    recon: mx.array,
    sample_rate: int,
    out_path: Path,
    step: int,
) -> bool:
    """Save orig vs recon spectrograms (first sample in batch). Returns False if matplotlib missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    mx.eval(orig, recon)
    o = np.array(orig).astype(np.float64).flatten()
    r = np.array(recon).astype(np.float64).flatten()
    dur = o.size / float(sample_rate)
    n_fft = 512
    hop = 128  # matches NFFT=512, noverlap=384 in old specgram
    s_orig = _log_mag_spectrogram_np(o, sample_rate, n_fft, hop)
    s_recon = _log_mag_spectrogram_np(r, sample_rate, n_fft, hop)
    both = np.concatenate([s_orig.reshape(-1), s_recon.reshape(-1)])
    vmin = float(np.percentile(both, 1.0))
    vmax = float(np.percentile(both, 99.5))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = None, None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = min(22.0, max(10.0, 6.0 + dur * 1.5))
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 6), sharex=True)
    for ax, data, spec, title in (
        (axes[0], o, s_orig, "original"),
        (axes[1], r, s_recon, "reconstruction"),
    ):
        t1 = data.size / float(sample_rate)
        ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0.0, t1, 0.0, sample_rate / 2.0],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel("Hz")
        ax.set_title(title)
    axes[1].set_xlabel("Time (s)")
    fig.suptitle(f"MLX codec  step {step}  sr={sample_rate}  T={dur:.2f}s  (log10 magnitude)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def save_reconstruction_wavs(
    orig: mx.array,
    recon: mx.array,
    sample_rate: int,
    out_stem: Path,
) -> bool:
    """Write ``{out_stem}_orig.wav`` and ``{out_stem}_recon.wav`` (mono PCM_16, sanitized)."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return False

    mx.eval(orig, recon)
    o = np.array(orig, dtype=np.float32).reshape(-1)
    r = np.array(recon, dtype=np.float32).reshape(-1)
    # clip does not remove NaN/Inf — players then see corrupt PCM
    o = np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    o = np.clip(o, -1.0, 1.0)
    r = np.clip(r, -1.0, 1.0)
    o_i16 = np.round(o * 32767.0).astype(np.int16)
    r_i16 = np.round(r * 32767.0).astype(np.int16)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"samplerate": int(sample_rate), "subtype": "PCM_16", "format": "WAV"}
    sf.write(str(out_stem) + "_orig.wav", o_i16, **kwargs)
    sf.write(str(out_stem) + "_recon.wav", r_i16, **kwargs)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 6. main
# ═══════════════════════════════════════════════════════════════════════════
_CKPT_NAME_RE = re.compile(r"^codec_step(\d+)\.npz$")
_CKPT_FULL_RE = re.compile(r"^ckpt_step(\d+)\.safetensors$")


def _parse_checkpoint_step(path: Path) -> int | None:
    m = _CKPT_NAME_RE.match(path.name)
    return int(m.group(1)) if m else None


def _parse_full_checkpoint_step(path: Path) -> int | None:
    m = _CKPT_FULL_RE.match(path.name)
    return int(m.group(1)) if m else None


def _find_latest_checkpoint(ck_dir: Path) -> Path | None:
    """Latest checkpoint by step ``N``; prefers ``ckpt_stepN.safetensors`` over ``codec_stepN.npz`` at same ``N``."""
    best_key: tuple[int, int] | None = None
    best_p: Path | None = None
    if not ck_dir.is_dir():
        return None
    for p in ck_dir.glob("codec_step*.npz"):
        n = _parse_checkpoint_step(p)
        if n is None:
            continue
        key = (n, 0)
        if best_key is None or key > best_key:
            best_key, best_p = key, p
    for p in ck_dir.glob("ckpt_step*.safetensors"):
        n = _parse_full_checkpoint_step(p)
        if n is None:
            continue
        key = (n, 1)
        if best_key is None or key > best_key:
            best_key, best_p = key, p
    return best_p


def _is_full_checkpoint_path(path: Path) -> bool:
    return path.suffix.lower() == ".safetensors" and _CKPT_FULL_RE.match(path.name) is not None


_LEGACY_ENC_LAYERS = re.compile(r"^encoder\.layers\.(\d+)\.(weight|bias)$")
_LEGACY_DEC_LAYERS = re.compile(r"^decoder\.layers\.(\d+)\.(weight|bias)$")


def _remap_legacy_codec_subdict(sub: dict[str, mx.array], module: MLXCodec) -> tuple[dict[str, mx.array], int]:
    """Older checkpoints used ``encoder.layers.N`` / ``decoder.layers.N``; current code uses ``_b*`` / ``_d*``.

    ``ConvTranspose1d`` weights at ``decoder.layers.{N}`` map to ``decoder._d{N}`` (transpose) or
    ``decoder._d{N}.conv`` (``UpsampleRepeatConv``) when shapes match.
    """
    tgt_shapes = {k: tuple(v.shape) for k, v in dict(tree_flatten(module.parameters())).items()}
    out = dict(sub)
    n_moved = 0
    for k in list(out.keys()):
        m_e = _LEGACY_ENC_LAYERS.match(k)
        if m_e:
            idx, suf = m_e.group(1), m_e.group(2)
            cand = f"encoder._b{idx}.{suf}"
            v = out.pop(k)
            if cand in tgt_shapes and tuple(v.shape) == tgt_shapes[cand]:
                out[cand] = v
                n_moved += 1
            else:
                print(
                    f"[resume] warn: legacy {k!r} shape {tuple(v.shape)} → no match for {cand!r}",
                    flush=True,
                )
            continue
        m_d = _LEGACY_DEC_LAYERS.match(k)
        if not m_d:
            continue
        idx, suf = m_d.group(1), m_d.group(2)
        v = out.pop(k)
        candidates = (f"decoder._d{idx}.{suf}", f"decoder._d{idx}.conv.{suf}")
        placed = False
        for cand in candidates:
            if cand in tgt_shapes and tuple(v.shape) == tgt_shapes[cand]:
                out[cand] = v
                n_moved += 1
                placed = True
                break
        if not placed:
            print(
                f"[resume] warn: legacy {k!r} shape {tuple(v.shape)} did not match "
                f"{candidates[0]!r} or {candidates[1]!r} — decoder may be partially random",
                flush=True,
            )
    return out, n_moved


def _load_weights_prefix(module: nn.Module, flat: dict[str, mx.array], prefix: str) -> None:
    pl = len(prefix)
    sub = {k[pl:]: v for k, v in flat.items() if k.startswith(prefix) and isinstance(v, mx.array)}
    if not sub:
        return
    try:
        module.load_weights(list(sub.items()), strict=True)
        return
    except Exception:
        pass
    if prefix == "model/" and isinstance(module, MLXCodec):
        sub2, n_rem = _remap_legacy_codec_subdict(sub, module)
        if n_rem > 0:
            print(
                f"[resume] remapped {n_rem} legacy ``encoder.layers``/``decoder.layers`` keys → ``_b``/``_d``",
                flush=True,
            )
        module.load_weights(list(sub2.items()), strict=False)
    else:
        module.load_weights(list(sub.items()), strict=False)


def _load_optimizer_prefix(opt: Adam, flat: dict[str, mx.array], prefix: str) -> None:
    pl = len(prefix)
    for k, v in flat.items():
        if not k.startswith(prefix):
            continue
        sk = k[pl:]
        if sk in opt.state:
            opt.state[sk] = v


def save_full_checkpoint(
    path: Path,
    *,
    step: int,
    data_off: int,
    model: nn.Module,
    opt: Adam,
    lr_plateau_state: dict | None = None,
) -> None:
    """Single-safetensors checkpoint with model weights + Adam state. No disc / no EMA (GAN-free recipe)."""
    flat: dict[str, mx.array] = {}
    flat.update({f"model/{k}": v for k, v in dict(tree_flatten(model.parameters())).items()})
    flat.update({f"opt_g/{k}": v for k, v in dict(tree_flatten(opt.state)).items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), flat)
    meta_path = path.with_suffix(".meta.json")
    meta: dict = {"step": step, "data_off": data_off}
    if lr_plateau_state is not None:
        meta["lr_plateau"] = lr_plateau_state
    meta_path.write_text(json.dumps(meta, indent=0) + "\n")


def load_full_checkpoint_into(
    flat: dict[str, mx.array],
    *,
    model: nn.Module,
    opt: Adam | None,
) -> None:
    _load_weights_prefix(model, flat, "model/")
    if opt is not None:
        _load_optimizer_prefix(opt, flat, "opt_g/")


def _cast_module_floats(module: nn.Module, dtype) -> None:
    flat = [(k, v.astype(dtype)) for k, v in tree_flatten(module.parameters()) if isinstance(v, mx.array)]
    if flat:
        module.load_weights(flat)


class PlateauLRSchedule:
    """ReduceLROnPlateau-style LR: drop ``_lr`` when EMA(train loss) stops improving (after warmup).

    MLX ``Adam`` takes ``lr(step)``; this object is callable and holds mutable ``_lr``. Call
    ``observe(loss)`` once per optimizer step after ``opt.update`` (warmup steps skipped).
    """

    __slots__ = (
        "peak",
        "min_lr",
        "wu",
        "factor",
        "patience",
        "threshold",
        "ema_beta",
        "cooldown",
        "_lr",
        "best",
        "bad",
        "ema",
        "_cool_left",
    )

    def __init__(
        self,
        cfg: Config,
        *,
        lr_scale: float = 1.0,
        resume: dict | None = None,
    ) -> None:
        self.peak = float(cfg.lr) * float(lr_scale)
        self.min_lr = self.peak * float(cfg.lr_min_ratio)
        self.wu = max(0, int(cfg.lr_warmup_steps))
        self.factor = float(cfg.lr_plateau_factor)
        self.patience = max(1, int(cfg.lr_plateau_patience))
        self.threshold = max(0.0, float(cfg.lr_plateau_threshold))
        self.ema_beta = float(cfg.lr_plateau_ema)
        self.cooldown = max(0, int(cfg.lr_plateau_cooldown))
        self._lr = self.peak
        self.best = float("inf")
        self.bad = 0
        self.ema: float | None = None
        self._cool_left = 0
        if resume:
            self.load_state(resume)

    def load_state(self, d: dict) -> None:
        self._lr = float(d.get("lr", self._lr))
        b = d.get("best")
        self.best = float("inf") if b is None else float(b)
        self.bad = int(d.get("bad", 0))
        e = d.get("ema")
        self.ema = None if e is None else float(e)
        self._cool_left = int(d.get("cool_left", 0))

    def state_dict(self) -> dict:
        return {
            "lr": self._lr,
            "best": None if self.best == float("inf") else self.best,
            "bad": self.bad,
            "ema": self.ema,
            "cool_left": self._cool_left,
        }

    def observe(self, loss: float) -> None:
        lf = float(loss)
        if not math.isfinite(lf):
            return
        beta = self.ema_beta
        if not (0.0 <= beta < 1.0):
            beta = 0.0
        if self.ema is None:
            self.ema = lf
        else:
            self.ema = beta * float(self.ema) + (1.0 - beta) * lf
        m = float(self.ema)
        if self._cool_left > 0:
            self._cool_left -= 1
            return
        if self.best == float("inf") or m < self.best * (1.0 - self.threshold):
            self.best = m
            self.bad = 0
            return
        self.bad += 1
        if self.bad > self.patience:
            new_lr = max(self._lr * self.factor, self.min_lr)
            if new_lr < self._lr - 1e-15:
                old = self._lr
                self._lr = new_lr
                print(
                    f"  [lr-plateau] lr {old:.2e} → {self._lr:.2e}  "
                    f"(ema_loss={m:.5f} best={self.best:.5f})",
                    flush=True,
                )
            self.bad = 0
            self._cool_left = self.cooldown

    def __call__(self, step_mx: mx.array) -> mx.array:
        s = int(step_mx.item())
        if self.wu > 0 and s < self.wu:
            return mx.array(self.peak * float(s + 1) / float(self.wu), dtype=mx.float32)
        return mx.array(max(self._lr, self.min_lr), dtype=mx.float32)


def build_lr_schedule(
    cfg: Config,
    *,
    lr_scale: float = 1.0,
    plateau_resume: dict | None = None,
) -> float | Callable[[mx.array], mx.array] | PlateauLRSchedule:
    """Return constant LR, cosine schedule, or a :class:`PlateauLRSchedule` instance.

    ``lr_scale`` scales the peak (and cosine floor / plateau min).
    """
    scale = float(lr_scale)
    mode = (cfg.lr_schedule or "none").strip().lower()
    if mode in ("none", "constant", "off"):
        return float(cfg.lr) * scale
    if mode == "plateau":
        return PlateauLRSchedule(cfg, lr_scale=scale, resume=plateau_resume)
    if mode != "cosine":
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r} (use none, cosine, or plateau)")
    peak = float(cfg.lr) * scale
    end = peak * float(cfg.lr_min_ratio)
    total = max(1, int(cfg.steps))
    wu = max(0, int(cfg.lr_warmup_steps))
    if wu >= total:
        wu = 0
    if wu > 0:
        warm = linear_schedule(0.0, peak, wu)
        tail_n = max(1, total - wu - 1)
        tail = cosine_decay(peak, tail_n, end=end)
        return join_schedules([warm, tail], [wu])
    decay_n = max(1, total - 1)
    return cosine_decay(peak, decay_n, end=end)


def _refresh_optimizer_schedules(opt: Adam) -> None:
    """Re-evaluate scheduled tensors after mutating ``opt.state['step']`` (e.g. resume)."""
    for name, fn in opt._schedulers.items():
        opt.state[name] = fn(opt.step)


def main() -> None:
    p = argparse.ArgumentParser(description="MLX Karpathy-style audio codec trainer")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        metavar="K",
        help="Per optimizer step: K micro-batches of size --batch; gradients averaged (effective batch B·K). Prefetch disabled when K>1.",
    )
    p.add_argument(
        "--load-audio-threads",
        type=int,
        default=0,
        metavar="T",
        help="Concurrent per-row disk decode for Libri batches (0 or 1=sequential; try 8–16 to overlap I/O)",
    )
    p.add_argument(
        "--prefetch-audio",
        action="store_true",
        help="While the GPU runs step t, load batch t+1 on a side thread (only with --data-dir / --librispeech)",
    )
    p.add_argument("--segment", type=int, default=16384, help="Waveform length per sample (more latent frames helps RVQ)")
    p.add_argument("--lr", type=float, default=5e-4, help="Adam peak LR (warmup/plateau/cosine) or constant (--lr-schedule none)")
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="plateau",
        choices=("none", "cosine", "plateau"),
        help="LR: plateau=ReduceLROnPlateau on EMA train loss; cosine=decay to --lr×--lr-min-ratio; none=constant",
    )
    p.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.1,
        metavar="R",
        help="Cosine floor or plateau minimum lr = --lr × R",
    )
    p.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        metavar="W",
        help="Linear warmup steps 0→--lr before cosine/plateau monitoring (0 = off)",
    )
    p.add_argument(
        "--lr-plateau-factor",
        type=float,
        default=0.25,
        metavar="F",
        help="Plateau: multiply lr by F when patience exceeded (try 0.2–0.35 for fast ramp down)",
    )
    p.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=1200,
        metavar="P",
        help="Plateau: optimizer steps without relative improvement before lr drop",
    )
    p.add_argument(
        "--lr-plateau-threshold",
        type=float,
        default=0.002,
        metavar="T",
        help="Plateau: rel improvement if ema_loss < best × (1−T)",
    )
    p.add_argument(
        "--lr-plateau-ema",
        type=float,
        default=0.985,
        metavar="B",
        help="Plateau: EMA decay on train loss for monitoring (0 = raw loss; use <1)",
    )
    p.add_argument(
        "--lr-plateau-cooldown",
        type=int,
        default=300,
        metavar="C",
        help="Plateau: steps after an lr drop before patience counts again",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        metavar="NORM",
        help="Scale gradients so global L2 norm ≤ this (0 = no clipping)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ae-only", action="store_true", help="Autoencoder only (no VQ)")
    p.add_argument(
        "--enc-channels",
        type=str,
        default="24,32,48,64,96,128,192,256",
        help="Comma-separated encoder widths; len = stride-2 stack depth (mlx_lb_v2: 8 → 256×)",
    )
    p.add_argument(
        "--stride1-blocks-per-scale",
        type=int,
        default=0,
        help="Stride-1 Conv+GELU blocks per resolution before each downsample / after each upsample (0 = default; 1 = v2 refinement but cold init can give NaN grads—resume ok)",
    )
    p.add_argument(
        "--decoder-refine-blocks-per-scale",
        type=int,
        default=0,
        help="Extra decoder-only stride-1 Conv+activation blocks after each upsample; keeps encoder stride and nominal bitrate unchanged",
    )
    p.add_argument(
        "--latent-dim",
        type=int,
        default=512,
        help="Latent / VQ embedding dimension D (same n_q,K → same nominal kbps; larger D = richer codewords in R^D)",
    )
    p.add_argument(
        "--pre-vq-layernorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LayerNorm on encoder output before RVQ (default: off)",
    )
    p.add_argument(
        "--latent-temporal-depth",
        type=int,
        default=0,
        metavar="N",
        help="Residual dilated Conv1d blocks on latents before RVQ (0=off; try 2–4)",
    )
    p.add_argument(
        "--latent-temporal-post-depth",
        type=int,
        default=0,
        metavar="M",
        help="Same temporal stack on quantized z_q before decoder (0=off)",
    )
    p.add_argument(
        "--self-attention-depth",
        type=int,
        default=0,
        metavar="N",
        help="SelfAttention1D blocks on latents before RVQ (0=off)",
    )
    p.add_argument(
        "--self-attention-post-depth",
        type=int,
        default=0,
        metavar="M",
        help="SelfAttention1D blocks on quantized z_q before decoder (0=off)",
    )
    p.add_argument(
        "--self-attention-heads",
        type=int,
        default=8,
        metavar="H",
        help="Attention heads for SelfAttention1D; must divide --latent-dim",
    )
    p.add_argument(
        "--state-space-depth",
        type=int,
        default=0,
        metavar="N",
        help="S4D/DSS-style state-space blocks on latents before RVQ (0=off; mutually exclusive with pre SelfAttention1D)",
    )
    p.add_argument(
        "--state-space-post-depth",
        type=int,
        default=0,
        metavar="M",
        help="S4D/DSS-style state-space blocks on quantized z_q before decoder (0=off)",
    )
    p.add_argument(
        "--state-space-state-dim",
        type=int,
        default=16,
        metavar="S",
        help="Diagonal SSM states per expanded latent channel",
    )
    p.add_argument(
        "--state-space-expand",
        type=int,
        default=1,
        metavar="E",
        help="Channel expansion inside state-space mixer",
    )
    p.add_argument(
        "--state-space-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use forward+backward SSM convolutions (disable for causal experiments)",
    )
    p.add_argument(
        "--n-codebooks",
        type=int,
        default=3,
        help="Residual VQ depth (default 3; paired with default K=32 → nominal <1 kb/s indices @ 16 kHz / 256× stride)",
    )
    p.add_argument(
        "--codebook-size",
        type=int,
        default=32,
        metavar="K",
        help="Uniform K per stage when --codebook-sizes not set (default 32 for <1 kb/s with 3 books)",
    )
    p.add_argument(
        "--codebook-sizes",
        type=str,
        default=None,
        metavar="K0,K1,…",
        help="Comma-separated K per RVQ stage (length must match --n-codebooks), e.g. decreasing 256,128,64",
    )
    p.add_argument(
        "--rvq-band-splits",
        type=str,
        default=None,
        metavar="N0,N1,…",
        help="Optional grouped RVQ stage counts over latent channel slices; sum must equal --n-codebooks, e.g. 5,5,5",
    )
    p.add_argument(
        "--turboquant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="TurboQuant-compatible RVQ: fixed random orthogonal rotation before code assignment, inverse rotation after lookup",
    )
    p.add_argument(
        "--turboquant-seed",
        type=int,
        default=1729,
        help="Seed for fixed TurboQuant RVQ rotations",
    )
    p.add_argument("--lambda-time", type=float, default=1.0, help="L1 waveform weight")
    p.add_argument(
        "--lambda-stft",
        type=float,
        default=0.35,
        help="Multi-scale log-STFT weight (lower + ramp helps RVQ vs STFT dominance)",
    )
    p.add_argument(
        "--lambda-stft-grad",
        type=float,
        default=0.08,
        metavar="W",
        help="If >0: add W·λ_stft_eff·mean L1 on ∂/∂f and ∂/∂t of log-mag STFT (same scales as STFT). Default 0.08; 0=off; try 0.03–0.15",
    )
    p.add_argument(
        "--lambda-stft-cos",
        type=float,
        default=0.0,
        metavar="W",
        help="If >0: add W·λ_stft_eff·mean(1−cos) on flattened log-mag STFT per scale (spectral cosine; separate from --lambda-cos on waveform)",
    )
    p.add_argument(
        "--lambda-stft-excess",
        type=float,
        default=0.0,
        metavar="W",
        help="If >0: penalize log-STFT energy above target by --stft-excess-margin; fights broadband haze in silent valleys",
    )
    p.add_argument(
        "--stft-excess-margin",
        type=float,
        default=0.20,
        metavar="M",
        help="Log-magnitude margin before --lambda-stft-excess activates",
    )
    p.add_argument(
        "--lambda-hf-under",
        type=float,
        default=0.0,
        metavar="W",
        help="Target-gated high-frequency missing-energy loss; restores HF peaks without rewarding broadband haze",
    )
    p.add_argument(
        "--lambda-hf-sc",
        type=float,
        default=0.0,
        metavar="W",
        help="High-frequency-only spectral convergence weight",
    )
    p.add_argument(
        "--lambda-hf-complex",
        type=float,
        default=0.0,
        metavar="W",
        help="High-frequency-only complex STFT L1; penalizes missing HF and unmatched broadband HF noise",
    )
    p.add_argument("--hf-min-hz", type=float, default=2500.0, help="Lower frequency edge for HF losses")
    p.add_argument("--hf-gate-db", type=float, default=-18.0, help="Target HF gate relative to per-sample STFT peak, in dB")
    p.add_argument(
        "--hf-under-margin",
        type=float,
        default=0.05,
        help="Log-magnitude margin before HF under-energy loss activates",
    )
    p.add_argument(
        "--hf-peak-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict HF-under to target local maxima along frequency; avoids rewarding broad HF shelves",
    )
    p.add_argument(
        "--lambda-peak-contrast",
        type=float,
        default=0.0,
        metavar="W",
        help="HF target-peak contrast loss on log-STFT; encourages harmonic ridges instead of broadband HF shelves",
    )
    p.add_argument(
        "--lambda-peak-mag",
        type=float,
        default=0.0,
        metavar="W",
        help="HF target-peak logmag under-loss; raises missing harmonic ridge bins without rewarding all HF bins",
    )
    p.add_argument(
        "--peak-contrast-radius",
        type=int,
        default=3,
        help="Frequency-bin radius for target-peak masks used by --lambda-peak-contrast and --lambda-peak-mag",
    )
    p.add_argument(
        "--lambda-freq-ac",
        type=float,
        default=0.0,
        metavar="W",
        help="Frequency-axis autocorrelation loss on log-STFT; encourages harmonic spacing without semantic/GAN teacher",
    )
    p.add_argument(
        "--freq-ac-min-hz",
        type=float,
        default=300.0,
        metavar="HZ",
        help="Lowest spectral bin included in frequency-autocorrelation loss",
    )
    p.add_argument(
        "--freq-ac-lags-hz",
        type=str,
        default="60,80,100,125,160,200,250,315,400,500",
        metavar="HZ,...",
        help="Frequency lag list in Hz for --lambda-freq-ac",
    )
    p.add_argument(
        "--lambda-stationary-line",
        type=float,
        default=0.0,
        metavar="W",
        help="Penalize prediction-only stationary narrow spectral lines in mean-over-time log-STFT",
    )
    p.add_argument(
        "--stationary-line-min-hz",
        type=float,
        default=1000.0,
        metavar="HZ",
        help="Lowest bin included in stationary-line excess loss",
    )
    p.add_argument(
        "--stationary-line-radius",
        type=int,
        default=5,
        metavar="BINS",
        help="Frequency-neighborhood radius for stationary-line contrast",
    )
    p.add_argument(
        "--stationary-line-margin",
        type=float,
        default=0.08,
        metavar="M",
        help="Allowed prediction-over-target stationary-line contrast margin",
    )
    p.add_argument(
        "--lambda-harmonic-f0",
        type=float,
        default=0.0,
        metavar="W",
        help="Autocorrelation-derived F0 supervision weight for --decoder-harmonic-source",
    )
    p.add_argument(
        "--lambda-harmonic-amp",
        type=float,
        default=0.0,
        metavar="W",
        help="Autocorrelation-derived voicing/amplitude supervision weight for --decoder-harmonic-source",
    )
    p.add_argument(
        "--harmonic-f0-frame",
        type=int,
        default=512,
        metavar="N",
        help="Frame size in samples for harmonic-source autocorrelation F0 target",
    )
    p.add_argument(
        "--harmonic-f0-hop",
        type=int,
        default=256,
        metavar="N",
        help="Hop size in samples for harmonic-source autocorrelation F0 target",
    )
    p.add_argument(
        "--harmonic-f0-lags",
        type=str,
        default="32,36,40,45,50,56,63,71,80,90,101,113,127,143,160,180,202,226,254",
        metavar="SAMPLES,...",
        help="Candidate autocorrelation lags in samples for harmonic-source F0 target",
    )
    p.add_argument(
        "--harmonic-voicing-threshold",
        type=float,
        default=0.30,
        metavar="R",
        help="Normalized autocorrelation threshold for voiced harmonic-source frames",
    )
    p.add_argument("--hf-start-step", type=int, default=0, help="Keep HF losses disabled until this optimizer step")
    p.add_argument("--hf-ramp-steps", type=int, default=0, help="Linearly ramp HF losses to full weight over this many steps")
    p.add_argument(
        "--stft-grad-freq-weight",
        type=float,
        default=2.0,
        metavar="W",
        help="When --lambda-stft-grad>0: weight on ∂/∂f L1 (freq axis; higher → sharper horizontal harmonics / stripes)",
    )
    p.add_argument(
        "--stft-grad-time-weight",
        type=float,
        default=1.0,
        metavar="W",
        help="When --lambda-stft-grad>0: weight on ∂/∂t L1 (time axis); combined with freq weight as weighted average per scale",
    )
    p.add_argument(
        "--stft-ramp-steps",
        type=int,
        default=12_000,
        help="Linear ramp λ_stft from stft-ramp-start to 1.0× over this many steps (0=no ramp)",
    )
    p.add_argument(
        "--stft-ramp-start",
        type=float,
        default=0.12,
        help="Initial fraction of λ_stft at step 0 when ramping (0<.<=1); lower leaves more headroom for RVQ early",
    )
    p.add_argument(
        "--stft-scales",
        type=str,
        default=None,
        metavar="SPEC",
        help='Multi-scale STFT for loss: semicolon-separated "n_fft,hop" pairs, e.g. 512,128;1024,256;2048,512. '
        "Default: built-in three scales (includes 2048 FFT). See also --fast.",
    )
    p.add_argument(
        "--stft-scale-weights",
        type=str,
        default=None,
        metavar="W1,W2,…",
        help="Comma weights for each STFT scale (same count as resolved --stft-scales); "
        "loss uses sum(w_i·L_i)/sum(w_i). Omit for uniform. Example with hi-res focus: 1,1.5,2",
    )
    p.add_argument(
        "--spectral-batch-items",
        type=int,
        default=0,
        metavar="N",
        help="Limit spectral/mel losses to N items per micro-batch; 0 = full micro-batch",
    )
    p.add_argument(
        "--spectral-loss-every",
        type=int,
        default=1,
        metavar="N",
        help="Compute full spectral/mel/HF loss bundle every N optimizer steps and scale active spectral gradients by N",
    )
    p.add_argument(
        "--stft-large-min-fft",
        type=int,
        default=0,
        metavar="NFFT",
        help="Treat STFT scales with n_fft >= NFFT as large; 0 disables large-scale cycling",
    )
    p.add_argument(
        "--stft-large-every",
        type=int,
        default=1,
        metavar="N",
        help="Run large STFT scales every N optimizer steps",
    )
    p.add_argument(
        "--stft-hf-emphasis",
        type=float,
        default=1.0,
        metavar="G",
        help="HF weight on log-mag L1: per-bin weight 1+G·(f/Fmax)² (0=uniform mean). STFT-grad term unchanged.",
    )
    p.add_argument(
        "--mel-n-fft",
        type=int,
        default=1024,
        help="Single-scale STFT for mel losses (linear mag → mel bins → log; L1/L2 on log-mel)",
    )
    p.add_argument("--mel-hop", type=int, default=256, help="Hop length for mel STFT")
    p.add_argument("--n-mels", type=int, default=80, metavar="M", help="Number of mel bands")
    p.add_argument("--mel-fmin", type=float, default=0.0, help="Mel filter low edge (Hz)")
    p.add_argument(
        "--mel-fmax",
        type=float,
        default=None,
        metavar="HZ",
        help="Mel filter high edge (Hz); default Nyquist (sample_rate/2)",
    )
    p.add_argument(
        "--lambda-mel-l1",
        type=float,
        default=0.12,
        metavar="W",
        help="add W·λ_stft_eff·mean|Δlog mel| (0=off). Default 0.12 (L1-only mel); lower if recon smears",
    )
    p.add_argument(
        "--lambda-mel-l2",
        type=float,
        default=0.0,
        metavar="W",
        help="add W·λ_stft_eff·mean((Δlog mel)²) (0=off). Default 0 — use L1 only unless you know you need L2",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Faster training: use two STFT scales (512,128)+(1024,256) unless --stft-scales is set",
    )
    p.add_argument("--lambda-vq", type=float, default=5.0, help="Weight on VQ commit+codebook sum")
    p.add_argument(
        "--vq-loss",
        type=str,
        choices=("mse", "huber"),
        default="mse",
        help="Shape for VQ commit/codebook match loss; huber limits outlier gradients from latent spikes",
    )
    p.add_argument(
        "--vq-huber-delta",
        type=float,
        default=1.0,
        metavar="D",
        help="Huber delta for --vq-loss huber (larger approaches MSE; must be >0)",
    )
    p.add_argument(
        "--vq-loss-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize VQ match residuals by per-frame residual RMS before computing --vq-loss",
    )
    p.add_argument(
        "--vq-loss-norm-eps",
        type=float,
        default=1e-4,
        metavar="EPS",
        help="Epsilon inside residual-RMS normalization for --vq-loss-normalize",
    )
    p.add_argument(
        "--lambda-entropy",
        type=float,
        default=0.0,
        help="Per-position softmax entropy bonus (often misleading; 0=off). Prefer --lambda-marginal.",
    )
    p.add_argument(
        "--lambda-marginal",
        type=float,
        default=0.35,
        help="Batch marginal diversity: adds λ·(log K − H_marg) (0=off; same gradients as old −λ·H)",
    )
    p.add_argument(
        "--marginal-tau",
        type=float,
        default=0.04,
        help="τ for marginal softmax(-dist/τ); lower → sharper p̄ → H tracks hard code usage (try 0.03–0.06; too high masks collapse)",
    )
    p.add_argument(
        "--marginal-boost-steps",
        type=int,
        default=24_000,
        help="Linear decay of λ_marg from marginal-boost-mult×base down to base over this many steps (0=off)",
    )
    p.add_argument(
        "--marginal-boost-mult",
        type=float,
        default=2.5,
        help="Step-0 multiplier for λ_marg during boost (>=1); fights fast post-init collapse",
    )
    p.add_argument(
        "--vq-reset-every",
        type=int,
        default=1000,
        help="Every N steps: reset unused codebook rows when a stage is collapsed (0=off; lower=higher overhead)",
    )
    p.add_argument(
        "--vq-reset-collapse-frac",
        type=float,
        default=0.42,
        help="Per stage: reset dead rows if len(unique)/K < this (e.g. 0.42 resets unless ≥~54/128 codes used; 0.08 only <~10)",
    )
    p.add_argument(
        "--vq-reset-noise",
        type=float,
        default=0.12,
        help="Gaussian noise scale on dead-code replacements (× residual batch std); 0=off",
    )
    p.add_argument(
        "--vq-reset-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If unique/K below ~35%% of vq-reset-collapse-frac after fill, permute all codebook rows",
    )
    p.add_argument(
        "--vq-reset-kmeans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dead-code refill: K-means centroids on batch residuals (--no-vq-reset-kmeans = random picks)",
    )
    p.add_argument(
        "--vq-reset-full-refresh-max-unique",
        type=int,
        default=4,
        metavar="U",
        help="If unique indices in a stage ≤U, rewrite all K codebook rows (fixes u=1 when dead-fill skips the winner)",
    )
    p.add_argument(
        "--vq-reset-log-every",
        type=int,
        default=5000,
        help="Print [vq-reset] at most when step%%N==0 (0=every reset)",
    )
    p.add_argument(
        "--vq-cosine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cosine/spherical RVQ: same n_q×K → same nominal bitrate; assign by direction, z_q=ê‖r‖. Old .npz from Euclidean runs: --no-vq-cosine",
    )
    p.add_argument("--vq-beta", type=float, default=1.35, help="Per-stage commitment β (higher → less collapse)")
    p.add_argument(
        "--lambda-cos",
        type=float,
        default=0.15,
        help="Weight on (1 − cosine); waveform shape alignment toward cos → 100%% (default 0.15; 0=off)",
    )
    p.add_argument(
        "--cos-hinge",
        type=float,
        default=0.0,
        help="Weight on max(0, cos_target − cos); set with --cos-target 0.9 to push 90%%+",
    )
    p.add_argument(
        "--cos-target",
        type=float,
        default=0.9,
        help="Floor for cosine hinge (default 0.9)",
    )
    p.add_argument("--lambda-sisdr", type=float, default=0.0, help="Differentiable negative log SI-SDR waveform loss weight")
    p.add_argument("--lambda-preemph", type=float, default=0.0, help="Pre-emphasized waveform L1 loss weight")
    p.add_argument("--preemph-coef", type=float, default=0.97, help="Pre-emphasis coefficient for --lambda-preemph")
    p.add_argument(
        "--lambda-multidelta",
        type=float,
        default=0.0,
        help="Multi-lag finite-difference waveform L1 weight; HF/time anchor that does not reward spectral-floor haze",
    )
    p.add_argument(
        "--multidelta-lags",
        type=str,
        default="1,2,4,8",
        metavar="LAGS",
        help="Comma-separated positive sample lags for --lambda-multidelta",
    )
    p.add_argument(
        "--lambda-ae-anchor-time",
        type=float,
        default=0.0,
        help="Train continuous z->decoder L1 anchor alongside hard RVQ reconstruction",
    )
    p.add_argument(
        "--lambda-ae-anchor-cos",
        type=float,
        default=0.0,
        help="Train continuous z->decoder waveform cosine anchor alongside hard RVQ reconstruction",
    )
    p.add_argument(
        "--lambda-sc",
        type=float,
        default=1.0,
        metavar="λ",
        help="Spectral Convergence weight: W·λ_stft_eff·‖|S(ŷ)|−|S(y)|‖_F/‖|S(y)|‖_F per scale (GAN-free HF emphasis). 0=off.",
    )
    p.add_argument(
        "--lambda-complex-stft",
        type=float,
        default=0.1,
        metavar="λ",
        help="Complex STFT L1 weight on Re/Im (captures phase). W·λ_stft_eff·0.5·(L1(Re)+L1(Im)) per scale. 0=off.",
    )
    p.add_argument(
        "--lambda-mag-l1",
        type=float,
        default=0.0,
        metavar="λ",
        help="Extra mean L1 on linear STFT magnitudes (same scales as log-STFT; 0=off)",
    )
    p.add_argument(
        "--activation",
        type=str,
        default="gelu",
        choices=("gelu", "snake", "snake_beta", "harmonic_beta", "periodic_gelu"),
        help="Encoder/decoder nonlinearity (snake*=DAC sin²; harmonic_beta 2-scale; periodic_gelu GELU+sin²)",
    )
    p.add_argument(
        "--rvq-code-dim",
        type=int,
        default=0,
        metavar="d",
        help="Factorized RVQ: project latent to d before VQ (0 = full latent_dim)",
    )
    p.add_argument(
        "--vq-ema-decay",
        type=float,
        default=0.0,
        metavar="ρ",
        help="EMA decay for codebook rows in (0,1), e.g. 0.99 (0=disabled)",
    )
    p.add_argument(
        "--vq-ema-every",
        type=int,
        default=1,
        metavar="N",
        help="Run CPU/NumPy VQ EMA codebook refresh every N optimizer steps (1=every step; higher=faster)",
    )
    p.add_argument(
        "--decoder-upsample",
        type=str,
        default="transpose",
        choices=("transpose", "repeat_conv"),
        help="Decoder upsampling: ConvTranspose1d or repeat×2+Conv (also forced when --causal)",
    )
    p.add_argument(
        "--decoder-transpose-kernel",
        type=int,
        default=7,
        metavar="K",
        help="ConvTranspose1d upsample kernel. Even K, e.g. 8, gives even stride-2 overlap; legacy is 7.",
    )
    p.add_argument(
        "--decoder-band-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decoder predicts low/mid/high branches filtered by a fixed FIR bank, then summed (same RVQ bitrate)",
    )
    p.add_argument(
        "--band-split-cutoffs-hz",
        type=str,
        default="2500,5000",
        metavar="LOW,MID",
        help="Two band split cutoffs in Hz for --decoder-band-split and --lambda-band-l1",
    )
    p.add_argument(
        "--band-split-taps",
        type=int,
        default=129,
        metavar="N",
        help="Odd FIR tap count for band split filters",
    )
    p.add_argument(
        "--decoder-band-head-depth",
        type=int,
        default=0,
        metavar="N",
        help="Extra independent Conv1d blocks per low/mid/high decoder head before fixed FIR filtering",
    )
    p.add_argument(
        "--decoder-band-latent-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decode three rvq-band-split latent channel groups with separate decoder towers before fixed FIR sum",
    )
    p.add_argument(
        "--decoder-harmonic-source",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decoder adds a differentiable voiced source-filter branch with harmonic partials; RVQ bitrate is unchanged",
    )
    p.add_argument(
        "--decoder-harmonic-partials",
        type=int,
        default=6,
        metavar="N",
        help="Number of 1/k-weighted harmonic partials in --decoder-harmonic-source",
    )
    p.add_argument(
        "--decoder-harmonic-fmin-hz",
        type=float,
        default=60.0,
        metavar="HZ",
        help="Minimum predicted F0 for --decoder-harmonic-source",
    )
    p.add_argument(
        "--decoder-harmonic-fmax-hz",
        type=float,
        default=450.0,
        metavar="HZ",
        help="Maximum predicted F0 for --decoder-harmonic-source",
    )
    p.add_argument(
        "--decoder-harmonic-gain",
        type=float,
        default=0.25,
        metavar="G",
        help="Gain for the harmonic excitation branch before tanh",
    )
    p.add_argument(
        "--decoder-harmonic-amp-bias",
        type=float,
        default=-2.0,
        metavar="B",
        help="Additive bias before sigmoid amplitude gate in --decoder-harmonic-source; negative starts the source quieter",
    )
    p.add_argument(
        "--decoder-harmonic-rolloff",
        type=float,
        default=1.0,
        metavar="P",
        help="Partial amplitude rolloff for --decoder-harmonic-source: weight(k)=1/k^P",
    )
    p.add_argument(
        "--decoder-harmonic-mode",
        type=str,
        default="additive",
        choices=("additive", "source_filter"),
        help="Harmonic decoder topology. source_filter constrains mid/high bands to deterministic harmonic excitation plus small conv residual",
    )
    p.add_argument(
        "--decoder-harmonic-control-smooth",
        type=int,
        default=0,
        metavar="N",
        help="Odd moving-average taps for predicted F0/amplitude/envelopes; 0 disables smoothing",
    )
    p.add_argument(
        "--decoder-harmonic-env-bias",
        type=float,
        default=0.0,
        metavar="B",
        help="Additive bias before sigmoid mid/high harmonic envelopes in source_filter mode",
    )
    p.add_argument(
        "--decoder-harmonic-band-weights",
        type=str,
        default="0.0,0.35,1.0",
        metavar="LOW,MID,HIGH",
        help="When harmonic source and band split are both enabled, route harmonic excitation into these filtered bands",
    )
    p.add_argument(
        "--decoder-harmonic-residual-band-weights",
        type=str,
        default="1.0,0.25,0.05",
        metavar="LOW,MID,HIGH",
        help="In source_filter mode, attenuate generic conv residual bands before adding harmonic mid/high source",
    )
    p.add_argument(
        "--decoder-harmonic-groups",
        type=int,
        default=0,
        metavar="N",
        help="Optional learned partial-group gates for --decoder-harmonic-source; 0 keeps fixed 1/k partial weights",
    )
    p.add_argument(
        "--decoder-harmonic-group-bias",
        type=float,
        default=-1.5,
        metavar="B",
        help="Additive bias before sigmoid partial-group gates; negative starts harmonic groups quieter",
    )
    p.add_argument(
        "--lambda-band-l1",
        type=float,
        default=0.0,
        metavar="λ",
        help="Band-normalized waveform L1 weight (low/mid/high filtered, target-energy normalized)",
    )
    p.add_argument(
        "--band-l1-weights",
        type=str,
        default="0.25,1.0,1.5",
        metavar="LOW,MID,HIGH",
        help="Weights for band-normalized L1",
    )
    p.add_argument(
        "--band-l1-floor",
        type=float,
        default=0.015,
        metavar="A",
        help="Target-band magnitude floor for band-normalized L1",
    )
    p.add_argument(
        "--lambda-band-branch-l1",
        type=float,
        default=0.0,
        metavar="λ",
        help="Train decoder low/mid/high branches against matching target bands before summing",
    )
    p.add_argument(
        "--band-branch-weights",
        type=str,
        default="0.10,1.0,2.0",
        metavar="LOW,MID,HIGH",
        help="Weights for branch-supervised band L1",
    )
    p.add_argument("--causal", action="store_true", help="Causal encoder/decoder convs (left pad only)")
    p.add_argument("--bf16", action="store_true", help="Run codec in bfloat16 (STFT fp32)")
    p.add_argument(
        "--compile-loss",
        action="store_true",
        help="Try mx.compile on multi-scale STFT spectral block (experimental)",
    )
    p.add_argument(
        "--full-checkpoint",
        action="store_true",
        help="Also save/load ckpt_stepN.safetensors (+ meta JSON) with weights + optimizer state",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        metavar="N",
        help="Run eval every N steps: holdout if --eval-dir is set, otherwise a train-batch probe (0=off)",
    )
    p.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Holdout audio root for eval metrics; files are selected once with --eval-seed",
    )
    p.add_argument(
        "--eval-samples",
        "--eval-clips",
        dest="eval_clips",
        type=int,
        default=4,
        metavar="N",
        help="Number of holdout files for --eval-dir (alias: --eval-clips)",
    )
    p.add_argument(
        "--eval-seconds",
        type=float,
        default=3.0,
        metavar="S",
        help="Seconds per holdout file for eval (0 = full file)",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        metavar="SEED",
        help="Seed for deterministic holdout file selection",
    )
    p.add_argument(
        "--log-mlx-tsv",
        type=str,
        default="",
        metavar="PATH",
        help="Append eval / checkpoint metrics as TSV (empty = disabled)",
    )
    p.add_argument(
        "--results-tsv",
        type=str,
        default="",
        metavar="PATH",
        help="Append one SYSTEM.md-style results row on checkpoint (empty = disabled)",
    )
    p.add_argument(
        "--adaptive-mode",
        type=str,
        default="none",
        choices=("none", "bwe_stub", "fps_stub"),
        help="Stub for nominal-bitrate bookkeeping (see sirencodec.adaptive)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Folder with audio: recursive *.wav, *.flac, *.ogg, *.mp3 (default: data/cv-corpus)",
    )
    p.add_argument(
        "--librispeech",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use default local corpus when no --data-dir is set; --no-librispeech for synthetic data",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--log-loss-ema-beta",
        type=float,
        default=0.98,
        metavar="B",
        help="EMA of raw train loss and VQ loss in logs only (0=off, diagnostic only)",
    )
    p.add_argument(
        "--log-cos-ema-beta",
        type=float,
        default=0.99,
        metavar="B",
        help="EMA of cos%% in log: ema←B·ema+(1−B)·cos (0=off, typical 0.95–0.995)",
    )
    p.add_argument(
        "--best-holdout-metric",
        type=str,
        default="sisdr_db",
        metavar="M",
        help="Metric used to update checkpoints/best_holdout.json (sisdr_db, pesq_wb, stoi, lsd_db, cos)",
    )
    p.add_argument(
        "--spectrogram-every",
        type=int,
        default=1000,
        help="Save orig/recon spectrogram PNG (and WAV if save-audio) every N steps (0=disable)",
    )
    p.add_argument(
        "--spectrogram-dir",
        type=str,
        default="mlx_spectrograms",
        help="Output folder for spectrogram PNGs",
    )
    p.add_argument(
        "--save-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With each spectrogram PNG, save *_orig.wav and *_recon.wav (first batch item); --no-save-audio to disable",
    )
    p.add_argument(
        "--spectrogram-seconds",
        type=float,
        default=8.0,
        help="Duration (s) for PNG/WAV export only; 0 = use training batch length. Does not change --segment.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save model weights (codec_stepN.npz) every N steps; 0 = off. When N>0, also saves the last step.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="mlx_checkpoints",
        help="Output folder for MLX weight checkpoints",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="__latest__",
        default=None,
        metavar="PATH",
        help="Load weights and continue: optional path to codec_stepN.npz; if flag given alone, use highest N in --checkpoint-dir",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Print per-block timing (data/D/G-fwd+bwd/optim/ema/vq-reset) every --log-every; small overhead from mx.eval syncs",
    )
    p.set_defaults(**argparse_defaults_from_config())
    argv = sys.argv[1:]
    user_stft_scales = any(a == "--stft-scales" or a.startswith("--stft-scales=") for a in argv)
    user_stft_weights = any(a == "--stft-scale-weights" or a.startswith("--stft-scale-weights=") for a in argv)
    args = p.parse_args()

    if not args.data_dir and args.librispeech:
        args.data_dir = str(Path("data") / "cv-corpus")

    if args.spectrogram_seconds < 0:
        print("--spectrogram-seconds must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.eval_every < 0:
        print("--eval-every must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.eval_clips < 1:
        print("--eval-samples/--eval-clips must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.eval_seconds < 0:
        print("--eval-seconds must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.checkpoint_every < 0:
        print("--checkpoint-every must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.load_audio_threads < 0 or args.load_audio_threads > 128:
        print("--load-audio-threads must be in [0, 128]", file=sys.stderr)
        sys.exit(1)
    if args.grad_accum_steps < 1:
        print("--grad-accum-steps must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.vq_ema_every < 1:
        print("--vq-ema-every must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.vq_huber_delta <= 0:
        print("--vq-huber-delta must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.vq_loss_norm_eps <= 0:
        print("--vq-loss-norm-eps must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.lr_min_ratio < 0 or args.lr_min_ratio > 1.0:
        print("--lr-min-ratio must be in [0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.lr_warmup_steps < 0:
        print("--lr-warmup-steps must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 < args.lr_plateau_factor < 1.0):
        print("--lr-plateau-factor must be in (0, 1)", file=sys.stderr)
        sys.exit(1)
    if args.lr_plateau_patience < 1:
        print("--lr-plateau-patience must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.lr_plateau_threshold < 0.0:
        print("--lr-plateau-threshold must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.lr_plateau_ema < 1.0):
        print("--lr-plateau-ema must be in [0, 1)", file=sys.stderr)
        sys.exit(1)
    if args.lr_plateau_cooldown < 0:
        print("--lr-plateau-cooldown must be >= 0", file=sys.stderr)
        sys.exit(1)

    if args.n_codebooks < 1 and not args.ae_only:
        print("--n-codebooks must be >= 1 unless --ae-only", file=sys.stderr)
        sys.exit(1)

    try:
        enc_ch = tuple(int(x.strip()) for x in args.enc_channels.split(",") if x.strip())
    except ValueError:
        enc_ch = ()
    if len(enc_ch) < 1:
        print("--enc-channels must list at least one integer width (comma-separated)", file=sys.stderr)
        sys.exit(1)

    if args.lambda_stft_grad < 0:
        print("--lambda-stft-grad must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_stft_cos < 0:
        print("--lambda-stft-cos must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_stft_excess < 0:
        print("--lambda-stft-excess must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.stft_excess_margin < 0:
        print("--stft-excess-margin must be >= 0", file=sys.stderr)
        sys.exit(1)
    if (
        args.lambda_hf_under < 0
        or args.lambda_hf_sc < 0
        or args.lambda_hf_complex < 0
        or args.lambda_peak_contrast < 0
        or args.lambda_peak_mag < 0
        or args.lambda_freq_ac < 0
        or args.lambda_stationary_line < 0
        or args.lambda_harmonic_f0 < 0
        or args.lambda_harmonic_amp < 0
    ):
        print(
            "--lambda-hf-under, --lambda-hf-sc, --lambda-hf-complex, --lambda-peak-contrast, --lambda-peak-mag, --lambda-freq-ac, --lambda-stationary-line, --lambda-harmonic-f0, and --lambda-harmonic-amp must be >= 0",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.hf_min_hz < 0 or args.hf_min_hz >= 0.5 * Config().sample_rate:
        print("--hf-min-hz must be >= 0 and below Nyquist", file=sys.stderr)
        sys.exit(1)
    if args.hf_under_margin < 0:
        print("--hf-under-margin must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.hf_start_step < 0 or args.hf_ramp_steps < 0:
        print("--hf-start-step and --hf-ramp-steps must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.peak_contrast_radius < 1:
        print("--peak-contrast-radius must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.freq_ac_min_hz < 0 or args.freq_ac_min_hz >= 0.5 * Config().sample_rate:
        print("--freq-ac-min-hz must be >= 0 and below Nyquist", file=sys.stderr)
        sys.exit(1)
    try:
        if isinstance(args.freq_ac_lags_hz, (tuple, list)):
            freq_ac_lags_hz = tuple(float(x) for x in args.freq_ac_lags_hz)
        else:
            freq_ac_lags_hz = parse_float_list_arg(str(args.freq_ac_lags_hz))
    except ValueError as e:
        print(f"--freq-ac-lags-hz: {e}", file=sys.stderr)
        sys.exit(1)
    if len(freq_ac_lags_hz) < 1 or any(x <= 0.0 for x in freq_ac_lags_hz):
        print("--freq-ac-lags-hz must contain at least one positive lag", file=sys.stderr)
        sys.exit(1)
    if args.stationary_line_min_hz < 0 or args.stationary_line_min_hz >= 0.5 * Config().sample_rate:
        print("--stationary-line-min-hz must be >= 0 and below Nyquist", file=sys.stderr)
        sys.exit(1)
    if args.stationary_line_radius < 1:
        print("--stationary-line-radius must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.stationary_line_margin < 0:
        print("--stationary-line-margin must be >= 0", file=sys.stderr)
        sys.exit(1)
    try:
        if isinstance(args.harmonic_f0_lags, (tuple, list)):
            harmonic_f0_lags = tuple(int(x) for x in args.harmonic_f0_lags)
        else:
            harmonic_f0_lags = parse_positive_int_list_arg(str(args.harmonic_f0_lags))
    except ValueError as e:
        print(f"--harmonic-f0-lags: {e}", file=sys.stderr)
        sys.exit(1)
    if args.harmonic_f0_frame < 16:
        print("--harmonic-f0-frame must be >= 16", file=sys.stderr)
        sys.exit(1)
    if args.harmonic_f0_hop < 1:
        print("--harmonic-f0-hop must be >= 1", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.harmonic_voicing_threshold < 1.0):
        print("--harmonic-voicing-threshold must be in [0, 1)", file=sys.stderr)
        sys.exit(1)
    if any(int(l) >= int(args.harmonic_f0_frame) for l in harmonic_f0_lags):
        print("--harmonic-f0-lags must be smaller than --harmonic-f0-frame", file=sys.stderr)
        sys.exit(1)
    if args.stft_grad_freq_weight < 0 or args.stft_grad_time_weight < 0:
        print("--stft-grad-freq-weight and --stft-grad-time-weight must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_stft_grad > 0 and args.stft_grad_freq_weight <= 0 and args.stft_grad_time_weight <= 0:
        print("with --lambda-stft-grad>0, at least one of --stft-grad-freq-weight / --stft-grad-time-weight must be >0", file=sys.stderr)
        sys.exit(1)
    if args.stft_hf_emphasis < 0:
        print("--stft-hf-emphasis must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.spectral_batch_items < 0:
        print("--spectral-batch-items must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.spectral_loss_every < 1:
        print("--spectral-loss-every must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.stft_large_min_fft < 0:
        print("--stft-large-min-fft must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.stft_large_every < 1:
        print("--stft-large-every must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.lambda_mel_l1 < 0 or args.lambda_mel_l2 < 0:
        print("--lambda-mel-l1 and --lambda-mel-l2 must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.mel_n_fft < 2 or args.mel_n_fft % 2 != 0:
        print("--mel-n-fft must be a positive even integer", file=sys.stderr)
        sys.exit(1)
    if args.mel_hop < 1:
        print("--mel-hop must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.n_mels < 1:
        print("--n-mels must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.mel_fmin < 0:
        print("--mel-fmin must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.log_loss_ema_beta < 0 or args.log_loss_ema_beta >= 1.0:
        print("--log-loss-ema-beta must be in [0, 1) (0 = disable EMA)", file=sys.stderr)
        sys.exit(1)
    if args.log_cos_ema_beta < 0 or args.log_cos_ema_beta >= 1.0:
        print("--log-cos-ema-beta must be in [0, 1) (0 = disable EMA)", file=sys.stderr)
        sys.exit(1)
    best_holdout_metric = _canonical_eval_metric_name(args.best_holdout_metric)
    if best_holdout_metric not in _EVAL_METRIC_ALIASES.values():
        print(
            "--best-holdout-metric must be one of: "
            + ", ".join(sorted(set(_EVAL_METRIC_ALIASES))),
            file=sys.stderr,
        )
        sys.exit(1)
    if not (0.0 < args.stft_ramp_start <= 1.0):
        print("--stft-ramp-start must be in (0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.stft_ramp_steps < 0:
        print("--stft-ramp-steps must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.marginal_boost_mult < 1.0:
        print("--marginal-boost-mult must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.stride1_blocks_per_scale < 0:
        print("--stride1-blocks-per-scale must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.decoder_refine_blocks_per_scale < 0:
        print("--decoder-refine-blocks-per-scale must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.vq_reset_full_refresh_max_unique < 1:
        print("--vq-reset-full-refresh-max-unique must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.latent_temporal_depth < 0 or args.latent_temporal_post_depth < 0:
        print("--latent-temporal-depth and --latent-temporal-post-depth must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.self_attention_depth < 0 or args.self_attention_post_depth < 0:
        print("--self-attention-depth and --self-attention-post-depth must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.self_attention_heads < 1:
        print("--self-attention-heads must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.state_space_depth < 0 or args.state_space_post_depth < 0:
        print("--state-space-depth and --state-space-post-depth must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.state_space_state_dim < 1:
        print("--state-space-state-dim must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.state_space_expand < 1:
        print("--state-space-expand must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.decoder_transpose_kernel < 2:
        print("--decoder-transpose-kernel must be >= 2", file=sys.stderr)
        sys.exit(1)
    try:
        if isinstance(args.band_split_cutoffs_hz, (tuple, list)):
            band_split_cutoffs = tuple(float(x) for x in args.band_split_cutoffs_hz)
        else:
            band_split_cutoffs = parse_float_list_arg(str(args.band_split_cutoffs_hz))
    except ValueError as e:
        print(f"--band-split-cutoffs-hz: {e}", file=sys.stderr)
        sys.exit(1)
    if len(band_split_cutoffs) != 2:
        print("--band-split-cutoffs-hz must contain exactly two cutoffs", file=sys.stderr)
        sys.exit(1)
    nyq = 0.5 * float(Config().sample_rate)
    if not (0.0 < band_split_cutoffs[0] < band_split_cutoffs[1] < nyq):
        print("--band-split-cutoffs-hz must satisfy 0 < low < high < Nyquist", file=sys.stderr)
        sys.exit(1)
    if args.band_split_taps < 3 or args.band_split_taps % 2 == 0:
        print("--band-split-taps must be odd and >= 3", file=sys.stderr)
        sys.exit(1)
    if args.decoder_band_head_depth < 0:
        print("--decoder-band-head-depth must be >= 0", file=sys.stderr)
        sys.exit(1)
    if bool(args.decoder_band_latent_split) and not bool(args.decoder_band_split):
        print("--decoder-band-latent-split requires --decoder-band-split", file=sys.stderr)
        sys.exit(1)
    if bool(args.decoder_band_latent_split) and bool(args.decoder_harmonic_source):
        print("--decoder-band-latent-split is incompatible with --decoder-harmonic-source", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_partials < 1:
        print("--decoder-harmonic-partials must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_fmin_hz <= 0 or args.decoder_harmonic_fmax_hz <= args.decoder_harmonic_fmin_hz:
        print("--decoder-harmonic-fmin-hz must be > 0 and below --decoder-harmonic-fmax-hz", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_fmax_hz >= nyq:
        print("--decoder-harmonic-fmax-hz must be below Nyquist", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_gain < 0:
        print("--decoder-harmonic-gain must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_rolloff <= 0:
        print("--decoder-harmonic-rolloff must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_control_smooth < 0:
        print("--decoder-harmonic-control-smooth must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_control_smooth > 1 and args.decoder_harmonic_control_smooth % 2 == 0:
        print("--decoder-harmonic-control-smooth must be odd when > 1", file=sys.stderr)
        sys.exit(1)
    try:
        if isinstance(args.decoder_harmonic_band_weights, (tuple, list)):
            harmonic_band_weights = tuple(float(x) for x in args.decoder_harmonic_band_weights)
        else:
            harmonic_band_weights = parse_float_list_arg(str(args.decoder_harmonic_band_weights))
    except ValueError as e:
        print(f"--decoder-harmonic-band-weights: {e}", file=sys.stderr)
        sys.exit(1)
    if len(harmonic_band_weights) != 3:
        print("--decoder-harmonic-band-weights must contain exactly three weights", file=sys.stderr)
        sys.exit(1)
    if any(x < 0.0 for x in harmonic_band_weights) or sum(harmonic_band_weights) <= 0.0:
        print("--decoder-harmonic-band-weights must be non-negative and sum to > 0", file=sys.stderr)
        sys.exit(1)
    try:
        if isinstance(args.decoder_harmonic_residual_band_weights, (tuple, list)):
            harmonic_residual_band_weights = tuple(float(x) for x in args.decoder_harmonic_residual_band_weights)
        else:
            harmonic_residual_band_weights = parse_float_list_arg(str(args.decoder_harmonic_residual_band_weights))
    except ValueError as e:
        print(f"--decoder-harmonic-residual-band-weights: {e}", file=sys.stderr)
        sys.exit(1)
    if len(harmonic_residual_band_weights) != 3:
        print("--decoder-harmonic-residual-band-weights must contain exactly three weights", file=sys.stderr)
        sys.exit(1)
    if any(x < 0.0 for x in harmonic_residual_band_weights):
        print("--decoder-harmonic-residual-band-weights must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_groups < 0:
        print("--decoder-harmonic-groups must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.decoder_harmonic_mode == "source_filter" and not (
        bool(args.decoder_harmonic_source) and bool(args.decoder_band_split)
    ):
        print("--decoder-harmonic-mode source_filter requires --decoder-harmonic-source and --decoder-band-split", file=sys.stderr)
        sys.exit(1)
    if (args.lambda_harmonic_f0 > 0 or args.lambda_harmonic_amp > 0) and not bool(args.decoder_harmonic_source):
        print("--lambda-harmonic-f0/--lambda-harmonic-amp require --decoder-harmonic-source", file=sys.stderr)
        sys.exit(1)
    if bool(args.causal) and bool(args.decoder_band_split):
        print("--decoder-band-split is centered-FIR and incompatible with --causal", file=sys.stderr)
        sys.exit(1)
    if args.lambda_band_l1 < 0:
        print("--lambda-band-l1 must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_band_branch_l1 < 0:
        print("--lambda-band-branch-l1 must be >= 0", file=sys.stderr)
        sys.exit(1)
    try:
        if isinstance(args.band_l1_weights, (tuple, list)):
            band_l1_weights = tuple(float(x) for x in args.band_l1_weights)
        else:
            band_l1_weights = parse_float_list_arg(str(args.band_l1_weights))
    except ValueError as e:
        print(f"--band-l1-weights: {e}", file=sys.stderr)
        sys.exit(1)
    if len(band_l1_weights) != 3:
        print("--band-l1-weights must contain exactly three weights", file=sys.stderr)
        sys.exit(1)
    if any(x < 0.0 for x in band_l1_weights) or sum(band_l1_weights) <= 0.0:
        print("--band-l1-weights must be non-negative and sum to > 0", file=sys.stderr)
        sys.exit(1)
    try:
        if isinstance(args.band_branch_weights, (tuple, list)):
            band_branch_weights = tuple(float(x) for x in args.band_branch_weights)
        else:
            band_branch_weights = parse_float_list_arg(str(args.band_branch_weights))
    except ValueError as e:
        print(f"--band-branch-weights: {e}", file=sys.stderr)
        sys.exit(1)
    if len(band_branch_weights) != 3:
        print("--band-branch-weights must contain exactly three weights", file=sys.stderr)
        sys.exit(1)
    if any(x < 0.0 for x in band_branch_weights) or sum(band_branch_weights) <= 0.0:
        print("--band-branch-weights must be non-negative and sum to > 0", file=sys.stderr)
        sys.exit(1)
    if args.band_l1_floor < 0:
        print("--band-l1-floor must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_band_branch_l1 > 0 and not bool(args.decoder_band_split):
        print("--lambda-band-branch-l1 requires --decoder-band-split", file=sys.stderr)
        sys.exit(1)
    if (args.self_attention_depth > 0 or args.self_attention_post_depth > 0) and args.latent_dim % args.self_attention_heads != 0:
        print("--latent-dim must be divisible by --self-attention-heads when SelfAttention1D is enabled", file=sys.stderr)
        sys.exit(1)
    if args.self_attention_depth > 0 and args.state_space_depth > 0:
        print("--self-attention-depth and --state-space-depth are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if args.self_attention_post_depth > 0 and args.state_space_post_depth > 0:
        print("--self-attention-post-depth and --state-space-post-depth are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if bool(args.causal) and bool(args.state_space_bidirectional) and (
        args.state_space_depth > 0 or args.state_space_post_depth > 0
    ):
        print("--state-space-bidirectional is incompatible with --causal when state-space blocks are enabled", file=sys.stderr)
        sys.exit(1)
    if args.lambda_sc < 0:
        print("--lambda-sc must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_complex_stft < 0:
        print("--lambda-complex-stft must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_sisdr < 0 or args.lambda_preemph < 0 or args.lambda_multidelta < 0:
        print("--lambda-sisdr, --lambda-preemph, and --lambda-multidelta must be >= 0", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.preemph_coef < 1.0):
        print("--preemph-coef must be >= 0 and < 1", file=sys.stderr)
        sys.exit(1)
    try:
        if isinstance(args.multidelta_lags, (tuple, list)):
            multidelta_lags = tuple(int(x) for x in args.multidelta_lags)
            if any(x < 1 for x in multidelta_lags):
                raise ValueError("expected positive integers")
        else:
            multidelta_lags = parse_positive_int_list_arg(str(args.multidelta_lags))
    except ValueError as e:
        print(f"--multidelta-lags: {e}", file=sys.stderr)
        sys.exit(1)
    if args.lambda_ae_anchor_time < 0 or args.lambda_ae_anchor_cos < 0:
        print("--lambda-ae-anchor-time and --lambda-ae-anchor-cos must be >= 0", file=sys.stderr)
        sys.exit(1)

    codebook_sizes_arg: tuple[int, ...] | None = None
    if args.codebook_sizes:
        try:
            codebook_sizes_arg = parse_codebook_sizes_arg(args.codebook_sizes)
        except ValueError as e:
            print(f"--codebook-sizes: {e}", file=sys.stderr)
            sys.exit(1)
        if len(codebook_sizes_arg) != args.n_codebooks:
            print(
                f"--codebook-sizes: expected {args.n_codebooks} values (got {len(codebook_sizes_arg)})",
                file=sys.stderr,
            )
            sys.exit(1)
    rvq_band_splits_arg: tuple[int, ...] | None = None
    if args.rvq_band_splits:
        try:
            rvq_band_splits_arg = parse_positive_int_list_arg(args.rvq_band_splits)
        except ValueError as e:
            print(f"--rvq-band-splits: {e}", file=sys.stderr)
            sys.exit(1)
        if len(rvq_band_splits_arg) < 2:
            print("--rvq-band-splits must contain at least two groups", file=sys.stderr)
            sys.exit(1)
        if sum(rvq_band_splits_arg) != args.n_codebooks:
            print(
                f"--rvq-band-splits must sum to --n-codebooks ({args.n_codebooks}), got {sum(rvq_band_splits_arg)}",
                file=sys.stderr,
            )
            sys.exit(1)
    if bool(args.decoder_band_latent_split) and (rvq_band_splits_arg is None or len(rvq_band_splits_arg) != 3):
        print("--decoder-band-latent-split requires --rvq-band-splits with exactly three groups", file=sys.stderr)
        sys.exit(1)

    if user_stft_scales and args.stft_scales:
        try:
            stft_scales_resolved = parse_stft_scales_arg(args.stft_scales)
        except ValueError as e:
            print(f"--stft-scales: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.fast:
        stft_scales_resolved = FAST_STFT_SCALES
    else:
        stft_scales_resolved = Config().stft_scales

    stft_scale_weights_resolved: tuple[float, ...] | None = None
    if user_stft_weights and args.stft_scale_weights is not None:
        try:
            stft_scale_weights_resolved = parse_stft_scale_weights_arg(args.stft_scale_weights)
        except ValueError as e:
            print(f"--stft-scale-weights: {e}", file=sys.stderr)
            sys.exit(1)
        if len(stft_scale_weights_resolved) != len(stft_scales_resolved):
            print(
                f"--stft-scale-weights: expected {len(stft_scales_resolved)} values (one per STFT scale), "
                f"got {len(stft_scale_weights_resolved)}",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.fast and not user_stft_scales:
        stft_scale_weights_resolved = None
    else:
        stft_scale_weights_resolved = Config().stft_scale_weights

    cfg = Config(
        steps=args.steps,
        batch=args.batch,
        load_audio_threads=args.load_audio_threads,
        prefetch_audio=bool(args.prefetch_audio),
        segment=args.segment,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_min_ratio=args.lr_min_ratio,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_threshold=args.lr_plateau_threshold,
        lr_plateau_ema=args.lr_plateau_ema,
        lr_plateau_cooldown=args.lr_plateau_cooldown,
        grad_clip_norm=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        ae_only=args.ae_only,
        enc_channels=enc_ch,
        stride1_blocks_per_scale=args.stride1_blocks_per_scale,
        decoder_refine_blocks_per_scale=args.decoder_refine_blocks_per_scale,
        latent_dim=args.latent_dim,
        pre_vq_layernorm=args.pre_vq_layernorm,
        latent_temporal_depth=args.latent_temporal_depth,
        latent_temporal_post_depth=args.latent_temporal_post_depth,
        self_attention_depth=args.self_attention_depth,
        self_attention_post_depth=args.self_attention_post_depth,
        self_attention_heads=args.self_attention_heads,
        state_space_depth=args.state_space_depth,
        state_space_post_depth=args.state_space_post_depth,
        state_space_state_dim=args.state_space_state_dim,
        state_space_expand=args.state_space_expand,
        state_space_bidirectional=args.state_space_bidirectional,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        codebook_sizes=codebook_sizes_arg,
        rvq_band_splits=rvq_band_splits_arg,
        turboquant=bool(args.turboquant),
        turboquant_seed=int(args.turboquant_seed),
        lambda_time=args.lambda_time,
        lambda_stft=args.lambda_stft,
        lambda_stft_grad=args.lambda_stft_grad,
        lambda_stft_cos=args.lambda_stft_cos,
        lambda_stft_excess=args.lambda_stft_excess,
        stft_excess_margin=args.stft_excess_margin,
        lambda_hf_under=args.lambda_hf_under,
        lambda_hf_sc=args.lambda_hf_sc,
        lambda_hf_complex=args.lambda_hf_complex,
        hf_min_hz=args.hf_min_hz,
        hf_gate_db=args.hf_gate_db,
        hf_under_margin=args.hf_under_margin,
        hf_peak_mask=args.hf_peak_mask,
        lambda_peak_contrast=args.lambda_peak_contrast,
        lambda_peak_mag=args.lambda_peak_mag,
        peak_contrast_radius=args.peak_contrast_radius,
        lambda_freq_ac=args.lambda_freq_ac,
        freq_ac_min_hz=args.freq_ac_min_hz,
        freq_ac_lags_hz=tuple(float(x) for x in freq_ac_lags_hz),
        lambda_stationary_line=args.lambda_stationary_line,
        stationary_line_min_hz=args.stationary_line_min_hz,
        stationary_line_radius=args.stationary_line_radius,
        stationary_line_margin=args.stationary_line_margin,
        lambda_harmonic_f0=args.lambda_harmonic_f0,
        lambda_harmonic_amp=args.lambda_harmonic_amp,
        harmonic_f0_frame=args.harmonic_f0_frame,
        harmonic_f0_hop=args.harmonic_f0_hop,
        harmonic_f0_lags=tuple(int(x) for x in harmonic_f0_lags),
        harmonic_voicing_threshold=args.harmonic_voicing_threshold,
        hf_start_step=args.hf_start_step,
        hf_ramp_steps=args.hf_ramp_steps,
        stft_grad_freq_weight=args.stft_grad_freq_weight,
        stft_grad_time_weight=args.stft_grad_time_weight,
        stft_ramp_steps=args.stft_ramp_steps,
        stft_ramp_start_frac=args.stft_ramp_start,
        stft_scales=stft_scales_resolved,
        stft_scale_weights=stft_scale_weights_resolved,
        spectral_batch_items=args.spectral_batch_items,
        spectral_loss_every=args.spectral_loss_every,
        stft_large_min_fft=args.stft_large_min_fft,
        stft_large_every=args.stft_large_every,
        stft_hf_emphasis=args.stft_hf_emphasis,
        mel_n_fft=args.mel_n_fft,
        mel_hop=args.mel_hop,
        n_mels=args.n_mels,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
        lambda_mel_l1=args.lambda_mel_l1,
        lambda_mel_l2=args.lambda_mel_l2,
        lambda_vq=args.lambda_vq,
        vq_loss=args.vq_loss,
        vq_huber_delta=args.vq_huber_delta,
        vq_loss_normalize=bool(args.vq_loss_normalize),
        vq_loss_norm_eps=args.vq_loss_norm_eps,
        lambda_entropy=args.lambda_entropy,
        lambda_marginal=args.lambda_marginal,
        marginal_tau=args.marginal_tau,
        marginal_boost_steps=args.marginal_boost_steps,
        marginal_boost_mult=args.marginal_boost_mult,
        vq_reset_every=args.vq_reset_every,
        vq_reset_collapse_frac=args.vq_reset_collapse_frac,
        vq_reset_noise=args.vq_reset_noise,
        vq_reset_shuffle=args.vq_reset_shuffle,
        vq_reset_kmeans=args.vq_reset_kmeans,
        vq_reset_full_refresh_max_unique=args.vq_reset_full_refresh_max_unique,
        vq_reset_log_every=args.vq_reset_log_every,
        vq_cosine=args.vq_cosine,
        vq_commitment=args.vq_beta,
        lambda_cos=args.lambda_cos,
        cos_hinge=args.cos_hinge,
        cos_target=args.cos_target,
        lambda_sisdr=args.lambda_sisdr,
        lambda_preemph=args.lambda_preemph,
        preemph_coef=args.preemph_coef,
        lambda_multidelta=args.lambda_multidelta,
        multidelta_lags=multidelta_lags,
        lambda_ae_anchor_time=args.lambda_ae_anchor_time,
        lambda_ae_anchor_cos=args.lambda_ae_anchor_cos,
        lambda_sc=args.lambda_sc,
        lambda_complex_stft=args.lambda_complex_stft,
        log_every=args.log_every,
        log_loss_ema_beta=args.log_loss_ema_beta,
        log_cos_ema_beta=args.log_cos_ema_beta,
        best_holdout_metric=best_holdout_metric,
        spectrogram_every=args.spectrogram_every,
        spectrogram_dir=args.spectrogram_dir,
        save_audio=args.save_audio,
        spectrogram_seconds=args.spectrogram_seconds,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        use_librispeech=bool(args.librispeech),
        lambda_mag_l1=args.lambda_mag_l1,
        activation=args.activation,
        rvq_code_dim=args.rvq_code_dim,
        vq_ema_decay=args.vq_ema_decay,
        vq_ema_every=args.vq_ema_every,
        decoder_upsample=args.decoder_upsample,
        decoder_transpose_kernel=args.decoder_transpose_kernel,
        decoder_band_split=bool(args.decoder_band_split),
        band_split_cutoffs_hz=(float(band_split_cutoffs[0]), float(band_split_cutoffs[1])),
        band_split_taps=args.band_split_taps,
        decoder_band_head_depth=args.decoder_band_head_depth,
        decoder_band_latent_split=bool(args.decoder_band_latent_split),
        decoder_harmonic_source=bool(args.decoder_harmonic_source),
        decoder_harmonic_partials=args.decoder_harmonic_partials,
        decoder_harmonic_fmin_hz=args.decoder_harmonic_fmin_hz,
        decoder_harmonic_fmax_hz=args.decoder_harmonic_fmax_hz,
        decoder_harmonic_gain=args.decoder_harmonic_gain,
        decoder_harmonic_amp_bias=args.decoder_harmonic_amp_bias,
        decoder_harmonic_rolloff=args.decoder_harmonic_rolloff,
        decoder_harmonic_mode=args.decoder_harmonic_mode,
        decoder_harmonic_control_smooth=args.decoder_harmonic_control_smooth,
        decoder_harmonic_env_bias=args.decoder_harmonic_env_bias,
        decoder_harmonic_band_weights=(
            float(harmonic_band_weights[0]),
            float(harmonic_band_weights[1]),
            float(harmonic_band_weights[2]),
        ),
        decoder_harmonic_residual_band_weights=(
            float(harmonic_residual_band_weights[0]),
            float(harmonic_residual_band_weights[1]),
            float(harmonic_residual_band_weights[2]),
        ),
        decoder_harmonic_groups=args.decoder_harmonic_groups,
        decoder_harmonic_group_bias=args.decoder_harmonic_group_bias,
        lambda_band_l1=args.lambda_band_l1,
        band_l1_weights=(float(band_l1_weights[0]), float(band_l1_weights[1]), float(band_l1_weights[2])),
        band_l1_floor=args.band_l1_floor,
        lambda_band_branch_l1=args.lambda_band_branch_l1,
        band_branch_weights=(float(band_branch_weights[0]), float(band_branch_weights[1]), float(band_branch_weights[2])),
        causal=bool(args.causal),
        use_bf16=bool(args.bf16),
        use_compile=bool(args.compile_loss),
        full_checkpoint=bool(args.full_checkpoint),
        eval_every=args.eval_every,
        eval_clips=args.eval_clips,
        eval_seconds=args.eval_seconds,
        eval_dir=Path(args.eval_dir) if args.eval_dir else None,
        eval_seed=args.eval_seed,
        log_mlx_tsv=args.log_mlx_tsv,
        results_tsv_path=args.results_tsv,
        adaptive_mode=args.adaptive_mode,
    )

    mx.random.seed(cfg.seed)
    model = MLXCodec(cfg)

    start_step = 0
    res_ck_path: Path | None = None
    res_ck_n: int | None = None
    resume_flat: dict[str, mx.array] | None = None
    resume_data_off: int | None = None
    resume_meta: dict | None = None
    if args.resume is not None:
        if args.resume == "__latest__":
            ck_dir = Path(args.checkpoint_dir).expanduser().resolve()
            ck_path = _find_latest_checkpoint(ck_dir)
            if ck_path is None:
                print(
                    f"No codec_step*.npz or ckpt_step*.safetensors in {ck_dir} (use --resume PATH or train first)",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            ck_path = Path(args.resume).expanduser().resolve()
            if not ck_path.is_file():
                print(f"--resume file not found: {ck_path}", file=sys.stderr)
                sys.exit(1)
        if _is_full_checkpoint_path(ck_path):
            try:
                resume_flat = mx.load(str(ck_path))
            except Exception as e:
                print(f"load safetensors failed: {e}", file=sys.stderr)
                sys.exit(1)
            meta_path = ck_path.with_suffix(".meta.json")
            if meta_path.is_file():
                resume_meta = json.loads(meta_path.read_text())
                res_ck_n = int(resume_meta.get("step", _parse_full_checkpoint_step(ck_path) or 0))
                resume_data_off = int(resume_meta.get("data_off", 0))
            else:
                res_ck_n = _parse_full_checkpoint_step(ck_path)
                if res_ck_n is None:
                    print(f"Bad checkpoint filename: {ck_path.name}", file=sys.stderr)
                    sys.exit(1)
                resume_data_off = 0
            res_ck_path = ck_path
            start_step = int(res_ck_n) + 1
        else:
            n_ck = _parse_checkpoint_step(ck_path)
            if n_ck is None:
                print(
                    f"Filename must be codec_stepN.npz or ckpt_stepN.safetensors, got: {ck_path.name}",
                    file=sys.stderr,
                )
                sys.exit(1)
            try:
                model.load_weights(str(ck_path), strict=True)
            except Exception as e:
                flat_npz = dict(mx.load(str(ck_path)))
                flat2, n_rem = _remap_legacy_codec_subdict(flat_npz, model)
                if n_rem > 0:
                    print(
                        f"[resume] remapped {n_rem} legacy ``encoder.layers``/``decoder.layers`` keys → ``_b``/``_d``",
                        flush=True,
                    )
                try:
                    model.load_weights(list(flat2.items()), strict=False)
                except Exception as e2:
                    print(f"load_weights failed: {e}", file=sys.stderr)
                    print(f"  after legacy key remap + strict=False: {e2}", file=sys.stderr)
                    print(
                        "  Hint: checkpoint may be from ``decoder_upsample=transpose`` while current "
                        "``Config`` defaults to ``repeat_conv`` (or vice versa). Weights are not interchangeable.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            res_ck_path, res_ck_n = ck_path, n_ck
            start_step = n_ck + 1
        if start_step >= cfg.steps:
            print(
                f"Checkpoint ends at step {res_ck_n}, which is already at or past --steps {cfg.steps}",
                file=sys.stderr,
            )
            sys.exit(0)
        print(
            f"[resume] {ck_path}  (after step {res_ck_n})  → train steps {start_step}…{cfg.steps - 1}",
            flush=True,
        )

    plateau_resume = resume_meta.get("lr_plateau") if resume_meta else None
    lr_spec = build_lr_schedule(cfg, plateau_resume=None)
    plateau_schedule = lr_spec if isinstance(lr_spec, PlateauLRSchedule) else None
    opt = Adam(lr_spec)
    opt.init(model.parameters())
    if start_step > 0:
        opt.state["step"] = mx.array(int(start_step), dtype=mx.uint64)
        _refresh_optimizer_schedules(opt)

    if resume_flat is not None:
        load_full_checkpoint_into(resume_flat, model=model, opt=opt)
        mx.eval(model.parameters(), opt.state)
        print("[resume] loaded full safetensors (weights + optimizer state)", flush=True)
        if plateau_schedule is not None and plateau_resume is not None:
            plateau_schedule.load_state(plateau_resume)
            _refresh_optimizer_schedules(opt)

    compiled_multi_stft: Callable[[mx.array, mx.array], mx.array] | None = None
    if cfg.use_compile and cfg.lambda_stft_grad <= 0 and cfg.lambda_stft_cos <= 0:

        @mx.compile
        def _cmsft(pred: mx.array, tgt: mx.array) -> mx.array:
            return multi_stft_loss(
                pred,
                tgt,
                cfg.stft_scales,
                hf_emphasis=cfg.stft_hf_emphasis,
                scale_weights=cfg.stft_scale_weights,
            )

        compiled_multi_stft = _cmsft

    if cfg.use_bf16:
        _cast_module_floats(model, mx.bfloat16)
        mx.eval(model.parameters())

    n_params = _tree_n_params(model.parameters())
    st = encoder_time_stride(cfg)
    nom_kbps = nominal_rvq_kbps(cfg) * adaptive_mod.nominal_bitrate_multiplier(cfg.adaptive_mode)
    cb_sizes = effective_codebook_sizes(cfg)
    rvq_ks = "×".join(str(k) for k in cb_sizes)
    ramp_s = (
        f"λ_stft_ramp={cfg.stft_ramp_steps}@{cfg.stft_ramp_start_frac}→1  "
        if cfg.stft_ramp_steps > 0
        else "λ_stft_ramp=off  "
    )
    stft_nfo = "  ".join(f"nfft{n}/hop{h}" for n, h in cfg.stft_scales)
    stft_w_info = ""
    if cfg.stft_scale_weights is not None:
        stft_w_info = "  stft_w=" + ",".join(f"{w:g}" for w in cfg.stft_scale_weights)
    spec_items = "all" if cfg.spectral_batch_items <= 0 else f"{cfg.spectral_batch_items}/micro"
    spec_every = max(1, int(cfg.spectral_loss_every))
    spec_every_info = "every step" if spec_every <= 1 else f"every {spec_every} steps ×{spec_every:g}"
    if cfg.stft_large_min_fft <= 0:
        large_info = "off"
    elif cfg.stft_large_every <= 1:
        large_info = f"nfft>={cfg.stft_large_min_fft} every step"
    else:
        large_info = f"nfft>={cfg.stft_large_min_fft} every {cfg.stft_large_every} steps"
    _ls = (cfg.lr_schedule or "").lower()
    if _ls in ("none", "constant", "off", ""):
        lr_info = f"lr={cfg.lr} (const)"
    elif _ls == "plateau":
        lr_info = (
            f"lr plateau: peak={cfg.lr} min={cfg.lr * cfg.lr_min_ratio:g}  "
            f"factor={cfg.lr_plateau_factor} patience={cfg.lr_plateau_patience}  "
            f"thr={cfg.lr_plateau_threshold} ema={cfg.lr_plateau_ema} cd={cfg.lr_plateau_cooldown}  "
            f"warmup={cfg.lr_warmup_steps}"
        )
    else:
        lr_info = (
            f"lr {cfg.lr_schedule}: peak={cfg.lr} → min={cfg.lr * cfg.lr_min_ratio:g}  "
            f"warmup={cfg.lr_warmup_steps}"
        )
    mel_fb_mx: mx.array | None = None
    mel_info = ""
    if cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0:
        fmax_mel = float(cfg.mel_fmax) if cfg.mel_fmax is not None else float(cfg.sample_rate) / 2.0
        if fmax_mel <= cfg.mel_fmin:
            print(
                f"mel: mel-fmax ({fmax_mel}) must be > mel-fmin ({cfg.mel_fmin})",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            mel_fb_mx = mel_filterbank_mx(
                cfg.sample_rate,
                cfg.mel_n_fft,
                cfg.n_mels,
                cfg.mel_fmin,
                fmax_mel,
            )
        except ValueError as e:
            print(f"mel filterbank: {e}", file=sys.stderr)
            sys.exit(1)
        mel_info = (
            f"  mel=nfft{cfg.mel_n_fft}/hop{cfg.mel_hop} M={cfg.n_mels} "
            f"[{cfg.mel_fmin:g}-{fmax_mel:g}]Hz  λ_mel_L1={cfg.lambda_mel_l1}  λ_mel_L2={cfg.lambda_mel_l2}"
        )
    lt_pre_post = (
        f"lat_tmp={cfg.latent_temporal_depth}/{cfg.latent_temporal_post_depth}  "
        f"self_attn={cfg.self_attention_depth}/{cfg.self_attention_post_depth}×h{cfg.self_attention_heads}  "
        f"ssm={cfg.state_space_depth}/{cfg.state_space_post_depth}×S{cfg.state_space_state_dim}"
        f"/e{cfg.state_space_expand}/{'bi' if cfg.state_space_bidirectional else 'uni'}"
    )
    sc_info = f"  λ_sc={cfg.lambda_sc:g}" if cfg.lambda_sc > 0 else ""
    cx_info = f"  λ_cx={cfg.lambda_complex_stft:g}" if cfg.lambda_complex_stft > 0 else ""
    excess_info = f"  λ_excess={cfg.lambda_stft_excess:g}@{cfg.stft_excess_margin:g}" if cfg.lambda_stft_excess > 0 else ""
    peak_info = (
        f"  λ_peak={cfg.lambda_peak_contrast:g}@r{cfg.peak_contrast_radius}"
        if cfg.lambda_peak_contrast > 0
        else ""
    )
    peak_mag_info = (
        f"  λ_pmag={cfg.lambda_peak_mag:g}@r{cfg.peak_contrast_radius}"
        if cfg.lambda_peak_mag > 0
        else ""
    )
    freq_ac_info = (
        f"  λ_fac={cfg.lambda_freq_ac:g}@≥{cfg.freq_ac_min_hz:g}Hz"
        if cfg.lambda_freq_ac > 0
        else ""
    )
    stationary_info = (
        f"  λ_stat={cfg.lambda_stationary_line:g}@≥{cfg.stationary_line_min_hz:g}Hz/r{cfg.stationary_line_radius}/m{cfg.stationary_line_margin:g}"
        if cfg.lambda_stationary_line > 0
        else ""
    )
    harm_loss_info = (
        f"  λ_hf0={cfg.lambda_harmonic_f0:g}  λ_hamp={cfg.lambda_harmonic_amp:g}"
        f" acf={cfg.harmonic_f0_frame}/{cfg.harmonic_f0_hop}@thr{cfg.harmonic_voicing_threshold:g}"
        if cfg.lambda_harmonic_f0 > 0 or cfg.lambda_harmonic_amp > 0
        else ""
    )
    hf_info = ""
    if cfg.lambda_hf_under > 0 or cfg.lambda_hf_sc > 0 or cfg.lambda_hf_complex > 0:
        hf_info = (
            f"  λ_hf_under={cfg.lambda_hf_under:g}  λ_hf_sc={cfg.lambda_hf_sc:g}  λ_hf_cx={cfg.lambda_hf_complex:g}  "
            f"hf≥{cfg.hf_min_hz:g}Hz gate={cfg.hf_gate_db:g}dB margin={cfg.hf_under_margin:g}"
            f"{' peak_mask' if cfg.hf_peak_mask else ''}"
            f" start={cfg.hf_start_step} ramp={cfg.hf_ramp_steps}"
        )
    wave_info = ""
    if (
        cfg.lambda_sisdr > 0
        or cfg.lambda_preemph > 0
        or cfg.lambda_multidelta > 0
        or cfg.lambda_ae_anchor_time > 0
        or cfg.lambda_ae_anchor_cos > 0
    ):
        wave_info = (
            f"  λ_sisdr={cfg.lambda_sisdr:g}  λ_pre={cfg.lambda_preemph:g}@{cfg.preemph_coef:g}  "
            f"λ_md={cfg.lambda_multidelta:g}@{','.join(str(x) for x in cfg.multidelta_lags)}  "
            f"λ_ae_anchor={cfg.lambda_ae_anchor_time:g}/{cfg.lambda_ae_anchor_cos:g}"
        )
    band_info = ""
    if cfg.decoder_band_split or cfg.lambda_band_l1 > 0 or cfg.lambda_band_branch_l1 > 0:
        band_info = (
            f"  band_split={cfg.decoder_band_split}@{cfg.band_split_cutoffs_hz[0]:g},{cfg.band_split_cutoffs_hz[1]:g}Hz/"
            f"{cfg.band_split_taps}t/head{cfg.decoder_band_head_depth}/lat{int(cfg.decoder_band_latent_split)}  λ_band_l1={cfg.lambda_band_l1:g}  "
            f"band_w={','.join(f'{w:g}' for w in cfg.band_l1_weights)}  "
            f"λ_band_br={cfg.lambda_band_branch_l1:g}  br_w={','.join(f'{w:g}' for w in cfg.band_branch_weights)} "
            f"floor={cfg.band_l1_floor:g}"
        )
    harm_info = ""
    if cfg.decoder_harmonic_source:
        group_s = ""
        if cfg.decoder_harmonic_groups > 0:
            group_s = f"/G{cfg.decoder_harmonic_groups}@bias{cfg.decoder_harmonic_group_bias:g}"
        harm_info = (
            f"  harm_src=P{cfg.decoder_harmonic_partials}@{cfg.decoder_harmonic_fmin_hz:g}-"
            f"{cfg.decoder_harmonic_fmax_hz:g}Hz×{cfg.decoder_harmonic_gain:g}/bias{cfg.decoder_harmonic_amp_bias:g}"
            f"/roll{cfg.decoder_harmonic_rolloff:g}/mode={cfg.decoder_harmonic_mode}"
            f"/smooth{cfg.decoder_harmonic_control_smooth}"
            f"/band{','.join(f'{w:g}' for w in cfg.decoder_harmonic_band_weights)}"
            f"/res{','.join(f'{w:g}' for w in cfg.decoder_harmonic_residual_band_weights)}"
            f"/env_bias{cfg.decoder_harmonic_env_bias:g}{group_s}"
        )
    _acm_p = max(1, int(cfg.grad_accum_steps))
    batch_info = (
        f"  batch={cfg.batch}×accum={_acm_p}→{cfg.batch * _acm_p}"
        if _acm_p > 1
        else f"  batch={cfg.batch}"
    )
    rvq_group_info = ""
    if cfg.rvq_band_splits:
        rvq_group_info = f"/groups{','.join(str(int(x)) for x in cfg.rvq_band_splits)}"
    vq_loss_info = f"{cfg.vq_loss}"
    if str(cfg.vq_loss).lower() == "huber":
        vq_loss_info = f"{vq_loss_info}@δ{cfg.vq_huber_delta:g}"
    if cfg.vq_loss_normalize:
        vq_loss_info = f"{vq_loss_info}/rms"
    print(
        f"Parameters: {n_params / 1e6:.2f}M{batch_info}  latent={cfg.latent_dim}  {lt_pre_post}  pre_vq_ln={cfg.pre_vq_layernorm}  enc_stride={st}×  "
        f"s1/scale={cfg.stride1_blocks_per_scale}  dec_refine/scale={cfg.decoder_refine_blocks_per_scale}  "
        f"dec_up={cfg.decoder_upsample}/k{cfg.decoder_transpose_kernel}{band_info}{harm_info}  "
        f"grad_clip={cfg.grad_clip_norm}  {lr_info}  "
        f"~{nom_kbps:.1f} kbps nominal  "
        f"STFT×{len(cfg.stft_scales)}: {stft_nfo}{stft_w_info}  stft_hf={cfg.stft_hf_emphasis:g}  "
        f"spectral_items={spec_items}  spectral_loss={spec_every_info}  large={large_info}  "
        f"λ_L1={cfg.lambda_time}  λ_stft={cfg.lambda_stft}  λ_stft_grad={cfg.lambda_stft_grad}{sc_info}{cx_info}{excess_info}{peak_info}{peak_mag_info}{freq_ac_info}{stationary_info}{harm_loss_info}{hf_info}  "
        f"sgrad_df/dt={cfg.stft_grad_freq_weight}/{cfg.stft_grad_time_weight}  λ_stft_cos={cfg.lambda_stft_cos}  {ramp_s}"
        f"RVQ={cfg.n_codebooks}×K={rvq_ks}{rvq_group_info}  vq_cos={cfg.vq_cosine}  turboquant={cfg.turboquant}  λ_vq={cfg.lambda_vq}  λ_ent={cfg.lambda_entropy}  "
        f"λ_marg={cfg.lambda_marginal}  τ_marg={cfg.marginal_tau}  "
        f"{'λ_marg_boost=' + str(cfg.marginal_boost_steps) + '@×' + str(cfg.marginal_boost_mult) + '→1  ' if cfg.marginal_boost_steps > 0 else ''}"
        f"vq_reset={cfg.vq_reset_every}@{cfg.vq_reset_collapse_frac}  "
        f"σ={cfg.vq_reset_noise}  shuffle={cfg.vq_reset_shuffle}  vq_km={cfg.vq_reset_kmeans}  "
        f"vq_full≤{cfg.vq_reset_full_refresh_max_unique}u→allK  vq_rst_log={cfg.vq_reset_log_every}  "
        f"vq_ema={cfg.vq_ema_decay:g}/{cfg.vq_ema_every}  vq_loss={vq_loss_info}  "
        f"λ_cos={cfg.lambda_cos}  cos_hinge={cfg.cos_hinge}@{cfg.cos_target}{wave_info}"
        f"{mel_info}"
    )
    if cfg.causal:
        tms = streaming_mod.benchmark_causal_latency_ms(sample_rate=cfg.sample_rate, chunk_ms=20.0)
        print(f"  [causal] benchmark CausalConv1d ~{tms:.2f} ms / 20 ms chunk (wall, indicative)", flush=True)

    audio_paths: list[Path] | None = None
    if cfg.data_dir is not None:
        if args.librispeech and not cfg.data_dir.is_dir():
            print(
                f"Dataset dir missing: {cfg.data_dir}\n"
                "  Expected e.g. data/cv-corpus/pl/clips/*.mp3",
                file=sys.stderr,
            )
            sys.exit(1)
        audio_paths = _collect_audio_paths(cfg.data_dir)
        if not audio_paths:
            print(
                f"No audio files (.wav/.flac/.ogg/.mp3) under {cfg.data_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        tag = " (default corpus)" if args.librispeech else ""
        io_bits: list[str] = []
        if cfg.load_audio_threads > 1:
            io_bits.append(f"io_threads={cfg.load_audio_threads}")
        if cfg.prefetch_audio and max(1, int(cfg.grad_accum_steps)) <= 1:
            io_bits.append("prefetch=on")
        io_extra = f" ({' '.join(io_bits)})" if io_bits else ""
        print(f"Using {len(audio_paths)} audio files from {cfg.data_dir}{tag}{io_extra}")
    else:
        print("Using synthetic sine/noise batches")

    eval_paths: list[Path] | None = None
    if cfg.eval_dir is not None:
        eval_root = Path(cfg.eval_dir).expanduser().resolve()
        if not eval_root.is_dir():
            print(f"Holdout eval dir missing: {eval_root}", file=sys.stderr)
            sys.exit(1)
        all_eval_paths = _collect_audio_paths(eval_root)
        if not all_eval_paths:
            print(f"No audio files (.wav/.flac/.ogg/.mp3) under holdout eval dir {eval_root}", file=sys.stderr)
            sys.exit(1)
        eval_paths = _select_eval_audio_paths(all_eval_paths, cfg)
        seconds_s = "full files" if float(cfg.eval_seconds) <= 0 else f"{float(cfg.eval_seconds):g}s clips"
        print(
            f"Using {len(eval_paths)}/{len(all_eval_paths)} holdout eval files from {eval_root} "
            f"({seconds_s}, seed={cfg.eval_seed})"
        )

    t0 = time.time()
    acm = max(1, int(cfg.grad_accum_steps))
    if audio_paths is not None:
        data_off = (
            int(resume_data_off)
            if resume_data_off is not None
            else int(start_step) * cfg.batch * acm
        )
    else:
        data_off = 0
    ema_loss_v: float | None = None
    ema_vq_v: float | None = None
    ema_cos_pct: float | None = None
    best_holdout_metric = _canonical_eval_metric_name(cfg.best_holdout_metric)
    best_holdout_value: float | None = None

    prof_on = bool(getattr(args, "profile", False))
    prof_acc: dict[str, float] = {}
    prof_count = 0

    def _psync(*arrs):
        if prof_on and arrs:
            mx.eval(*arrs)

    def _ptic() -> float:
        return time.perf_counter() if prof_on else 0.0

    def _ptoc(tag: str, t: float) -> None:
        if prof_on:
            prof_acc[tag] = prof_acc.get(tag, 0.0) + (time.perf_counter() - t) * 1000.0

    prefetch_ex: ThreadPoolExecutor | None = None
    prefetch_future = None
    use_prefetch = (
        audio_paths is not None
        and cfg.prefetch_audio
        and cfg.steps > start_step + 1
        and acm <= 1
    )
    if use_prefetch:
        prefetch_ex = ThreadPoolExecutor(max_workers=1)
        prefetch_future = prefetch_ex.submit(_load_audio_batch, cfg, audio_paths, data_off)
        data_off += cfg.batch

    for step in range(start_step, cfg.steps):
        _t_data = _ptic()
        if audio_paths is not None:
            if prefetch_future is not None:
                batch0 = prefetch_future.result()
                if step < cfg.steps - 1:
                    prefetch_future = prefetch_ex.submit(
                        _load_audio_batch, cfg, audio_paths, data_off
                    )
                    data_off += cfg.batch
                else:
                    prefetch_future = None
            else:
                batch0 = _load_audio_batch(cfg, audio_paths, data_off)
                data_off += cfg.batch
        else:
            batch0 = synth_batch(cfg, key=step * 1_000_003 + cfg.seed * 10007)
        _psync(batch0)
        _ptoc("data", _t_data)

        _t_g = _ptic()
        inv_acm = 1.0 / float(acm)
        grads_acc = None
        loss_fn = None
        loss_acc = mx.array(0.0, dtype=mx.float32)
        skip_outer = False
        batch = batch0
        for micro in range(acm):
            b = (
                batch0
                if micro == 0
                else (
                    _load_audio_batch(cfg, audio_paths, data_off)
                    if audio_paths is not None
                    else synth_batch(cfg, key=step * 1_000_003 + micro * 97 + cfg.seed * 10007)
                )
            )
            if micro > 0 and audio_paths is not None:
                data_off += cfg.batch
            batch = b
            loss_fn = make_train_fn(
                model,
                cfg,
                b,
                step,
                mel_fb_mx,
                compiled_multi_stft=compiled_multi_stft,
            )
            lg = nn.value_and_grad(model, loss_fn)
            loss, grads = lg(model)
            # Lazy: sum losses and grad trees without mx.eval per micro-batch.
            # NaN detection is deferred to the single post-accum sync below.
            loss_acc = loss_acc + loss
            g_scaled = _grad_tree_scale(grads, inv_acm)
            if grads_acc is None:
                grads_acc = g_scaled
            else:
                grads_acc = _grad_tree_add(grads_acc, g_scaled)

        if grads_acc is None:
            print(f"  [skip] empty accum at step {step}", flush=True)
            continue
        # Single sync at end of accum: evaluates loss + accumulated grads + finite check together.
        lv_m_total, bad_grad = _eval_loss_grad_and_nonfinite(loss_acc, grads_acc)
        if not math.isfinite(lv_m_total):
            print(f"  [skip] non-finite summed loss at step {step}", flush=True)
            continue
        if bad_grad:
            print(f"  [skip] non-finite accumulated gradient at step {step}", flush=True)
            continue
        _ptoc("G_fwdbwd", _t_g)

        _t_misc = _ptic()
        lv0 = lv_m_total * inv_acm
        grads = grads_acc
        if cfg.grad_clip_norm > 0:
            grads = clip_gradients_global_norm(grads, cfg.grad_clip_norm)
        _ptoc("clip", _t_misc)
        # Log before optimizer step (``forward_metrics`` from last micro-batch).
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            lv = lv0
            elapsed = time.time() - t0
            fm = loss_fn.forward_metrics
            lt, ls = fm["lt"], fm["ls"]
            lsg = fm["lsg"]
            lsc = fm["lsc"]
            lm1, lm2 = fm["lm1"], fm["lm2"]
            vq_l, ent_pos, marg_ent, cos_m = fm["vq_l"], fm["ent_pos"], fm["marg_ent"], fm["cos_m"]
            idx = fm["idx"]
            l_lin = fm.get("l_lin")
            l_sc = fm.get("l_sc")
            l_cx = fm.get("l_cx")
            l_excess = fm.get("l_excess")
            l_peak = fm.get("l_peak")
            l_peak_mag = fm.get("l_peak_mag")
            l_freq_ac = fm.get("l_freq_ac")
            l_stationary = fm.get("l_stationary")
            l_harm_f0 = fm.get("l_harm_f0")
            l_harm_amp = fm.get("l_harm_amp")
            harm_voice = fm.get("harm_voice")
            l_hf_under = fm.get("l_hf_under")
            l_hf_sc = fm.get("l_hf_sc")
            l_hf_complex = fm.get("l_hf_complex")
            hf_mult = fm.get("hf_mult")
            l_sisdr = fm.get("l_sisdr")
            l_preemph = fm.get("l_preemph")
            l_multidelta = fm.get("l_multidelta")
            l_band = fm.get("l_band")
            l_band_branch = fm.get("l_band_branch")
            ae_anchor = fm.get("ae_anchor")
            spec_mult_m = fm.get("spec_mult")
            to_ev = [lt, ls, lsg, lsc, lm1, lm2, vq_l, ent_pos, marg_ent, cos_m]
            if isinstance(l_lin, mx.array):
                to_ev.append(l_lin)
            if isinstance(l_sc, mx.array):
                to_ev.append(l_sc)
            if isinstance(l_cx, mx.array):
                to_ev.append(l_cx)
            if isinstance(l_excess, mx.array):
                to_ev.append(l_excess)
            if isinstance(l_peak, mx.array):
                to_ev.append(l_peak)
            if isinstance(l_peak_mag, mx.array):
                to_ev.append(l_peak_mag)
            if isinstance(l_freq_ac, mx.array):
                to_ev.append(l_freq_ac)
            if isinstance(l_stationary, mx.array):
                to_ev.append(l_stationary)
            if isinstance(l_harm_f0, mx.array):
                to_ev.append(l_harm_f0)
            if isinstance(l_harm_amp, mx.array):
                to_ev.append(l_harm_amp)
            if isinstance(harm_voice, mx.array):
                to_ev.append(harm_voice)
            if isinstance(l_hf_under, mx.array):
                to_ev.append(l_hf_under)
            if isinstance(l_hf_sc, mx.array):
                to_ev.append(l_hf_sc)
            if isinstance(l_hf_complex, mx.array):
                to_ev.append(l_hf_complex)
            if isinstance(hf_mult, mx.array):
                to_ev.append(hf_mult)
            if isinstance(l_sisdr, mx.array):
                to_ev.append(l_sisdr)
            if isinstance(l_preemph, mx.array):
                to_ev.append(l_preemph)
            if isinstance(l_multidelta, mx.array):
                to_ev.append(l_multidelta)
            if isinstance(l_band, mx.array):
                to_ev.append(l_band)
            if isinstance(l_band_branch, mx.array):
                to_ev.append(l_band_branch)
            if isinstance(ae_anchor, mx.array):
                to_ev.append(ae_anchor)
            if isinstance(spec_mult_m, mx.array):
                to_ev.append(spec_mult_m)
            mx.eval(*to_ev)
            util = _format_vq_util(idx, cb_sizes)
            cos_pct = 100.0 * float(cos_m.item())
            beta_ema = float(cfg.log_cos_ema_beta)
            if beta_ema > 0.0:
                ema_cos_pct = cos_pct if ema_cos_pct is None else beta_ema * ema_cos_pct + (1.0 - beta_ema) * cos_pct
                cos_ema_extra = f"  cos_ema={ema_cos_pct:.1f}%"
            else:
                cos_ema_extra = ""
            vq_v = float(vq_l.item())
            vq_s = f"{vq_v:.4f}" if vq_v >= 1e-3 else f"{vq_v:.4e}"
            loss_ema_extra = ""
            loss_beta = float(cfg.log_loss_ema_beta)
            if loss_beta > 0.0:
                loss_v = float(lv)
                ema_loss_v = loss_v if ema_loss_v is None else loss_beta * ema_loss_v + (1.0 - loss_beta) * loss_v
                ema_vq_v = vq_v if ema_vq_v is None else loss_beta * ema_vq_v + (1.0 - loss_beta) * vq_v
                loss_ema_extra = f"  loss_ema={ema_loss_v:.4f}  vq_ema={ema_vq_v:.4f}"
            ent_extra = ""
            if cfg.lambda_entropy > 0:
                ent_extra = f"  ent_pos={float(ent_pos.item()):.4f}"
            spec_mult_v = float(spec_mult_m.item()) if isinstance(spec_mult_m, mx.array) else 1.0
            w_stft = effective_lambda_stft(step, cfg) * spec_mult_v
            w_marg = effective_lambda_marginal(step, cfg)
            ramp_extra = f"  λ_stft_eff={w_stft:.4f}" if cfg.stft_ramp_steps > 0 else ""
            if cfg.spectral_loss_every > 1:
                ramp_extra = f"{ramp_extra}  spec×={spec_mult_v:g}/{cfg.spectral_loss_every}"
            if cfg.lambda_marginal > 0 and cfg.marginal_boost_steps > 0:
                ramp_extra = f"{ramp_extra}  λ_m_eff={w_marg:.4f}"
            sgrad_extra = ""
            if cfg.lambda_stft_grad > 0:
                sgrad_extra = f"  sgrad={float(lsg.item()):.4f}"
            scos_extra = ""
            if cfg.lambda_stft_cos > 0:
                scos_extra = f"  stft_cos={float(lsc.item()):.4f}"
            mel_extra = ""
            if cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0:
                mel_extra = f"  mel_l1={float(lm1.item()):.4f}  mel_l2={float(lm2.item()):.4f}"
            lin_extra = ""
            if cfg.lambda_mag_l1 > 0 and isinstance(l_lin, mx.array):
                lin_extra = f"  lin_mag={float(l_lin.item()):.4f}"
            sc_extra = ""
            if cfg.lambda_sc > 0 and isinstance(l_sc, mx.array):
                sc_extra = f"  sc={float(l_sc.item()):.4f}"
            cx_extra = ""
            if cfg.lambda_complex_stft > 0 and isinstance(l_cx, mx.array):
                cx_extra = f"  cx={float(l_cx.item()):.4f}"
            excess_extra = ""
            if cfg.lambda_stft_excess > 0 and isinstance(l_excess, mx.array):
                excess_extra = f"  excess={float(l_excess.item()):.4f}"
            peak_extra = ""
            if cfg.lambda_peak_contrast > 0 and isinstance(l_peak, mx.array):
                peak_extra = f"  peak={float(l_peak.item()):.4f}"
            if cfg.lambda_peak_mag > 0 and isinstance(l_peak_mag, mx.array):
                peak_extra = f"{peak_extra}  pmag={float(l_peak_mag.item()):.4f}"
            if cfg.lambda_freq_ac > 0 and isinstance(l_freq_ac, mx.array):
                peak_extra = f"{peak_extra}  fac={float(l_freq_ac.item()):.4f}"
            if cfg.lambda_stationary_line > 0 and isinstance(l_stationary, mx.array):
                peak_extra = f"{peak_extra}  stat={float(l_stationary.item()):.4f}"
            if cfg.lambda_harmonic_f0 > 0 and isinstance(l_harm_f0, mx.array):
                peak_extra = f"{peak_extra}  hf0={float(l_harm_f0.item()):.4f}"
            if cfg.lambda_harmonic_amp > 0 and isinstance(l_harm_amp, mx.array):
                peak_extra = f"{peak_extra}  hamp={float(l_harm_amp.item()):.4f}"
            if (cfg.lambda_harmonic_f0 > 0 or cfg.lambda_harmonic_amp > 0) and isinstance(harm_voice, mx.array):
                peak_extra = f"{peak_extra}  voice={100.0 * float(harm_voice.item()):.1f}%"
            hf_extra = ""
            if (cfg.lambda_hf_under > 0 or cfg.lambda_hf_sc > 0) and isinstance(hf_mult, mx.array):
                hf_extra = f"{hf_extra}  λ_hf_eff={float(hf_mult.item()):.3f}"
            if cfg.lambda_hf_under > 0 and isinstance(l_hf_under, mx.array):
                hf_extra = f"{hf_extra}  hf_under={float(l_hf_under.item()):.4f}"
            if cfg.lambda_hf_sc > 0 and isinstance(l_hf_sc, mx.array):
                hf_extra = f"{hf_extra}  hf_sc={float(l_hf_sc.item()):.4f}"
            if cfg.lambda_hf_complex > 0 and isinstance(l_hf_complex, mx.array):
                hf_extra = f"{hf_extra}  hf_cx={float(l_hf_complex.item()):.4f}"
            wave_extra = ""
            if cfg.lambda_sisdr > 0 and isinstance(l_sisdr, mx.array):
                wave_extra = f"{wave_extra}  sisdr_loss={float(l_sisdr.item()):.4f}"
            if cfg.lambda_preemph > 0 and isinstance(l_preemph, mx.array):
                wave_extra = f"{wave_extra}  pre={float(l_preemph.item()):.4f}"
            if cfg.lambda_multidelta > 0 and isinstance(l_multidelta, mx.array):
                wave_extra = f"{wave_extra}  md={float(l_multidelta.item()):.4f}"
            if cfg.lambda_band_l1 > 0 and isinstance(l_band, mx.array):
                wave_extra = f"{wave_extra}  band_l1={float(l_band.item()):.4f}"
            if cfg.lambda_band_branch_l1 > 0 and isinstance(l_band_branch, mx.array):
                wave_extra = f"{wave_extra}  band_br={float(l_band_branch.item()):.4f}"
            if (cfg.lambda_ae_anchor_time > 0 or cfg.lambda_ae_anchor_cos > 0) and isinstance(ae_anchor, mx.array):
                wave_extra = f"{wave_extra}  ae_anchor={float(ae_anchor.item()):.4f}"
            if callable(lr_spec):
                tlr = lr_spec(opt.step)
                mx.eval(tlr)
                lr_log = float(tlr.item())
            else:
                lr_log = float(lr_spec)
            prof_extra = ""
            if prof_on and prof_acc:
                n = max(1, prof_count)
                order = ["data", "G_fwdbwd", "clip", "optim", "vq_ema", "vq_reset"]
                parts = [f"{k}={prof_acc[k] / n:.0f}" for k in order if k in prof_acc]
                prof_extra = f"  prof[ms]: {' '.join(parts)}"
                prof_acc.clear()
                prof_count = 0
            print(
                f"step {step:6d}/{cfg.steps}  loss={lv:.5f}  "
                f"L1={float(lt.item()):.4f}  stft={float(ls.item()):.4f}{sgrad_extra}{scos_extra}{sc_extra}{cx_extra}{excess_extra}{peak_extra}{hf_extra}{mel_extra}{lin_extra}{wave_extra}  "
                f"vq={vq_s}{loss_ema_extra}  marg_ent={float(marg_ent.item()):.4f}{ent_extra}{ramp_extra}  "
                f"cos={cos_pct:.1f}%{cos_ema_extra}{util}  lr={lr_log:.2e}  "
                f"{elapsed / (step - start_step + 1) * 1000:.1f} ms/step{prof_extra}",
                flush=True,
            )
        _t_opt = _ptic()
        opt.update(model, grads)
        # MLX is lazy: materialize the parameter + optimizer-state update here so the next
        # forward does not inherit an ever-growing graph from prior steps.
        mx.eval(model.parameters(), opt.state)
        _ptoc("optim", _t_opt)
        plateau_active = cfg.spectral_loss_every <= 1 or spectral_loss_step_multiplier(step, cfg) > 0.0
        if plateau_schedule is not None and step >= cfg.lr_warmup_steps and plateau_active:
            plateau_schedule.observe(lv0)

        if (
            0.0 < float(cfg.vq_ema_decay) < 1.0
            and not cfg.ae_only
            and step % int(cfg.vq_ema_every) == 0
        ):
            _t_vqe = _ptic()
            update_vq_ema_codebooks(model, batch0, cfg)
            mx.eval(model.parameters())
            _ptoc("vq_ema", _t_vqe)

        if (
            cfg.vq_reset_every > 0
            and not cfg.ae_only
            and step > 0
            and step % cfg.vq_reset_every == 0
        ):
            _t_rst = _ptic()
            n_reset, per_st = vq_reset_dead_codes(model, batch, cfg)
            mx.eval(model.parameters())
            _ptoc("vq_reset", _t_rst)
            if n_reset > 0:
                le = cfg.vq_reset_log_every
                if le <= 0 or step % le == 0:
                    parts = [f"s{i}={c}" for i, c in enumerate(per_st) if c > 0]
                    extra = f" ({', '.join(parts)})" if parts else ""
                    print(f"  [vq-reset] replaced {n_reset} dead codes{extra}", flush=True)

        if cfg.eval_every > 0 and step > 0 and step % int(cfg.eval_every) == 0:
            try:
                import numpy as np

                if eval_paths is not None:
                    summary = _holdout_eval_summary(model, cfg, eval_paths)
                    print(_format_eval_summary("eval-holdout", summary), flush=True)
                    _append_eval_tsv(cfg.log_mlx_tsv, step, "holdout", summary)
                    metric_v = summary.get(best_holdout_metric)
                    if metric_v is not None:
                        mv = float(metric_v)
                        if _is_better_eval_value(mv, best_holdout_value, best_holdout_metric):
                            prev_v = best_holdout_value
                            best_holdout_value = mv
                            _write_best_holdout_marker(
                                cfg,
                                step=step,
                                metric=best_holdout_metric,
                                value=mv,
                                previous=prev_v,
                                summary=summary,
                            )
                            prev_s = "none" if prev_v is None else _fmt_eval_metric(prev_v)
                            print(
                                f"  [best-holdout] {best_holdout_metric}={_fmt_eval_metric(mv)} "
                                f"(prev={prev_s}) step={step} marker={Path(cfg.checkpoint_dir) / 'best_holdout.json'}",
                                flush=True,
                            )
                else:
                    y_ev, _, _, _, idx_ev = model.forward_full(batch0)
                    to_eval: list[mx.array] = [y_ev]
                    if idx_ev is not None:
                        to_eval.extend(idx_ev)
                    mx.eval(*to_eval)
                    o = np.array(batch0[0, :, 0], dtype=np.float64)
                    r = np.array(y_ev[0, :, 0], dtype=np.float64)
                    summary = eval_metrics_mod.quality_metrics_16k(o, r)
                    h_mean, idx_bps = None, None
                    if idx_ev is not None and len(idx_ev) > 0:
                        code_rows = [[int(x) for x in np.array(ix).ravel().tolist()] for ix in idx_ev]
                        h_mean, idx_bps = _rvq_entropy_summary(cfg, code_rows)
                    summary["rvq_h_mean_bits_sym"] = h_mean
                    summary["empirical_index_bps"] = idx_bps
                    summary["n"] = 1
                    print(_format_eval_summary("train-probe", summary), flush=True)
                    _append_eval_tsv(cfg.log_mlx_tsv, step, "train_probe", summary)
            except Exception as e:
                print(f"  [eval-warn] {type(e).__name__}: {e}", flush=True)

        if cfg.checkpoint_every > 0 and step > 0:
            if step % cfg.checkpoint_every == 0 or step == cfg.steps - 1:
                ck_dir = Path(cfg.checkpoint_dir)
                ck_dir.mkdir(parents=True, exist_ok=True)
                ck_path = ck_dir / f"codec_step{step}.npz"
                model.save_weights(str(ck_path))
                print(f"  [checkpoint] {ck_path}", flush=True)
                if cfg.full_checkpoint:
                    full_p = ck_dir / f"ckpt_step{step}.safetensors"
                    save_full_checkpoint(
                        full_p,
                        step=step,
                        data_off=data_off,
                        model=model,
                        opt=opt,
                        lr_plateau_state=(
                            plateau_schedule.state_dict() if plateau_schedule is not None else None
                        ),
                    )
                    print(f"  [checkpoint] {full_p}", flush=True)
                if cfg.results_tsv_path:
                    rp = Path(cfg.results_tsv_path)
                    rp.parent.mkdir(parents=True, exist_ok=True)
                    need_hdr = not rp.is_file()
                    with rp.open("a") as rf:
                        if need_hdr:
                            rf.write(
                                "cycle\tphase\thypothesis\tarch_id\tbitrate_bps\tpesq_est\tvisqol_est\t"
                                "latency_ms\tparams_M\tverdict\tkey_finding\tnext_action\n"
                            )
                        bps = nom_kbps * 1000.0
                        rf.write(
                            f"0\tmlx_train\tplan_arch\tmlx_cfg\t{bps:.1f}\tna\tna\tna\t{n_params / 1e6:.2f}\t"
                            f"train\tstep={step}\tcheckpoint\n"
                        )

        # Spectrogram viz after the optimizer step.
        se = cfg.spectrogram_every
        viz_model = model
        if se > 0 and step > 0:
            if step % se == 0 or step == cfg.steps - 1:
                if cfg.spectrogram_seconds > 0:
                    n_viz = max(
                        int(cfg.spectrogram_seconds * cfg.sample_rate),
                        cfg.segment,
                    )
                    if audio_paths is not None:
                        batch_viz = _load_audio_viz_clip(cfg, audio_paths, step, n_viz)
                    else:
                        batch_viz = synth_viz_clip(cfg, step + cfg.seed * 10007, n_viz)
                    src_for_viz = batch_viz
                    y_viz, _, _, _, _ = viz_model.forward_full(batch_viz)
                    orig_1d = batch_viz[0, :, 0]
                    recon_1d = y_viz[0, :, 0]
                else:
                    src_for_viz = batch
                    y_viz, _, _, _, _ = viz_model.forward_full(batch)
                    orig_1d = batch[0, :, 0]
                    recon_1d = y_viz[0, :, 0]
                # Materialize full batch + output so 1D views are valid for numpy/soundfile.
                mx.eval(src_for_viz, y_viz)
                mx.eval(orig_1d, recon_1d)
                out = Path(cfg.spectrogram_dir) / f"step_{step:08d}.png"
                ok = save_spectrogram_png(
                    orig_1d,
                    recon_1d,
                    cfg.sample_rate,
                    out,
                    step,
                )
                if ok:
                    print(f"  [spectrogram] {out}", flush=True)
                else:
                    print(
                        "  [spectrogram] skipped — pip install matplotlib",
                        flush=True,
                    )
                if cfg.save_audio:
                    stem = Path(cfg.spectrogram_dir) / f"step_{step:08d}"
                    if save_reconstruction_wavs(
                        orig_1d,
                        recon_1d,
                        cfg.sample_rate,
                        stem,
                    ):
                        print(
                            f"  [audio] {stem}_orig.wav  {stem}_recon.wav",
                            flush=True,
                        )
                    else:
                        print(
                            "  [audio] skipped — pip install soundfile",
                            flush=True,
                        )

        if prof_on:
            prof_count += 1

    if prefetch_ex is not None:
        prefetch_ex.shutdown(wait=False, cancel_futures=True)

    total = time.time() - t0
    ran = cfg.steps - start_step
    if ran > 0:
        print(
            f"done steps {start_step}…{cfg.steps - 1}  ({ran} steps) in {total:.1f}s "
            f"({total / ran * 1000:.1f} ms/step)"
        )
    else:
        print("done (0 steps)")


if __name__ == "__main__":
    main()
