#!/usr/bin/env python3
"""
train_mlx.py — Karpathy-style single-file MLX neural audio codec (Apple Silicon).

Inlined stack: SEANet-ish encoder → **residual VQ** (EnCodec-style) → decoder,
time + **multi-scale** log-STFT loss + default **log-mel bin L1** only (``λ_mel_L1=0.06``, ``λ_mel_L2=0``); set ``--lambda-mel-l1 0`` to disable mel, optional WAV batches. No PyTorch / sirencodec.

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
  **Resume:** ``--resume`` loads weights; continues from step ``N+1``. Adam moments re-init; **LR schedule** uses global step ``N+1`` (no restart of cosine). ``--lr-schedule none`` keeps constant ``--lr``.
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
from .codec import MLXCodec, VectorQuantizerStage
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
    mel_filterbank_mx,
    mel_log_bin_losses,
    multi_stft_complex_l1,
    multi_stft_loss,
    multi_stft_mag_l1_linear,
    multi_stft_spectral_convergence,
    multi_stft_spectral_terms,
)


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


def _eval_loss_and_grad_tree(loss: mx.array, grads) -> None:
    """Materialize backward: MLX keeps ``grads`` lazy until evaluated; ``opt.update`` needs real values."""
    flat = tree_flatten(grads)
    arrs = [a for _, a in flat]
    if arrs:
        mx.eval(loss, *arrs)
    else:
        mx.eval(loss)


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
        y_hat, vq_l, ent_pos, marg_ent, idx = m.forward_full(batch)
        lt = mx.mean(mx.abs(y_hat - batch))
        if cfg.lambda_stft_grad > 0 or cfg.lambda_stft_cos > 0:
            ls, lsg, lsc = multi_stft_spectral_terms(
                y_hat,
                batch,
                cfg.stft_scales,
                with_grad=cfg.lambda_stft_grad > 0,
                with_cos_1m=cfg.lambda_stft_cos > 0,
                grad_freq_weight=cfg.stft_grad_freq_weight,
                grad_time_weight=cfg.stft_grad_time_weight,
                hf_emphasis=cfg.stft_hf_emphasis,
                scale_weights=cfg.stft_scale_weights,
            )
        else:
            if compiled_multi_stft is not None:
                ls = compiled_multi_stft(y_hat, batch)
            else:
                ls = multi_stft_loss(
                    y_hat,
                    batch,
                    cfg.stft_scales,
                    hf_emphasis=cfg.stft_hf_emphasis,
                    scale_weights=cfg.stft_scale_weights,
                )
            lsg = mx.array(0.0)
            lsc = mx.array(0.0)
        cos = batch_mean_cosine(batch, y_hat)
        ls_w = effective_lambda_stft(step, cfg)
        total = cfg.lambda_time * lt + ls_w * ls + cfg.lambda_vq * vq_l
        l_lin = mx.array(0.0)
        if cfg.lambda_mag_l1 > 0:
            l_lin = multi_stft_mag_l1_linear(
                y_hat,
                batch,
                cfg.stft_scales,
                hf_emphasis=cfg.stft_hf_emphasis,
                scale_weights=cfg.stft_scale_weights,
            )
            total = total + ls_w * cfg.lambda_mag_l1 * l_lin
        if cfg.lambda_stft_grad > 0:
            total = total + ls_w * cfg.lambda_stft_grad * lsg
        if cfg.lambda_stft_cos > 0:
            total = total + ls_w * cfg.lambda_stft_cos * lsc
        l_sc = mx.array(0.0)
        if cfg.lambda_sc > 0:
            l_sc = multi_stft_spectral_convergence(
                y_hat,
                batch,
                cfg.stft_scales,
                scale_weights=cfg.stft_scale_weights,
            )
            total = total + ls_w * cfg.lambda_sc * l_sc
        l_cx = mx.array(0.0)
        if cfg.lambda_complex_stft > 0:
            l_cx = multi_stft_complex_l1(
                y_hat,
                batch,
                cfg.stft_scales,
                scale_weights=cfg.stft_scale_weights,
            )
            total = total + ls_w * cfg.lambda_complex_stft * l_cx
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
        lm1 = mx.array(0.0)
        lm2 = mx.array(0.0)
        if mel_fb is not None and (cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0):
            lm1, lm2 = mel_log_bin_losses(
                y_hat,
                batch,
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
            l_lin=l_lin,
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
        res_for_km = residual
        if stage.in_proj is not None:
            res_for_km = stage.in_proj(residual)
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
    quantized = mx.zeros_like(z)
    for i in range(cfg.n_codebooks):
        stage: VectorQuantizerStage = getattr(model.rvq, f"q{i}")
        residual = z - quantized
        z_i, _, _, idx = stage(residual)
        mx.eval(z_i, idx)
        r = stage.in_proj(residual) if stage.in_proj is not None else residual
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = min(22.0, max(10.0, 6.0 + dur * 1.5))
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 6), sharex=True)
    for ax, data, title in (
        (axes[0], o, "original"),
        (axes[1], r, "reconstruction"),
    ):
        s_db = _log_mag_spectrogram_np(data, sample_rate, n_fft, hop)
        t1 = data.size / float(sample_rate)
        ax.imshow(
            s_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0.0, t1, 0.0, sample_rate / 2.0],
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
) -> None:
    """Single-safetensors checkpoint with model weights + Adam state. No disc / no EMA (GAN-free recipe)."""
    flat: dict[str, mx.array] = {}
    flat.update({f"model/{k}": v for k, v in dict(tree_flatten(model.parameters())).items()})
    flat.update({f"opt_g/{k}": v for k, v in dict(tree_flatten(opt.state)).items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), flat)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"step": step, "data_off": data_off}, indent=0) + "\n")


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


def build_lr_schedule(
    cfg: Config, *, lr_scale: float = 1.0
) -> float | Callable[[mx.array], mx.array]:
    """Return constant LR or an MLX scheduler callable (``step`` = optimizer step counter).

    ``lr_scale`` scales the peak (and cosine floor).
    """
    scale = float(lr_scale)
    mode = (cfg.lr_schedule or "none").strip().lower()
    if mode in ("none", "constant", "off"):
        return float(cfg.lr) * scale
    if mode != "cosine":
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r} (use none or cosine)")
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
    p.add_argument("--lr", type=float, default=5e-4, help="Adam peak LR (cosine) or constant LR (--lr-schedule none)")
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=("none", "cosine"),
        help="LR schedule: cosine decay to --lr×--lr-min-ratio by final step, or none (constant --lr)",
    )
    p.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.1,
        metavar="R",
        help="Cosine floor = --lr × R (only for cosine schedule; use 0 for decay to 0)",
    )
    p.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        metavar="W",
        help="Linear warmup steps 0→--lr before cosine (0 = off)",
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
        default=0.06,
        metavar="W",
        help="add W·λ_stft_eff·mean|Δlog mel| (0=off). Default 0.06 (L1-only mel); lower if recon smears",
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
        choices=("gelu", "snake", "snake_beta"),
        help="Encoder/decoder nonlinearity (snake/snake_beta = DAC-style periodic)",
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
        "--decoder-upsample",
        type=str,
        default="transpose",
        choices=("transpose", "repeat_conv"),
        help="Decoder upsampling: ConvTranspose1d or repeat×2+Conv (also forced when --causal)",
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
        help="Run SI-SDR (+ PESQ if installed) every N steps on a short val clip (0=off)",
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
        "--log-cos-ema-beta",
        type=float,
        default=0.99,
        metavar="B",
        help="EMA of cos%% in log: ema←B·ema+(1−B)·cos (0=off, typical 0.95–0.995)",
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
    args = p.parse_args()

    if not args.data_dir and args.librispeech:
        args.data_dir = str(Path("data") / "cv-corpus")

    if args.spectrogram_seconds < 0:
        print("--spectrogram-seconds must be >= 0", file=sys.stderr)
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
    if args.lr_min_ratio < 0 or args.lr_min_ratio > 1.0:
        print("--lr-min-ratio must be in [0, 1]", file=sys.stderr)
        sys.exit(1)
    if args.lr_warmup_steps < 0:
        print("--lr-warmup-steps must be >= 0", file=sys.stderr)
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
    if args.stft_grad_freq_weight < 0 or args.stft_grad_time_weight < 0:
        print("--stft-grad-freq-weight and --stft-grad-time-weight must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_stft_grad > 0 and args.stft_grad_freq_weight <= 0 and args.stft_grad_time_weight <= 0:
        print("with --lambda-stft-grad>0, at least one of --stft-grad-freq-weight / --stft-grad-time-weight must be >0", file=sys.stderr)
        sys.exit(1)
    if args.stft_hf_emphasis < 0:
        print("--stft-hf-emphasis must be >= 0", file=sys.stderr)
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
    if args.log_cos_ema_beta < 0 or args.log_cos_ema_beta >= 1.0:
        print("--log-cos-ema-beta must be in [0, 1) (0 = disable EMA)", file=sys.stderr)
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
    if args.vq_reset_full_refresh_max_unique < 1:
        print("--vq-reset-full-refresh-max-unique must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.latent_temporal_depth < 0 or args.latent_temporal_post_depth < 0:
        print("--latent-temporal-depth and --latent-temporal-post-depth must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_sc < 0:
        print("--lambda-sc must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.lambda_complex_stft < 0:
        print("--lambda-complex-stft must be >= 0", file=sys.stderr)
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

    if args.stft_scales:
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
    if args.stft_scale_weights is not None:
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
        grad_clip_norm=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        seed=args.seed,
        ae_only=args.ae_only,
        enc_channels=enc_ch,
        stride1_blocks_per_scale=args.stride1_blocks_per_scale,
        latent_dim=args.latent_dim,
        pre_vq_layernorm=args.pre_vq_layernorm,
        latent_temporal_depth=args.latent_temporal_depth,
        latent_temporal_post_depth=args.latent_temporal_post_depth,
        n_codebooks=args.n_codebooks,
        codebook_size=args.codebook_size,
        codebook_sizes=codebook_sizes_arg,
        lambda_time=args.lambda_time,
        lambda_stft=args.lambda_stft,
        lambda_stft_grad=args.lambda_stft_grad,
        lambda_stft_cos=args.lambda_stft_cos,
        stft_grad_freq_weight=args.stft_grad_freq_weight,
        stft_grad_time_weight=args.stft_grad_time_weight,
        stft_ramp_steps=args.stft_ramp_steps,
        stft_ramp_start_frac=args.stft_ramp_start,
        stft_scales=stft_scales_resolved,
        stft_scale_weights=stft_scale_weights_resolved,
        stft_hf_emphasis=args.stft_hf_emphasis,
        mel_n_fft=args.mel_n_fft,
        mel_hop=args.mel_hop,
        n_mels=args.n_mels,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
        lambda_mel_l1=args.lambda_mel_l1,
        lambda_mel_l2=args.lambda_mel_l2,
        lambda_vq=args.lambda_vq,
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
        lambda_sc=args.lambda_sc,
        lambda_complex_stft=args.lambda_complex_stft,
        log_every=args.log_every,
        log_cos_ema_beta=args.log_cos_ema_beta,
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
        decoder_upsample=args.decoder_upsample,
        causal=bool(args.causal),
        use_bf16=bool(args.bf16),
        use_compile=bool(args.compile_loss),
        full_checkpoint=bool(args.full_checkpoint),
        eval_every=args.eval_every,
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
                meta = json.loads(meta_path.read_text())
                res_ck_n = int(meta.get("step", _parse_full_checkpoint_step(ck_path) or 0))
                resume_data_off = int(meta.get("data_off", 0))
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

    lr_spec = build_lr_schedule(cfg)
    opt = Adam(lr_spec)
    opt.init(model.parameters())
    if start_step > 0:
        opt.state["step"] = mx.array(int(start_step), dtype=mx.uint64)
        _refresh_optimizer_schedules(opt)

    if resume_flat is not None:
        load_full_checkpoint_into(resume_flat, model=model, opt=opt)
        mx.eval(model.parameters(), opt.state)
        print("[resume] loaded full safetensors (weights + optimizer state)", flush=True)

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
    if (cfg.lr_schedule or "").lower() in ("none", "constant", "off", ""):
        lr_info = f"lr={cfg.lr} (const)"
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
    lt_pre_post = f"lat_tmp={cfg.latent_temporal_depth}/{cfg.latent_temporal_post_depth}"
    sc_info = f"  λ_sc={cfg.lambda_sc:g}" if cfg.lambda_sc > 0 else ""
    cx_info = f"  λ_cx={cfg.lambda_complex_stft:g}" if cfg.lambda_complex_stft > 0 else ""
    _acm_p = max(1, int(cfg.grad_accum_steps))
    batch_info = (
        f"  batch={cfg.batch}×accum={_acm_p}→{cfg.batch * _acm_p}"
        if _acm_p > 1
        else f"  batch={cfg.batch}"
    )
    print(
        f"Parameters: {n_params / 1e6:.2f}M{batch_info}  latent={cfg.latent_dim}  {lt_pre_post}  pre_vq_ln={cfg.pre_vq_layernorm}  enc_stride={st}×  "
        f"s1/scale={cfg.stride1_blocks_per_scale}  "
        f"grad_clip={cfg.grad_clip_norm}  {lr_info}  "
        f"~{nom_kbps:.1f} kbps nominal  "
        f"STFT×{len(cfg.stft_scales)}: {stft_nfo}{stft_w_info}  stft_hf={cfg.stft_hf_emphasis:g}  "
        f"λ_L1={cfg.lambda_time}  λ_stft={cfg.lambda_stft}  λ_stft_grad={cfg.lambda_stft_grad}{sc_info}{cx_info}  "
        f"sgrad_df/dt={cfg.stft_grad_freq_weight}/{cfg.stft_grad_time_weight}  λ_stft_cos={cfg.lambda_stft_cos}  {ramp_s}"
        f"RVQ={cfg.n_codebooks}×K={rvq_ks}  vq_cos={cfg.vq_cosine}  λ_vq={cfg.lambda_vq}  λ_ent={cfg.lambda_entropy}  "
        f"λ_marg={cfg.lambda_marginal}  τ_marg={cfg.marginal_tau}  "
        f"{'λ_marg_boost=' + str(cfg.marginal_boost_steps) + '@×' + str(cfg.marginal_boost_mult) + '→1  ' if cfg.marginal_boost_steps > 0 else ''}"
        f"vq_reset={cfg.vq_reset_every}@{cfg.vq_reset_collapse_frac}  "
        f"σ={cfg.vq_reset_noise}  shuffle={cfg.vq_reset_shuffle}  vq_km={cfg.vq_reset_kmeans}  "
        f"vq_full≤{cfg.vq_reset_full_refresh_max_unique}u→allK  vq_rst_log={cfg.vq_reset_log_every}  "
        f"λ_cos={cfg.lambda_cos}  cos_hinge={cfg.cos_hinge}@{cfg.cos_target}"
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
    ema_cos_pct: float | None = None

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
        # Single sync at end of accum: evaluates loss + accumulated grads together.
        _eval_loss_and_grad_tree(loss_acc, grads_acc)
        lv_m_total = float(loss_acc.item())
        if not math.isfinite(lv_m_total):
            print(f"  [skip] non-finite summed loss at step {step}", flush=True)
            continue
        _ptoc("G_fwdbwd", _t_g)

        _t_misc = _ptic()
        lv0 = lv_m_total * inv_acm
        grads = grads_acc
        if _grad_tree_any_nonfinite(grads):
            print(f"  [skip] non-finite accumulated gradient at step {step}", flush=True)
            continue
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
            to_ev = [lt, ls, lsg, lsc, lm1, lm2, vq_l, ent_pos, marg_ent, cos_m]
            if isinstance(l_lin, mx.array):
                to_ev.append(l_lin)
            if isinstance(l_sc, mx.array):
                to_ev.append(l_sc)
            if isinstance(l_cx, mx.array):
                to_ev.append(l_cx)
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
            ent_extra = ""
            if cfg.lambda_entropy > 0:
                ent_extra = f"  ent_pos={float(ent_pos.item()):.4f}"
            w_stft = effective_lambda_stft(step, cfg)
            w_marg = effective_lambda_marginal(step, cfg)
            ramp_extra = f"  λ_stft_eff={w_stft:.4f}" if cfg.stft_ramp_steps > 0 else ""
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
                f"L1={float(lt.item()):.4f}  stft={float(ls.item()):.4f}{sgrad_extra}{scos_extra}{sc_extra}{cx_extra}{mel_extra}{lin_extra}  "
                f"vq={vq_s}  marg_ent={float(marg_ent.item()):.4f}{ent_extra}{ramp_extra}  "
                f"cos={cos_pct:.1f}%{cos_ema_extra}{util}  lr={lr_log:.2e}  "
                f"{elapsed / (step - start_step + 1) * 1000:.1f} ms/step{prof_extra}",
                flush=True,
            )
        _t_opt = _ptic()
        opt.update(model, grads)
        _ptoc("optim", _t_opt)

        if 0.0 < float(cfg.vq_ema_decay) < 1.0 and not cfg.ae_only:
            _t_vqe = _ptic()
            update_vq_ema_codebooks(model, batch0, cfg)
            mx.eval(model.parameters())
            _ptoc("vq_ema", _t_vqe)

        if cfg.eval_every > 0 and step > 0 and step % int(cfg.eval_every) == 0:
            try:
                import numpy as np
            except ImportError:
                np = None  # type: ignore
            if np is not None:
                y_ev, _, _, _, idx_ev = model.forward_full(batch0)
                mx.eval(y_ev)
                o = np.array(batch0[0, :, 0], dtype=np.float64)
                r = np.array(y_ev[0, :, 0], dtype=np.float64)
                sisdr = eval_metrics_mod.si_sdr_db(o, r)
                pesq_v = eval_metrics_mod.pesq_wb_16k(o, r)
                pesq_s = f"{pesq_v:.3f}" if pesq_v is not None else "na"
                print(f"  [eval] SI-SDR={sisdr:.2f} dB  PESQ_wb={pesq_s}", flush=True)
                if cfg.log_mlx_tsv:
                    p = Path(cfg.log_mlx_tsv)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    hdr = not p.is_file()
                    with p.open("a") as f:
                        if hdr:
                            f.write("step\tsisdr_db\tpesq_wb\n")
                        f.write(f"{step}\t{sisdr:.6f}\t{pesq_s}\n")
                if idx_ev is not None and len(idx_ev) > 0:
                    sizes = effective_codebook_sizes(cfg)
                    h_stages: list[float] = []
                    for si, ix in enumerate(idx_ev):
                        mx.eval(ix)
                        kk = sizes[si] if si < len(sizes) else sizes[-1]
                        row = [int(x) for x in np.array(ix).ravel().tolist()]
                        h_stages.append(entropy_coding_mod.empirical_cross_entropy_bits_per_symbol(row, kk))
                    h_mean = sum(h_stages) / float(len(h_stages))
                    fps = cfg.sample_rate / float(encoder_time_stride(cfg))
                    idx_bps = h_mean * fps * float(len(idx_ev))
                    print(
                        f"  [entropy] RVQ mean H={h_mean:.3f} b/sym  ~{idx_bps:.0f} b/s empirical index (vs nominal)",
                        flush=True,
                    )

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
