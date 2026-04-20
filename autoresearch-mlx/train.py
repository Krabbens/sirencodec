#!/usr/bin/env python3
"""
MLX codec autoresearch runner for sirencodec.

Editable: model config, optimizer, schedules, batching — see program.md.
Uses the same loss / ``val_bpb`` definition as ``prepare.py`` via ``tools/train_mlx``.
"""
from __future__ import annotations

import math
import os
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prepare as prep

from tools.train_mlx import (
    FAST_STFT_SCALES,
    Config,
    MLXCodec,
    _eval_loss_and_grad_tree,
    _grad_tree_any_nonfinite,
    _load_audio_batch,
    build_lr_schedule,
    clip_gradients_global_norm,
    make_train_fn,
    mel_filterbank_mx,
    synth_batch,
)

# --- experiment knobs (wall-clock training budget; program.md §5 min) ---
TRAIN_WALL_SECONDS = float(os.environ.get("AUTORESEARCH_TRAIN_SECONDS", "300"))
LOG_EVERY = 50
# Cosine LR horizon (only affects schedule shape; wall clock is TRAIN_WALL_SECONDS).
LR_SCHEDULE_STEPS = 100_000


def _reset_peak_memory() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def _peak_vram_mb() -> float:
    try:
        return float(mx.get_peak_memory()) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _rss_mb_approx() -> float:
    """Best-effort RSS (macOS: bytes in ru_maxrss; Linux: KB)."""
    try:
        r = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0.0
    if sys.platform == "darwin":
        return r / (1024.0 * 1024.0)
    return r / 1024.0


def run() -> None:
    total_t0 = time.time()
    _reset_peak_memory()

    cfg = Config()
    cfg.steps = LR_SCHEDULE_STEPS
    # experiment: match ``train_mlx --fast`` — two STFT scales → cheaper steps → more updates / 300s.
    cfg.stft_scales = FAST_STFT_SCALES
    # experiment: narrower latent → faster steps → more updates / 300s.
    cfg.latent_dim = 384
    # Shorter ramps vs long production runs so short budget runs see fuller STFT / marginal weights.
    cfg.stft_ramp_steps = min(cfg.stft_ramp_steps, 8000)
    cfg.marginal_boost_steps = min(cfg.marginal_boost_steps, 8000)
    cfg.vq_reset_every = 0
    cfg.log_every = LOG_EVERY
    cfg.spectrogram_every = 0
    cfg.checkpoint_every = 0

    mx.random.seed(cfg.seed)

    manifest = prep.ensure_manifest()
    train_paths, val_paths = prep.split_paths(manifest)

    model = MLXCodec(cfg)
    mx.eval(model.parameters())

    lr_spec = build_lr_schedule(cfg)
    opt = Adam(lr_spec)
    opt.init(model.parameters())

    fmax = cfg.mel_fmax if cfg.mel_fmax is not None else cfg.sample_rate / 2.0
    mel_fb = None
    if cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0:
        mel_fb = mel_filterbank_mx(
            cfg.sample_rate, cfg.mel_n_fft, cfg.n_mels, cfg.mel_fmin, float(fmax)
        )

    data_off = 0
    step = 0
    training_seconds = 0.0

    while training_seconds < TRAIN_WALL_SECONDS:
        t_step = time.time()
        if train_paths:
            batch = _load_audio_batch(cfg, train_paths, data_off)
            data_off += cfg.batch
        else:
            batch = synth_batch(cfg, key=step + cfg.seed * 10007)

        loss_fn = make_train_fn(model, cfg, batch, step, mel_fb)
        lg = nn.value_and_grad(model, loss_fn)
        loss, grads = lg(model)
        _eval_loss_and_grad_tree(loss, grads)
        lv0 = float(loss.item())
        if not math.isfinite(lv0) or _grad_tree_any_nonfinite(grads):
            print(
                "---\nval_bpb:          0.000000\ntraining_seconds: "
                f"{training_seconds:.3f}\ntotal_seconds:    {time.time() - total_t0:.3f}\n"
                "peak_vram_mb:     0.0\n(non-finite loss or grad)",
                flush=True,
            )
            sys.exit(1)
        if cfg.grad_clip_norm > 0:
            grads = clip_gradients_global_norm(grads, cfg.grad_clip_norm)

        if step % LOG_EVERY == 0:
            print(
                f"step {step:6d}  loss={lv0:.5f}  train_s={training_seconds:.1f}/{TRAIN_WALL_SECONDS:.0f}",
                flush=True,
            )

        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        training_seconds += time.time() - t_step
        step += 1

    last_step = max(0, step - 1)
    val = prep.evaluate_bpb(
        model,
        cfg,
        step=last_step,
        mel_fb=mel_fb,
        train_paths=train_paths,
        val_paths=val_paths,
    )
    total_seconds = time.time() - total_t0
    peak_mb = _peak_vram_mb()

    print("---", flush=True)
    print(f"val_bpb:          {val:.6f}", flush=True)
    print(f"training_seconds: {training_seconds:.3f}", flush=True)
    print(f"total_seconds:    {total_seconds:.3f}", flush=True)
    print(f"peak_vram_mb:     {peak_mb:.3f}", flush=True)
    print(f"rss_mb_approx:    {_rss_mb_approx():.3f}", flush=True)


if __name__ == "__main__":
    run()
