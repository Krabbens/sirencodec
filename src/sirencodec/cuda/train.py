#!/usr/bin/env python3
"""CUDA trainer replacing the old MLX trainer implementation.

The command stays compatible with ``tools/train_mlx.py`` so existing scripts keep working:

  uv run python tools/train_mlx.py --epochs 1 --no-librispeech --fast
  uv run python tools/train_mlx.py --epochs 10 --librispeech
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import wave
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from .. import adaptive as adaptive_mod
from .. import entropy_coding as entropy_coding_mod
from .. import eval_metrics as eval_metrics_mod
from ..config import (
    DEFAULT_STFT_SCALES,
    Config,
    effective_codebook_sizes,
    encoder_time_stride,
    mean_log_codebook_for_entropy,
    nominal_rvq_kbps,
    parse_codebook_sizes_arg,
    parse_stft_scales_arg,
    parse_stft_scale_weights_arg,
)
from .codec import CUDACodec, VectorQuantizerStage
from .data import collect_audio_paths, load_audio_batch, load_audio_batch_cpu, load_audio_viz_clip, synth_batch, synth_viz_clip
from .losses import (
    mel_filterbank_torch,
    mel_log_bin_losses,
    multi_stft_all_terms,
)


def effective_lambda_stft(step: int, cfg: Config) -> float:
    if cfg.stft_ramp_steps <= 0:
        return cfg.lambda_stft
    t = min(1.0, float(step) / float(cfg.stft_ramp_steps))
    return cfg.lambda_stft * (cfg.stft_ramp_start_frac + (1.0 - cfg.stft_ramp_start_frac) * t)


def effective_lambda_marginal(step: int, cfg: Config) -> float:
    if cfg.lambda_marginal <= 0 or cfg.marginal_boost_steps <= 0:
        return cfg.lambda_marginal
    m = max(1.0, float(cfg.marginal_boost_mult))
    t = min(1.0, float(step) / float(cfg.marginal_boost_steps))
    return cfg.lambda_marginal * (m + (1.0 - m) * t)


class CurriculumState(NamedTuple):
    phase: str
    ae_only: bool
    vq_mult: float
    entropy_mult: float
    marginal_mult: float
    reset_enabled: bool


def _smooth01(x: float) -> float:
    t = min(1.0, max(0.0, float(x)))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def curriculum_state(step: int, cfg: Config) -> CurriculumState:
    if not bool(getattr(cfg, "curriculum", False)):
        return CurriculumState("off", bool(cfg.ae_only), 1.0, 1.0, 1.0, True)
    if cfg.ae_only:
        return CurriculumState("ae", True, 0.0, 0.0, 0.0, False)
    total = max(1, int(cfg.steps))
    ae_steps = max(0, int(round(total * max(0.0, float(cfg.curriculum_ae_frac)))))
    ramp_steps = max(1, int(round(total * max(0.0, float(cfg.curriculum_vq_ramp_frac)))))
    if step < ae_steps:
        return CurriculumState("A/AE", True, 0.0, 0.0, 0.0, False)
    if step < ae_steps + ramp_steps:
        t = _smooth01((step - ae_steps + 1) / float(ramp_steps))
        vq_start = min(1.0, max(0.0, float(cfg.curriculum_vq_start)))
        ent_start = min(1.0, max(0.0, float(cfg.curriculum_entropy_start)))
        vq_mult = vq_start + (1.0 - vq_start) * t
        ent_mult = ent_start + (1.0 - ent_start) * t
        return CurriculumState("B/RVQ", False, vq_mult, ent_mult, ent_mult, False)
    return CurriculumState("D/full", False, 1.0, 1.0, 1.0, True)


def batch_mean_cosine(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(orig.reshape(orig.shape[0], -1), recon.reshape(recon.shape[0], -1), dim=1, eps=1e-8).mean()


def _balanced_reconstruction_loss(
    terms: list[tuple[str, torch.Tensor, float]],
    y_hat: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    active = [(name, loss, float(weight)) for name, loss, weight in terms if float(weight) > 0]
    if not active:
        return y_hat.new_zeros(()), {}
    mode = (cfg.loss_balancer or "off").lower()
    if mode == "off":
        return sum(weight * loss for _, loss, weight in active), {}
    if mode != "grad":
        raise ValueError(f"unknown loss_balancer={cfg.loss_balancer!r}")
    weight_sum = sum(weight for _, _, weight in active)
    if weight_sum <= 0:
        return y_hat.new_zeros(()), {}
    norms: list[torch.Tensor] = []
    for _, loss, _ in active:
        grad = torch.autograd.grad(loss, y_hat, retain_graph=True, create_graph=False, allow_unused=True)[0]
        if grad is None:
            norms.append(y_hat.new_zeros(()))
        else:
            norms.append(torch.linalg.vector_norm(grad.detach()) / math.sqrt(max(1, grad.numel())))
    ratios = [weight / weight_sum for _, _, weight in active]
    avg_norm = sum(norm * ratio for norm, ratio in zip(norms, ratios)).detach()
    eps = float(cfg.loss_balancer_eps)
    total = y_hat.new_zeros(())
    scales: dict[str, float] = {}
    for (name, loss, _), norm, ratio in zip(active, norms, ratios):
        scale_t = (float(weight_sum) * ratio * avg_norm / norm.clamp_min(eps)).clamp(max=float(cfg.loss_balancer_max_scale)).detach()
        total = total + scale_t * loss
        scales[name] = float(scale_t.float().item())
    return total, scales


def compute_loss(model: CUDACodec, cfg: Config, batch: torch.Tensor, step: int, mel_fb: torch.Tensor | None):
    curr = curriculum_state(step, cfg)
    old_ae = bool(model.cfg.ae_only)
    model.cfg.ae_only = bool(curr.ae_only)
    try:
        y_hat, vq_l, ent_pos, marg_ent, idx = model.forward_full(batch)
    finally:
        model.cfg.ae_only = old_ae
    lt = torch.mean(torch.abs(y_hat - batch))
    ls, lsg, lsc, l_lin, l_sc, l_cx = multi_stft_all_terms(
        y_hat,
        batch,
        cfg.stft_scales,
        with_grad=cfg.lambda_stft_grad > 0,
        with_cos_1m=cfg.lambda_stft_cos > 0,
        with_linear=cfg.lambda_mag_l1 > 0,
        with_sc=cfg.lambda_sc > 0,
        with_complex=cfg.lambda_complex_stft > 0,
        grad_freq_weight=cfg.stft_grad_freq_weight,
        grad_time_weight=cfg.stft_grad_time_weight,
        hf_emphasis=cfg.stft_hf_emphasis,
        scale_weights=cfg.stft_scale_weights,
    )
    cos = batch_mean_cosine(batch, y_hat)
    ls_w = effective_lambda_stft(step, cfg)
    lm1 = batch.new_zeros(())
    lm2 = batch.new_zeros(())
    if mel_fb is not None and (cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0):
        lm1, lm2 = mel_log_bin_losses(y_hat, batch, mel_fb, cfg.mel_n_fft, cfg.mel_hop)
    recon_terms = [
        ("time", lt, float(cfg.lambda_time)),
        ("stft", ls, float(ls_w)),
        ("mag", l_lin, float(ls_w * cfg.lambda_mag_l1)),
        ("sgrad", lsg, float(ls_w * cfg.lambda_stft_grad)),
        ("stft_cos", lsc, float(ls_w * cfg.lambda_stft_cos)),
        ("sc", l_sc, float(ls_w * cfg.lambda_sc)),
        ("complex", l_cx, float(ls_w * cfg.lambda_complex_stft)),
        ("mel_l1", lm1, float(ls_w * cfg.lambda_mel_l1)),
        ("mel_l2", lm2, float(ls_w * cfg.lambda_mel_l2)),
        ("wave_cos", 1.0 - cos, float(cfg.lambda_cos)),
    ]
    recon_total, bal_scales = _balanced_reconstruction_loss(recon_terms, y_hat, cfg)
    total = recon_total
    if cfg.cos_hinge > 0:
        total = total + cfg.cos_hinge * (float(cfg.cos_target) - cos).clamp_min(0.0)
    if cfg.lambda_vq > 0 and curr.vq_mult > 0:
        total = total + (cfg.lambda_vq * curr.vq_mult) * vq_l
    if cfg.lambda_entropy > 0 and curr.entropy_mult > 0:
        total = total + (cfg.lambda_entropy * curr.entropy_mult) * (mean_log_codebook_for_entropy(cfg) - ent_pos).clamp_min(0.0)
    if cfg.lambda_marginal > 0 and curr.marginal_mult > 0:
        total = total + (effective_lambda_marginal(step, cfg) * curr.marginal_mult) * (mean_log_codebook_for_entropy(cfg) - marg_ent).clamp_min(0.0)
    return total, {
        "y_hat": y_hat,
        "phase": curr.phase,
        "phase_vq_mult": curr.vq_mult,
        "phase_entropy_mult": curr.entropy_mult,
        "phase_marginal_mult": curr.marginal_mult,
        "bal_scales": bal_scales,
        "recon_total": recon_total.detach(),
        "lt": lt.detach(),
        "ls": ls.detach(),
        "lsg": lsg.detach(),
        "lsc": lsc.detach(),
        "l_lin": l_lin.detach(),
        "l_sc": l_sc.detach(),
        "l_cx": l_cx.detach(),
        "lm1": lm1.detach(),
        "lm2": lm2.detach(),
        "vq_l": vq_l.detach(),
        "ent_pos": ent_pos.detach(),
        "marg_ent": marg_ent.detach(),
        "cos_m": cos.detach(),
        "idx": [i.detach() for i in idx] if idx is not None else None,
    }


def _format_vq_util(indices: list[torch.Tensor] | None, sizes: tuple[int, ...]) -> str:
    if not indices:
        return ""
    parts = []
    for i, idx in enumerate(indices):
        k = sizes[i] if i < len(sizes) else sizes[-1]
        n_unique = int(torch.unique(idx.detach()).numel())
        parts.append(f"u{i}={n_unique}/{k}({100.0 * n_unique / max(1, k):.1f}%)")
    return " ".join(parts)


class _Ansi:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.reset = "\033[0m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.red = "\033[31m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""
        self.blue = "\033[34m" if enabled else ""
        self.magenta = "\033[35m" if enabled else ""
        self.cyan = "\033[36m" if enabled else ""

    def c(self, color: str, text: object) -> str:
        return f"{getattr(self, color)}{text}{self.reset}"


def _make_ansi(mode: str) -> _Ansi:
    m = (mode or "auto").lower()
    if m == "always":
        return _Ansi(True)
    if m == "never" or "NO_COLOR" in os.environ:
        return _Ansi(False)
    return _Ansi(sys.stdout.isatty())


def _kv(label: str, value: object, ansi: _Ansi, *, color: str = "bold", width: int = 11) -> str:
    return f"{ansi.c('dim', label.ljust(width))} {ansi.c(color, value)}"


def _bar(ansi: _Ansi) -> str:
    return ansi.c("dim", "-" * 72)


def _fmt_duration(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def print_run_header(
    *,
    ansi: _Ansi,
    device: torch.device,
    device_name: str,
    cfg: Config,
    n_params: int,
    st: int,
    nom_kbps: float,
    cb_sizes: tuple[int, ...],
    stft_nfo: str,
    resolved_epochs: int,
    steps_per_epoch: int,
    epoch_source: str,
    audio_paths: list[Path] | None,
    run_dir: Path,
) -> None:
    samples_per_update = cfg.batch * cfg.grad_accum_steps
    epoch_items = len(audio_paths) if audio_paths is not None else steps_per_epoch * samples_per_update
    print()
    print(ansi.c("cyan", ansi.bold + "SirenCodec CUDA training" + ansi.reset))
    print(_bar(ansi))
    print(_kv("device", f"{device}" + (f" ({device_name})" if device_name else ""), ansi, color="green" if device.type == "cuda" else "yellow"))
    if audio_paths is not None:
        print(_kv("data", f"{len(audio_paths)} audio files from {cfg.data_dir}", ansi, color="cyan"))
    else:
        print(_kv("data", f"synthetic sine/noise ({epoch_source})", ansi, color="yellow"))
    print(_kv("schedule", f"epochs={resolved_epochs}  updates/epoch={steps_per_epoch}  total_updates={cfg.steps}", ansi, color="blue"))
    print(_kv("epoch", f"{epoch_items} samples  {samples_per_update} samples/update", ansi, color="blue"))
    print(_kv("logging", f"every {cfg.log_every} updates (~{max(1, math.ceil(steps_per_epoch / max(1, cfg.log_every)))} logs/epoch)", ansi, color="cyan"))
    print(_kv("run", str(run_dir), ansi, color="green"))
    if cfg.spectrogram_every > 0:
        print(_kv("viz", f"every {cfg.spectrogram_every} updates (~{max(1, math.ceil(steps_per_epoch / max(1, cfg.spectrogram_every)))}x/epoch) -> {cfg.spectrogram_dir}", ansi, color="cyan"))
    else:
        print(_kv("viz", "disabled", ansi, color="yellow"))
    print(_kv("batch", f"micro={cfg.batch}  accum={cfg.grad_accum_steps}  effective={cfg.batch * cfg.grad_accum_steps}", ansi))
    if audio_paths is not None:
        print(_kv("prefetch", f"threads={cfg.load_audio_threads}  batches={cfg.prefetch_audio_batches if cfg.prefetch_audio else 0}", ansi, color="cyan"))
    print(_kv("training", f"balancer={cfg.loss_balancer}  curriculum={'on' if cfg.curriculum else 'off'}", ansi, color="cyan"))
    print(_kv("model", f"params={n_params / 1e6:.2f}M  latent={cfg.latent_dim}  stride={st}x  ~{nom_kbps:.1f} kbps", ansi))
    print(_kv("stft", f"{len(cfg.stft_scales)} scales: {stft_nfo}", ansi, color="magenta"))
    print(_kv("rvq", f"{cfg.n_codebooks} stages  K={'x'.join(str(k) for k in cb_sizes)}  cosine={cfg.vq_cosine}", ansi, color="magenta"))
    print(_bar(ansi), flush=True)


def print_event(ansi: _Ansi, kind: str, message: str, *, color: str = "cyan") -> None:
    print(f"{ansi.c(color, '[' + kind + ']')} {message}", flush=True)


class _ProfileStats:
    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.device = device
        self.n_updates = 0
        self.times = {
            "data": 0.0,
            "train": 0.0,
            "opt": 0.0,
            "aux": 0.0,
            "misc": 0.0,
        }

    def mark(self) -> float:
        if not self.enabled:
            return 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def add_since(self, name: str, start: float) -> float:
        if not self.enabled:
            return 0.0
        end = self.mark()
        self.times[name] += end - start
        return end

    def step_done(self) -> None:
        if self.enabled:
            self.n_updates += 1

    def summary(self) -> str:
        n = max(1, self.n_updates)
        total = sum(self.times.values())

        def part(name: str, label: str) -> str:
            s = self.times[name]
            pct = 100.0 * s / max(total, 1e-9)
            return f"{label} {1000.0 * s / n:.1f}ms ({pct:.0f}%)"

        return (
            f"avg/update over {self.n_updates}: "
            f"{part('data', 'data')}  "
            f"{part('train', 'loss+bwd')}  "
            f"{part('opt', 'opt')}  "
            f"{part('aux', 'vq/eval')}  "
            f"{part('misc', 'log/viz/ckpt')}  "
            f"measured {1000.0 * total / n:.1f}ms"
        )


def print_step_log(
    *,
    ansi: _Ansi,
    step: int,
    cfg: Config,
    steps_per_epoch: int,
    resolved_epochs: int,
    loss_value: float,
    metrics: dict,
    cos_pct: float,
    ema_cos_pct: float | None,
    util: str,
    lr_log: float,
    ms_per_step: float,
    grad_norm: float,
    samples_per_update: int,
    epoch_items: int,
    elapsed: float,
    start_step: int,
) -> None:
    epoch_now = (step + 1) / float(max(1, steps_per_epoch))
    iter_in_epoch = (step % max(1, steps_per_epoch)) + 1
    epoch_index = min(resolved_epochs, (step // max(1, steps_per_epoch)) + 1)
    epoch_pct = 100.0 * iter_in_epoch / float(max(1, steps_per_epoch))
    samples_seen_epoch = min(epoch_items, iter_in_epoch * samples_per_update)
    pct = 100.0 * (step + 1) / float(max(1, cfg.steps))
    ran_updates = max(1, step - start_step + 1)
    updates_per_s = ran_updates / max(elapsed, 1e-9)
    samples_per_s = updates_per_s * samples_per_update
    epochs_per_h = updates_per_s / float(max(1, steps_per_epoch)) * 3600.0
    s_per_epoch = float(max(1, steps_per_epoch)) / max(updates_per_s, 1e-9)
    remaining_updates = max(0, cfg.steps - step - 1)
    eta_s = remaining_updates / max(updates_per_s, 1e-9)
    cos_color = "green" if cos_pct >= 50.0 else ("yellow" if cos_pct >= 0.0 else "red")
    print()
    print(
        f"{ansi.c('cyan', 'update')} {ansi.c('bold', f'{step + 1}/{cfg.steps}')} "
        f"{ansi.c('dim', f'({pct:5.1f}%)')}  "
        f"{ansi.c('cyan', 'epoch')} {ansi.c('bold', f'{epoch_index}/{resolved_epochs}')} "
        f"{ansi.c('dim', f'({epoch_pct:5.1f}%)')}  "
        f"{ansi.c('cyan', 'iter')} {ansi.c('bold', f'{iter_in_epoch}/{steps_per_epoch}')}  "
        f"{ansi.c('cyan', 'seen')} {ansi.c('bold', f'{samples_seen_epoch}/{epoch_items}')}  "
        f"{ansi.c('cyan', 'time')} {ansi.c('bold', f'{ms_per_step:.1f} ms/update')}  "
        f"{ansi.c('cyan', 'lr')} {ansi.c('bold', f'{lr_log:.2e}')}"
    )
    print(
        f"  {ansi.c('blue', 'losses')}  "
        f"total {ansi.c('bold', f'{loss_value:.5f}')}  "
        f"L1 {float(metrics['lt'].float().item()):.4f}"
    )
    print(
        f"  {ansi.c('blue', 'spectral')} "
        f"stft {float(metrics['ls'].float().item()):.4f}  "
        f"sgrad {float(metrics['lsg'].float().item()):.4f}  "
        f"stft_cos {float(metrics['lsc'].float().item()):.4f}  "
        f"sc {float(metrics['l_sc'].float().item()):.4f}  "
        f"cx {float(metrics['l_cx'].float().item()):.4f}  "
        f"mel {float(metrics['lm1'].float().item()):.4f}  "
        f"lin {float(metrics['l_lin'].float().item()):.4f}"
    )
    ema = f"  ema {ema_cos_pct:.1f}%" if ema_cos_pct is not None else ""
    print(
        f"  {ansi.c('green', 'signal')}  "
        f"cos {ansi.c(cos_color, f'{cos_pct:.1f}%')}{ema}  "
        f"grad {grad_norm:.3f}"
    )
    print(
        f"  {ansi.c('magenta', 'vq')}      "
        f"loss {float(metrics['vq_l'].float().item()):.4g}  "
        f"marg_ent {float(metrics['marg_ent'].float().item()):.4f}  "
        f"{util}"
    )
    print(
        f"  {ansi.c('yellow', 'weights')} "
        f"stft {effective_lambda_stft(step, cfg):.4f}  "
        f"vq {cfg.lambda_vq * float(metrics.get('phase_vq_mult', 1.0)):.4f}  "
        f"marginal {effective_lambda_marginal(step, cfg) * float(metrics.get('phase_marginal_mult', 1.0)):.4f}  "
        f"phase {metrics.get('phase', 'off')}  "
        f"bal {cfg.loss_balancer}",
    )
    print(
        f"  {ansi.c('cyan', 'throughput')} "
        f"{updates_per_s:.2f} upd/s  "
        f"{samples_per_s:.0f} samples/s  "
        f"{epochs_per_h:.2f} epoch/h  "
        f"{_fmt_duration(s_per_epoch)}/epoch  "
        f"ETA {_fmt_duration(eta_s)}",
        flush=True,
    )


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _log_mag_spectrogram_np(x: np.ndarray, sr: int, n_fft: int, hop: int):
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    win = np.hanning(n_fft)
    cols = []
    for start in range(0, max(1, x.size - n_fft + 1), hop):
        frame = x[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        cols.append(np.log10(np.maximum(np.abs(np.fft.rfft(frame * win, n=n_fft)), 1e-10)))
    if not cols:
        return np.zeros((n_fft // 2 + 1, 1))
    return np.stack(cols, axis=1)


def save_spectrogram_png(orig: torch.Tensor, recon: torch.Tensor, sample_rate: int, out_path: Path, step: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    o = orig.detach().float().cpu().numpy().reshape(-1)
    r = recon.detach().float().cpu().numpy().reshape(-1)
    n_fft, hop = 1024, 256
    so = _log_mag_spectrogram_np(o, sample_rate, n_fft, hop)
    sr = _log_mag_spectrogram_np(r, sample_rate, n_fft, hop)
    vmin = min(float(np.percentile(so, 3)), float(np.percentile(sr, 3)))
    vmax = max(float(np.percentile(so, 99)), float(np.percentile(sr, 99)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axs[0].imshow(so, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axs[0].set_title("orig")
    axs[1].imshow(sr, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axs[1].set_title("recon")
    fig.suptitle(f"CUDA codec step {step} sr={sample_rate}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def save_reconstruction_wavs(orig: torch.Tensor, recon: torch.Tensor, sample_rate: int, stem: Path) -> bool:
    try:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix, x in (("_orig.wav", orig), ("_recon.wav", recon)):
            a = x.detach().float().cpu().numpy().reshape(-1)
            a = np.clip(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            pcm = np.round(a * 32767.0).astype(np.int16)
            with wave.open(str(stem) + suffix, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sample_rate))
                wf.writeframes(pcm.tobytes())
        return True
    except Exception:
        return False


def _find_latest_checkpoint(ck_dir: Path) -> Path | None:
    latest_alias = ck_dir / "latest.pt"
    if latest_alias.is_file():
        return latest_alias
    pts = list(ck_dir.glob("codec_step*.pt"))
    if not pts:
        return None
    return max(pts, key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or "0"))


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _checkpoint_run_dir(ck_path: Path, ck: dict | None = None) -> Path:
    if ck is not None:
        run_meta = ck.get("run")
        if isinstance(run_meta, dict) and run_meta.get("run_dir"):
            return Path(run_meta["run_dir"]).expanduser().resolve()
        if ck.get("run_dir"):
            return Path(ck["run_dir"]).expanduser().resolve()
    if ck_path.parent.name == "checkpoints":
        return ck_path.parent.parent.resolve()
    return ck_path.parent.resolve()


def _resolve_continue_checkpoint(value: str) -> Path:
    p = Path(value).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        state_path = p / "run_state.json"
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                latest_rel = state.get("latest_checkpoint")
                if latest_rel:
                    latest = (p / str(latest_rel)).resolve()
                    if latest.is_file():
                        return latest
            except Exception:
                pass
        ck_dir = p / "checkpoints" if (p / "checkpoints").is_dir() else p
        latest = _find_latest_checkpoint(ck_dir)
        if latest is not None:
            return latest.resolve()
    raise SystemExit(f"--continue checkpoint not found: {value}")


def _prepare_run_layout(cfg: Config, *, continue_ckpt: Path | None) -> dict[str, Path]:
    if continue_ckpt is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path("mlx_runs")
        run_dir = (base_dir / stamp).resolve()
        if run_dir.exists():
            idx = 2
            while True:
                cand = (base_dir / f"{stamp}_{idx:02d}").resolve()
                if not cand.exists():
                    run_dir = cand
                    break
                idx += 1
    else:
        run_dir = _checkpoint_run_dir(continue_ckpt).resolve()
    layout = {
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "inference": run_dir / "inference",
        "log_tsv": run_dir / "log_mlx.tsv",
        "results_tsv": run_dir / "results.tsv",
        "meta_json": run_dir / "train_config.json",
        "state_json": run_dir / "run_state.json",
    }
    layout["checkpoints"].mkdir(parents=True, exist_ok=True)
    layout["inference"].mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir = str(layout["checkpoints"])
    cfg.spectrogram_dir = str(layout["inference"])
    cfg.log_mlx_tsv = str(layout["log_tsv"])
    cfg.results_tsv_path = str(layout["results_tsv"])
    return layout


def _write_run_metadata(
    *,
    layout: dict[str, Path],
    cfg: Config,
    args,
    total_steps: int,
    steps_per_epoch: int,
    resolved_epochs: int,
    epoch_source: str,
    audio_paths: list[Path] | None,
    continue_ckpt: Path | None,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "run_dir": str(layout["run_dir"]),
        "paths": {
            "checkpoints": str(layout["checkpoints"]),
            "inference": str(layout["inference"]),
            "log_mlx_tsv": str(layout["log_tsv"]),
            "results_tsv": str(layout["results_tsv"]),
        },
        "resume_from": None if continue_ckpt is None else str(continue_ckpt),
        "args": _jsonable(vars(args)),
        "config": _jsonable(copy.deepcopy(cfg.__dict__)),
        "training": {
            "total_steps": int(total_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "resolved_epochs": int(resolved_epochs),
            "epoch_source": epoch_source,
            "dataset_items": None if audio_paths is None else int(len(audio_paths)),
        },
    }
    layout["meta_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_state(
    *,
    layout: dict[str, Path],
    latest_checkpoint: Path | None,
    step: int,
    total_steps: int,
    steps_per_epoch: int,
    data_off: int,
    resumed: bool,
) -> None:
    latest_rel = None if latest_checkpoint is None else str(latest_checkpoint.resolve().relative_to(layout["run_dir"]))
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(layout["run_dir"]),
        "latest_checkpoint": latest_rel,
        "step": int(step),
        "next_step": int(step + 1),
        "total_steps": int(total_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "epoch": float((step + 1) / float(max(1, steps_per_epoch))) if step >= 0 else 0.0,
        "data_off": int(data_off),
        "resumed": bool(resumed),
    }
    layout["state_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_lr_scheduler(opt: torch.optim.Optimizer, cfg: Config, start_step: int):
    if (cfg.lr_schedule or "").lower() in ("none", "constant", "off", ""):
        return None

    def lr_lambda(step: int) -> float:
        if cfg.lr_warmup_steps > 0 and step < cfg.lr_warmup_steps:
            return max(1e-8, float(step + 1) / float(cfg.lr_warmup_steps))
        span = max(1, cfg.steps - cfg.lr_warmup_steps)
        t = min(1.0, max(0.0, float(step - cfg.lr_warmup_steps) / float(span)))
        floor = float(cfg.lr_min_ratio)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    sched.last_epoch = max(-1, int(start_step) - 1)
    return sched


def _mean_metric(rows: list[dict[str, float | None]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return float(sum(float(v) for v in vals) / len(vals))


@torch.no_grad()
def run_quality_eval(
    model: CUDACodec,
    cfg: Config,
    audio_paths: list[Path] | None,
    step: int,
    device: torch.device,
    cb_sizes: tuple[int, ...],
) -> tuple[dict[str, float | None], list[float], float | None]:
    n_samples = int(cfg.segment) if cfg.eval_seconds <= 0 else max(int(cfg.eval_seconds * cfg.sample_rate), int(cfg.segment))
    rows: list[dict[str, float | None]] = []
    entropies: list[float] = []
    for i in range(max(1, int(cfg.eval_clips))):
        key = step + i * 104729
        batch_eval = (
            load_audio_viz_clip(cfg, audio_paths, key, n_samples, device)
            if audio_paths is not None
            else synth_viz_clip(cfg, key + cfg.seed * 10007, n_samples, device)
        )
        if cfg.use_bf16 and device.type == "cuda":
            batch_eval = batch_eval.to(torch.bfloat16)
        y_ev, _, _, _, idx_ev = model.forward_full(batch_eval)
        o = batch_eval[0, :, 0].detach().float().cpu().numpy()
        r = y_ev[0, :, 0].detach().float().cpu().numpy()
        rows.append(eval_metrics_mod.quality_metrics_16k(o, r))
        if idx_ev is not None:
            stage_h = []
            for si, ix in enumerate(idx_ev):
                kk = cb_sizes[si] if si < len(cb_sizes) else cb_sizes[-1]
                stage_h.append(entropy_coding_mod.empirical_cross_entropy_bits_per_symbol(ix.detach().cpu().reshape(-1).tolist(), kk))
            if stage_h:
                entropies.append(sum(stage_h) / len(stage_h))
    mean_h = sum(entropies) / len(entropies) if entropies else None
    idx_bps = None
    if mean_h is not None:
        idx_bps = mean_h * (cfg.sample_rate / float(encoder_time_stride(cfg))) * float(len(cb_sizes))
    return {
        "si_sdr_db": _mean_metric(rows, "si_sdr_db"),
        "pesq_wb": _mean_metric(rows, "pesq_wb"),
        "stoi": _mean_metric(rows, "stoi"),
        "lsd_db": _mean_metric(rows, "lsd_db"),
        "l1": _mean_metric(rows, "l1"),
        "cos": _mean_metric(rows, "cos"),
    }, entropies, idx_bps


def resolve_training_steps(cfg: Config, args, audio_paths: list[Path] | None) -> tuple[int, int, int, str]:
    """Return (total_steps, steps_per_epoch, epochs, source_label)."""
    acm = max(1, int(cfg.grad_accum_steps))
    if audio_paths is not None:
        samples_per_step = max(1, int(cfg.batch) * acm)
        steps_per_epoch = max(1, math.ceil(len(audio_paths) / samples_per_step))
        source = f"{len(audio_paths)} files / ({cfg.batch} batch * {acm} accum)"
    else:
        steps_per_epoch = max(1, int(args.synthetic_steps_per_epoch))
        source = "synthetic steps-per-epoch"

    if args.steps is not None:
        total_steps = int(args.steps)
        epochs = max(1, math.ceil(total_steps / float(steps_per_epoch)))
        return total_steps, steps_per_epoch, epochs, "--steps override"

    epochs = int(args.epochs)
    if epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    total_steps = epochs * steps_per_epoch
    return total_steps, steps_per_epoch, epochs, source


@torch.no_grad()
def vq_reset_dead_codes(model: CUDACodec, batch: torch.Tensor, cfg: Config) -> tuple[int, list[int]]:
    if cfg.ae_only:
        return 0, []
    z = model.latent_before_rvq(batch)
    quantized = torch.zeros_like(z)
    total = 0
    per_stage: list[int] = []
    for stage in model.rvq.stages:
        assert isinstance(stage, VectorQuantizerStage)
        residual = z - quantized
        r = stage.project_residual(residual)
        _, _, _, idx = stage(residual)
        k = stage.num_embeddings
        n_unique = int(torch.unique(idx).numel())
        if n_unique / max(1, k) >= float(cfg.vq_reset_collapse_frac):
            per_stage.append(0)
            z_i, _, _, _ = stage(residual)
            quantized = quantized + z_i
            continue
        used = torch.zeros(k, dtype=torch.bool, device=idx.device)
        used[idx.reshape(-1)] = True
        dead = torch.nonzero(~used, as_tuple=False).flatten()
        if n_unique <= int(cfg.vq_reset_full_refresh_max_unique):
            dead = torch.arange(k, device=idx.device)
        if dead.numel() > 0:
            samples = r.reshape(-1, r.shape[-1])
            pick = torch.randint(0, samples.shape[0], (dead.numel(),), device=idx.device)
            rows = samples[pick]
            noise = float(cfg.vq_reset_noise) * rows.std().clamp_min(1e-5) * torch.randn_like(rows)
            stage.embedding.weight.data[dead] = rows + noise
            total += int(dead.numel())
            per_stage.append(int(dead.numel()))
        else:
            per_stage.append(0)
        z_i, _, _, _ = stage(residual)
        quantized = quantized + z_i
    return total, per_stage


@torch.no_grad()
def update_vq_ema_codebooks(model: CUDACodec, batch: torch.Tensor, cfg: Config) -> None:
    dec = float(cfg.vq_ema_decay)
    if not (0.0 < dec < 1.0) or cfg.ae_only:
        return
    beta = 1.0 - dec
    z = model.latent_before_rvq(batch)
    quantized = torch.zeros_like(z)
    for stage in model.rvq.stages:
        assert isinstance(stage, VectorQuantizerStage)
        residual = z - quantized
        r = stage.project_residual(residual)
        z_i, _, _, idx = stage(residual)
        flat_idx = idx.reshape(-1)
        flat_r = r.reshape(-1, r.shape[-1])
        sums = torch.zeros((stage.num_embeddings, flat_r.shape[-1]), device=flat_r.device, dtype=flat_r.dtype)
        counts = torch.zeros((stage.num_embeddings, 1), device=flat_r.device, dtype=flat_r.dtype)
        sums.index_add_(0, flat_idx, flat_r)
        counts.index_add_(0, flat_idx, torch.ones((flat_idx.numel(), 1), device=flat_r.device, dtype=flat_r.dtype))
        used = counts.squeeze(1) > 0
        if torch.any(used):
            means = sums[used] / counts[used].clamp_min(1.0)
            stage.embedding.weight.data[used] = dec * stage.embedding.weight.data[used] + beta * means
        quantized = quantized + z_i


def _add_bool(p: argparse.ArgumentParser, name: str, default: bool, help: str = "") -> None:
    p.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help)


def parse_args(argv: list[str] | None = None):
    c = Config()
    p = argparse.ArgumentParser(description="CUDA Karpathy-style audio codec trainer (former train_mlx CLI)")
    p.add_argument("--epochs", type=int, default=1, help="Full passes over the corpus; steps = ceil(files/(batch*accum))*epochs")
    p.add_argument("--steps", type=int, default=None, help="Deprecated compatibility override: train exact optimizer steps")
    p.add_argument("--synthetic-steps-per-epoch", type=int, default=1000, help="Epoch size when using --no-librispeech synthetic batches")
    p.add_argument("--batch", type=int, default=c.batch)
    p.add_argument("--grad-accum-steps", type=int, default=c.grad_accum_steps)
    p.add_argument("--load-audio-threads", type=int, default=c.load_audio_threads)
    _add_bool(p, "--prefetch-audio", c.prefetch_audio)
    p.add_argument("--prefetch-audio-batches", type=int, default=c.prefetch_audio_batches, help="CPU audio batches queued ahead of GPU training")
    p.add_argument("--segment", type=int, default=c.segment)
    p.add_argument("--lr", type=float, default=c.lr)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default=c.lr_schedule)
    p.add_argument("--lr-min-ratio", type=float, default=c.lr_min_ratio)
    p.add_argument("--lr-warmup-steps", type=int, default=c.lr_warmup_steps)
    p.add_argument("--grad-clip", type=float, default=c.grad_clip_norm)
    p.add_argument("--seed", type=int, default=c.seed)
    p.add_argument("--ae-only", action="store_true", default=c.ae_only)
    p.add_argument("--enc-channels", type=str, default=",".join(str(x) for x in c.enc_channels))
    p.add_argument("--stride1-blocks-per-scale", type=int, default=c.stride1_blocks_per_scale)
    p.add_argument("--latent-dim", type=int, default=c.latent_dim)
    _add_bool(p, "--pre-vq-layernorm", c.pre_vq_layernorm)
    p.add_argument("--latent-temporal-depth", type=int, default=c.latent_temporal_depth)
    p.add_argument("--latent-temporal-post-depth", type=int, default=c.latent_temporal_post_depth)
    p.add_argument("--n-codebooks", type=int, default=c.n_codebooks)
    p.add_argument("--codebook-size", type=int, default=c.codebook_size)
    p.add_argument("--codebook-sizes", type=str, default=None)
    p.add_argument("--lambda-time", type=float, default=c.lambda_time)
    p.add_argument("--lambda-stft", type=float, default=c.lambda_stft)
    p.add_argument("--lambda-stft-grad", type=float, default=c.lambda_stft_grad)
    p.add_argument("--lambda-stft-cos", type=float, default=c.lambda_stft_cos)
    p.add_argument("--stft-grad-freq-weight", type=float, default=c.stft_grad_freq_weight)
    p.add_argument("--stft-grad-time-weight", type=float, default=c.stft_grad_time_weight)
    p.add_argument("--stft-ramp-steps", type=int, default=c.stft_ramp_steps)
    p.add_argument("--stft-ramp-start", type=float, default=c.stft_ramp_start_frac)
    p.add_argument("--stft-scales", type=str, default=";".join(f"{n},{h}" for n, h in c.stft_scales))
    p.add_argument("--stft-scale-weights", type=str, default=None)
    p.add_argument("--stft-hf-emphasis", type=float, default=c.stft_hf_emphasis)
    p.add_argument("--fast", action="store_true", help="Use 3-scale smaller STFT preset: 512/1024/2048")
    p.add_argument("--lambda-mag-l1", type=float, default=c.lambda_mag_l1)
    p.add_argument("--lambda-sc", type=float, default=c.lambda_sc)
    p.add_argument("--lambda-complex-stft", type=float, default=c.lambda_complex_stft)
    p.add_argument("--mel-n-fft", type=int, default=c.mel_n_fft)
    p.add_argument("--mel-hop", type=int, default=c.mel_hop)
    p.add_argument("--n-mels", type=int, default=c.n_mels)
    p.add_argument("--mel-fmin", type=float, default=c.mel_fmin)
    p.add_argument("--mel-fmax", type=float, default=c.mel_fmax)
    p.add_argument("--lambda-mel-l1", type=float, default=c.lambda_mel_l1)
    p.add_argument("--lambda-mel-l2", type=float, default=c.lambda_mel_l2)
    p.add_argument("--lambda-vq", type=float, default=c.lambda_vq)
    p.add_argument("--lambda-entropy", type=float, default=c.lambda_entropy)
    p.add_argument("--lambda-marginal", type=float, default=c.lambda_marginal)
    p.add_argument("--marginal-tau", type=float, default=c.marginal_tau)
    p.add_argument("--marginal-boost-steps", type=int, default=c.marginal_boost_steps)
    p.add_argument("--marginal-boost-mult", type=float, default=c.marginal_boost_mult)
    p.add_argument("--vq-reset-every", type=int, default=c.vq_reset_every)
    p.add_argument("--vq-reset-collapse-frac", type=float, default=c.vq_reset_collapse_frac)
    p.add_argument("--vq-reset-noise", type=float, default=c.vq_reset_noise)
    _add_bool(p, "--vq-reset-shuffle", c.vq_reset_shuffle)
    _add_bool(p, "--vq-reset-kmeans", c.vq_reset_kmeans)
    p.add_argument("--vq-reset-full-refresh-max-unique", type=int, default=c.vq_reset_full_refresh_max_unique)
    p.add_argument("--vq-reset-log-every", type=int, default=c.vq_reset_log_every)
    _add_bool(p, "--vq-cosine", c.vq_cosine)
    p.add_argument("--vq-beta", type=float, default=c.vq_commitment)
    p.add_argument("--lambda-cos", type=float, default=c.lambda_cos)
    p.add_argument("--cos-hinge", type=float, default=c.cos_hinge)
    p.add_argument("--cos-target", type=float, default=c.cos_target)
    p.add_argument("--loss-balancer", choices=["off", "grad"], default=c.loss_balancer, help="Balance reconstruction losses by gradient ratio wrt y_hat")
    p.add_argument("--loss-balancer-eps", type=float, default=c.loss_balancer_eps)
    p.add_argument("--loss-balancer-max-scale", type=float, default=c.loss_balancer_max_scale)
    _add_bool(p, "--curriculum", c.curriculum, "A=AE warmup, B=RVQ ramp, D=full fine-tune; no GAN phase")
    p.add_argument("--curriculum-ae-frac", type=float, default=c.curriculum_ae_frac)
    p.add_argument("--curriculum-vq-ramp-frac", type=float, default=c.curriculum_vq_ramp_frac)
    p.add_argument("--curriculum-vq-start", type=float, default=c.curriculum_vq_start)
    p.add_argument("--curriculum-entropy-start", type=float, default=c.curriculum_entropy_start)
    p.add_argument("--activation", choices=["gelu", "snake", "snake_beta"], default=c.activation)
    p.add_argument("--rvq-code-dim", type=int, default=c.rvq_code_dim)
    p.add_argument("--vq-ema-decay", type=float, default=c.vq_ema_decay)
    p.add_argument("--decoder-upsample", choices=["transpose", "repeat_conv"], default=c.decoder_upsample)
    p.add_argument("--causal", action="store_true", default=c.causal)
    _add_bool(p, "--bf16", c.use_bf16)
    p.add_argument("--compile-loss", action=argparse.BooleanOptionalAction, default=False, help="Accepted for old MLX CLI; CUDA ignores it")
    p.add_argument("--torch-compile", action="store_true", help="Compile model with torch.compile when available")
    p.add_argument("--full-checkpoint", action=argparse.BooleanOptionalAction, default=c.full_checkpoint)
    p.add_argument("--eval-every", type=int, default=c.eval_every)
    p.add_argument("--eval-clips", type=int, default=c.eval_clips)
    p.add_argument("--eval-seconds", type=float, default=c.eval_seconds)
    p.add_argument("--log-mlx-tsv", type=str, default=c.log_mlx_tsv)
    p.add_argument("--results-tsv", type=str, default=c.results_tsv_path)
    p.add_argument("--adaptive-mode", choices=["none", "bwe_stub", "fps_stub"], default=c.adaptive_mode)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--librispeech", action=argparse.BooleanOptionalAction, default=c.use_librispeech)
    p.add_argument("--logs-per-epoch", type=int, default=20, help="How many progress logs to print per epoch")
    p.add_argument("--log-every", type=int, default=None, help="Deprecated override: exact optimizer-step logging interval")
    p.add_argument("--log-cos-ema-beta", type=float, default=c.log_cos_ema_beta)
    p.add_argument("--viz-per-epoch", type=int, default=1, help="Run reconstruction inference + PNG/WAV export N times per epoch")
    p.add_argument("--spectrogram-every", type=int, default=None, help="Deprecated override: exact update interval for PNG/WAV export; 0 disables")
    p.add_argument("--spectrogram-dir", type=str, default=c.spectrogram_dir)
    p.add_argument("--save-audio", action=argparse.BooleanOptionalAction, default=c.save_audio)
    p.add_argument("--spectrogram-seconds", type=float, default=c.spectrogram_seconds)
    p.add_argument("--checkpoint-every", type=int, default=None, help="Numbered checkpoint interval in optimizer updates; omitted => every 10 epochs")
    p.add_argument("--checkpoint-dir", type=str, default=c.checkpoint_dir)
    p.add_argument("--continue", dest="continue_path", type=str, default=None, help="Resume full training state from checkpoint path or run dir")
    p.add_argument("--resume", nargs="?", const="__latest__", default=None, help=argparse.SUPPRESS)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto", help="ANSI colors in logs")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[warn] ignored old/unknown train_mlx args: {' '.join(unknown)}", file=sys.stderr)
    return args


def config_from_args(args) -> Config:
    if not args.data_dir and args.librispeech:
        args.data_dir = str(Path("data") / "cv-corpus")
    try:
        enc_ch = tuple(int(x.strip()) for x in args.enc_channels.split(",") if x.strip())
    except ValueError:
        raise SystemExit("--enc-channels must be comma-separated integers")
    if len(enc_ch) < 1:
        raise SystemExit("--enc-channels must list at least one width")
    codebook_sizes = parse_codebook_sizes_arg(args.codebook_sizes) if args.codebook_sizes else None
    if codebook_sizes is not None and len(codebook_sizes) != args.n_codebooks:
        raise SystemExit(f"--codebook-sizes: expected {args.n_codebooks} values, got {len(codebook_sizes)}")
    stft_scales = DEFAULT_STFT_SCALES if args.fast else parse_stft_scales_arg(args.stft_scales)
    stft_weights = parse_stft_scale_weights_arg(args.stft_scale_weights) if args.stft_scale_weights else None
    if stft_weights is not None and len(stft_weights) != len(stft_scales):
        raise SystemExit("--stft-scale-weights must have one value per STFT scale")
    return Config(
        steps=int(args.steps) if args.steps is not None else 0,
        batch=args.batch,
        load_audio_threads=args.load_audio_threads,
        prefetch_audio=bool(args.prefetch_audio),
        prefetch_audio_batches=args.prefetch_audio_batches,
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
        codebook_sizes=codebook_sizes,
        lambda_time=args.lambda_time,
        lambda_stft=args.lambda_stft,
        lambda_stft_grad=args.lambda_stft_grad,
        lambda_stft_cos=args.lambda_stft_cos,
        stft_grad_freq_weight=args.stft_grad_freq_weight,
        stft_grad_time_weight=args.stft_grad_time_weight,
        stft_ramp_steps=args.stft_ramp_steps,
        stft_ramp_start_frac=args.stft_ramp_start,
        stft_scales=stft_scales,
        stft_scale_weights=stft_weights,
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
        loss_balancer=args.loss_balancer,
        loss_balancer_eps=args.loss_balancer_eps,
        loss_balancer_max_scale=args.loss_balancer_max_scale,
        curriculum=bool(args.curriculum),
        curriculum_ae_frac=args.curriculum_ae_frac,
        curriculum_vq_ramp_frac=args.curriculum_vq_ramp_frac,
        curriculum_vq_start=args.curriculum_vq_start,
        curriculum_entropy_start=args.curriculum_entropy_start,
        lambda_sc=args.lambda_sc,
        lambda_complex_stft=args.lambda_complex_stft,
        log_every=int(args.log_every) if args.log_every is not None else Config().log_every,
        log_cos_ema_beta=args.log_cos_ema_beta,
        spectrogram_every=int(args.spectrogram_every) if args.spectrogram_every is not None else Config().spectrogram_every,
        spectrogram_dir=args.spectrogram_dir,
        save_audio=args.save_audio,
        spectrogram_seconds=args.spectrogram_seconds,
        checkpoint_every=int(args.checkpoint_every) if args.checkpoint_every is not None else 0,
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
        use_compile=bool(args.torch_compile),
        full_checkpoint=bool(args.full_checkpoint),
        eval_every=args.eval_every,
        eval_clips=args.eval_clips,
        eval_seconds=args.eval_seconds,
        log_mlx_tsv=args.log_mlx_tsv,
        results_tsv_path=args.results_tsv,
        adaptive_mode=args.adaptive_mode,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ansi = _make_ansi(args.color)
    if args.continue_path is not None and args.resume is not None:
        raise SystemExit("use either --continue or legacy --resume, not both")
    if args.resume is not None:
        if args.resume == "__latest__":
            raise SystemExit("legacy --resume without path is no longer supported; use --continue <run_dir> or --continue <checkpoint>")
        args.continue_path = args.resume
        print_event(ansi, "warn", "--resume is deprecated; use --continue <run_dir/checkpoint>", color="yellow")
    continue_ckpt = _resolve_continue_checkpoint(args.continue_path) if args.continue_path is not None else None
    resume_blob: dict | None = None
    resumed_from_checkpoint = continue_ckpt is not None
    if continue_ckpt is not None:
        resume_blob = torch.load(continue_ckpt, map_location="cpu", weights_only=False)
        if "config" not in resume_blob:
            raise SystemExit(f"--continue checkpoint missing config: {continue_ckpt}")
        try:
            cfg = Config(**resume_blob["config"])
        except TypeError as e:
            raise SystemExit(f"--continue incompatible checkpoint config: {e}")
    else:
        cfg = config_from_args(args)
    if (args.steps is not None and args.steps < 0) or cfg.batch < 1 or cfg.grad_accum_steps < 1:
        raise SystemExit("--steps must be >=0 and --batch/--grad-accum-steps must be >=1")
    if args.synthetic_steps_per_epoch < 1:
        raise SystemExit("--synthetic-steps-per-epoch must be >= 1")
    if args.logs_per_epoch < 1:
        raise SystemExit("--logs-per-epoch must be >= 1")
    if args.prefetch_audio_batches < 1:
        raise SystemExit("--prefetch-audio-batches must be >= 1")
    if args.loss_balancer_eps <= 0:
        raise SystemExit("--loss-balancer-eps must be > 0")
    if args.loss_balancer_max_scale <= 0:
        raise SystemExit("--loss-balancer-max-scale must be > 0")
    for name in ("curriculum_ae_frac", "curriculum_vq_ramp_frac", "curriculum_vq_start", "curriculum_entropy_start"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")
    if args.eval_clips < 1:
        raise SystemExit("--eval-clips must be >= 1")
    if args.eval_seconds < 0:
        raise SystemExit("--eval-seconds must be >= 0")
    if args.log_every is not None and args.log_every < 1:
        raise SystemExit("--log-every must be >= 1")
    if args.viz_per_epoch < 0:
        raise SystemExit("--viz-per-epoch must be >= 0")
    if args.spectrogram_every is not None and args.spectrogram_every < 0:
        raise SystemExit("--spectrogram-every must be >= 0")

    audio_paths: list[Path] | None = None
    if cfg.data_dir is not None:
        if args.librispeech and not cfg.data_dir.is_dir():
            raise SystemExit(f"Dataset dir missing: {cfg.data_dir}")
        audio_paths = collect_audio_paths(cfg.data_dir)
        if not audio_paths:
            raise SystemExit(f"No audio files under {cfg.data_dir}")

    if resume_blob is not None:
        total_steps = int(resume_blob.get("total_steps", cfg.steps))
        steps_per_epoch = int(resume_blob.get("steps_per_epoch", 0))
        if steps_per_epoch < 1:
            total_steps, steps_per_epoch, resolved_epochs, epoch_source = resolve_training_steps(cfg, args, audio_paths)
        else:
            resolved_epochs = int(resume_blob.get("resolved_epochs", max(1, math.ceil(total_steps / float(steps_per_epoch)))))
            epoch_source = str(resume_blob.get("epoch_source", "checkpoint"))
        cfg.steps = total_steps
    else:
        total_steps, steps_per_epoch, resolved_epochs, epoch_source = resolve_training_steps(cfg, args, audio_paths)
        cfg.steps = total_steps
        if args.log_every is None:
            cfg.log_every = max(1, math.ceil(steps_per_epoch / int(args.logs_per_epoch)))
        if args.spectrogram_every is None:
            cfg.spectrogram_every = (
                max(1, math.ceil(steps_per_epoch / int(args.viz_per_epoch)))
                if int(args.viz_per_epoch) > 0
                else 0
            )
        if args.checkpoint_every is None:
            cfg.checkpoint_every = max(1, 10 * steps_per_epoch)
    layout = _prepare_run_layout(cfg, continue_ckpt=continue_ckpt)
    _write_run_metadata(
        layout=layout,
        cfg=cfg,
        args=args,
        total_steps=cfg.steps,
        steps_per_epoch=steps_per_epoch,
        resolved_epochs=resolved_epochs,
        epoch_source=epoch_source,
        audio_paths=audio_paths,
        continue_ckpt=continue_ckpt,
    )
    if resume_blob is None:
        _write_run_state(
            layout=layout,
            latest_checkpoint=None,
            step=-1,
            total_steps=cfg.steps,
            steps_per_epoch=steps_per_epoch,
            data_off=0,
            resumed=False,
        )

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = torch.device("cpu")
        print_event(ansi, "warn", "CUDA not available; running on CPU", color="yellow")

    model = CUDACodec(cfg).to(device)
    if cfg.use_bf16 and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    if args.torch_compile and cfg.curriculum:
        print_event(ansi, "warn", "torch.compile disabled with --curriculum because AE/RVQ phase toggles model control flow", color="yellow")
    elif args.torch_compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print_event(ansi, "compile", "torch.compile enabled", color="green")
        except Exception as e:
            print_event(ansi, "warn", f"torch.compile failed: {e}", color="yellow")

    start_step = 0
    data_off = 0
    ck_path: Path | None = None
    if resume_blob is not None:
        ck_path = continue_ckpt
        ck = resume_blob if device.type == "cpu" else torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        start_step = int(ck.get("step", -1)) + 1
        data_off = int(ck.get("data_off", start_step * cfg.batch * cfg.grad_accum_steps))
        print_event(ansi, "resume", f"{ck_path} -> train steps {start_step}..{cfg.steps - 1}", color="cyan")
        _write_run_state(
            layout=layout,
            latest_checkpoint=ck_path,
            step=max(-1, start_step - 1),
            total_steps=cfg.steps,
            steps_per_epoch=steps_per_epoch,
            data_off=data_off,
            resumed=True,
        )
        if start_step >= cfg.steps:
            print_event(ansi, "done", f"checkpoint already reached target: step {start_step - 1} >= total steps {cfg.steps - 1}", color="green")
            return

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = build_lr_scheduler(opt, cfg, start_step)
    if ck_path is not None:
        ck = resume_blob if (resume_blob is not None and device.type == "cpu") else torch.load(ck_path, map_location=device, weights_only=False)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if sched is not None and "scheduler" in ck and ck["scheduler"] is not None:
            sched.load_state_dict(ck["scheduler"])

    mel_fb = None
    if cfg.lambda_mel_l1 > 0 or cfg.lambda_mel_l2 > 0:
        fmax_mel = float(cfg.mel_fmax) if cfg.mel_fmax is not None else float(cfg.sample_rate) / 2.0
        mel_fb = mel_filterbank_torch(cfg.sample_rate, cfg.mel_n_fft, cfg.n_mels, cfg.mel_fmin, fmax_mel, device=device)

    n_params = _count_params(model)
    st = encoder_time_stride(cfg)
    nom_kbps = nominal_rvq_kbps(cfg) * adaptive_mod.nominal_bitrate_multiplier(cfg.adaptive_mode)
    cb_sizes = effective_codebook_sizes(cfg)
    stft_nfo = " ".join(f"nfft{n}/hop{h}" for n, h in cfg.stft_scales)
    print_run_header(
        ansi=ansi,
        device=device,
        device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        cfg=cfg,
        n_params=n_params,
        st=st,
        nom_kbps=nom_kbps,
        cb_sizes=cb_sizes,
        stft_nfo=stft_nfo,
        resolved_epochs=resolved_epochs,
        steps_per_epoch=steps_per_epoch,
        epoch_source=epoch_source,
        audio_paths=audio_paths,
        run_dir=layout["run_dir"],
    )

    prefetch_ex: ThreadPoolExecutor | None = None
    prefetch_futures: deque[Future[torch.Tensor]] = deque()
    row_ex: ThreadPoolExecutor | None = None
    acm = max(1, int(cfg.grad_accum_steps))
    pin_audio_batches = device.type == "cuda" and audio_paths is not None
    if audio_paths is not None and int(cfg.load_audio_threads) > 1 and cfg.batch > 1:
        row_ex = ThreadPoolExecutor(max_workers=min(int(cfg.load_audio_threads), int(cfg.batch), 32))
    prefetch_depth = max(1, int(cfg.prefetch_audio_batches))

    def submit_prefetch() -> None:
        nonlocal data_off
        if prefetch_ex is None or audio_paths is None:
            return
        prefetch_futures.append(
            prefetch_ex.submit(load_audio_batch_cpu, cfg, audio_paths, data_off, row_ex, pin_memory=pin_audio_batches)
        )
        data_off += cfg.batch

    if audio_paths is not None and cfg.prefetch_audio and acm <= 1 and cfg.steps > start_step:
        prefetch_ex = ThreadPoolExecutor(max_workers=1)
        for _ in range(min(prefetch_depth, cfg.steps - start_step)):
            submit_prefetch()

    t0 = time.time()
    profiler = _ProfileStats(args.profile, device)
    if args.profile:
        print_event(ansi, "profile", "enabled; CUDA syncs timing, training will run slower while profiling", color="yellow")
    ema_cos_pct: float | None = None
    last_metrics = None
    last_batch = None
    for step in range(start_step, cfg.steps):
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro in range(acm):
            prof_t = profiler.mark()
            if audio_paths is not None:
                if micro == 0 and prefetch_futures:
                    batch = prefetch_futures.popleft().result().to(device, non_blocking=device.type == "cuda")
                    remaining_to_submit = cfg.steps - start_step - (step - start_step + 1) - len(prefetch_futures)
                    if remaining_to_submit > 0:
                        submit_prefetch()
                else:
                    batch = load_audio_batch(cfg, audio_paths, data_off, device, row_ex, pin_memory=pin_audio_batches)
                    data_off += cfg.batch
            else:
                batch = synth_batch(cfg, key=step * 1_000_003 + micro * 97 + cfg.seed * 10007, device=device)
            prof_t = profiler.add_since("data", prof_t)
            if cfg.use_bf16 and device.type == "cuda":
                batch = batch.to(torch.bfloat16)
            loss, metrics = compute_loss(model, cfg, batch, step, mel_fb)
            (loss / acm).backward()
            profiler.add_since("train", prof_t)
            loss_total += float(loss.detach().float().item())
            last_metrics = metrics
            last_batch = batch

        prof_t = profiler.mark()
        if not math.isfinite(loss_total):
            print_event(ansi, "skip", f"non-finite loss at step {step}", color="yellow")
            opt.zero_grad(set_to_none=True)
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm) if cfg.grad_clip_norm > 0 else torch.tensor(0.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            print_event(ansi, "skip", f"non-finite gradient at step {step}", color="yellow")
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        if sched is not None:
            sched.step()
        profiler.add_since("opt", prof_t)

        prof_t = profiler.mark()
        if 0.0 < float(cfg.vq_ema_decay) < 1.0 and last_batch is not None:
            update_vq_ema_codebooks(model, last_batch, cfg)
        curr_after = curriculum_state(step, cfg)
        if cfg.vq_reset_every > 0 and curr_after.reset_enabled and not cfg.ae_only and step > 0 and step % cfg.vq_reset_every == 0 and last_batch is not None:
            n_reset, per_st = vq_reset_dead_codes(model, last_batch, cfg)
            if n_reset > 0 and (cfg.vq_reset_log_every <= 0 or step % cfg.vq_reset_log_every == 0):
                parts = [f"s{i}={c}" for i, c in enumerate(per_st) if c > 0]
                print_event(ansi, "vq-reset", f"replaced {n_reset} dead codes" + (f" ({', '.join(parts)})" if parts else ""), color="magenta")
        profiler.add_since("aux", prof_t)
        profiler.step_done()

        epoch_boundary = (step + 1) % max(1, steps_per_epoch) == 0
        prof_t = profiler.mark()
        if last_metrics is not None and (
            step % cfg.log_every == 0 or epoch_boundary or step == cfg.steps - 1
        ):
            m = last_metrics
            cos_pct = 100.0 * float(m["cos_m"].float().item())
            if cfg.log_cos_ema_beta > 0:
                ema_cos_pct = cos_pct if ema_cos_pct is None else cfg.log_cos_ema_beta * ema_cos_pct + (1.0 - cfg.log_cos_ema_beta) * cos_pct
            util = _format_vq_util(m["idx"], cb_sizes)
            elapsed = time.time() - t0
            lr_log = opt.param_groups[0]["lr"]
            print_step_log(
                ansi=ansi,
                step=step,
                cfg=cfg,
                steps_per_epoch=steps_per_epoch,
                resolved_epochs=resolved_epochs,
                loss_value=loss_total / acm,
                metrics=m,
                cos_pct=cos_pct,
                ema_cos_pct=ema_cos_pct if cfg.log_cos_ema_beta > 0 else None,
                util=util,
                lr_log=lr_log,
                ms_per_step=elapsed / (step - start_step + 1) * 1000,
                grad_norm=float(torch.as_tensor(grad_norm).detach().float().item()),
                samples_per_update=cfg.batch * cfg.grad_accum_steps,
                epoch_items=len(audio_paths) if audio_paths is not None else steps_per_epoch * cfg.batch * cfg.grad_accum_steps,
                elapsed=elapsed,
                start_step=start_step,
            )
            if args.profile:
                print_event(ansi, "profile", profiler.summary(), color="yellow")

        if cfg.eval_every > 0 and step > 0 and step % int(cfg.eval_every) == 0 and last_batch is not None:
            eval_m, h_vals, idx_bps = run_quality_eval(model, cfg, audio_paths, step, device, cb_sizes)

            def fmt_opt(key: str, fmt: str = ".3f") -> str:
                v = eval_m.get(key)
                return "na" if v is None else format(float(v), fmt)

            print_event(
                ansi,
                "eval",
                (
                    f"clips={cfg.eval_clips}  "
                    f"SI-SDR={fmt_opt('si_sdr_db', '.2f')} dB  "
                    f"PESQ={fmt_opt('pesq_wb')}  "
                    f"STOI={fmt_opt('stoi')}  "
                    f"LSD={fmt_opt('lsd_db', '.2f')} dB  "
                    f"L1={fmt_opt('l1', '.4f')}  "
                    f"cos={100.0 * float(eval_m.get('cos') or 0.0):.1f}%"
                ),
                color="green",
            )
            if cfg.log_mlx_tsv:
                p = Path(cfg.log_mlx_tsv)
                p.parent.mkdir(parents=True, exist_ok=True)
                desired_hdr = "step\tphase\tsisdr_db\tpesq_wb\tstoi\tlsd_db\tl1\tcos\tidx_bps\n"
                hdr = not p.is_file()
                if not hdr:
                    try:
                        hdr = p.read_text(encoding="utf-8").splitlines()[0].strip() != desired_hdr.strip()
                    except Exception:
                        hdr = True
                with p.open("a", encoding="utf-8") as f:
                    if hdr:
                        f.write(desired_hdr)
                    vals = {
                        "sisdr": fmt_opt("si_sdr_db", ".6f"),
                        "pesq": fmt_opt("pesq_wb", ".6f"),
                        "stoi": fmt_opt("stoi", ".6f"),
                        "lsd": fmt_opt("lsd_db", ".6f"),
                        "l1": fmt_opt("l1", ".6f"),
                        "cos": fmt_opt("cos", ".6f"),
                        "bps": "na" if idx_bps is None else f"{idx_bps:.6f}",
                    }
                    f.write(f"{step}\t{curriculum_state(step, cfg).phase}\t{vals['sisdr']}\t{vals['pesq']}\t{vals['stoi']}\t{vals['lsd']}\t{vals['l1']}\t{vals['cos']}\t{vals['bps']}\n")
            if h_vals:
                h_mean = sum(h_vals) / len(h_vals)
                bps_s = "na" if idx_bps is None else f"{idx_bps:.0f}"
                print_event(ansi, "entropy", f"RVQ mean H={h_mean:.3f} b/sym  ~{bps_s} b/s empirical index", color="magenta")

        save_latest = step > 0 and (epoch_boundary or step == cfg.steps - 1)
        save_numbered = cfg.checkpoint_every > 0 and step > 0 and (step % cfg.checkpoint_every == 0 or step == cfg.steps - 1)
        if save_latest or save_numbered:
            ck_dir = Path(cfg.checkpoint_dir)
            ck_dir.mkdir(parents=True, exist_ok=True)
            latest_path = ck_dir / "latest.pt"
            numbered_path = ck_dir / f"codec_step{step}.pt"
            ck_payload = {
                "step": step,
                "epoch": (step + 1) / float(steps_per_epoch),
                "total_steps": cfg.steps,
                "resolved_epochs": resolved_epochs,
                "epoch_source": epoch_source,
                "steps_per_epoch": steps_per_epoch,
                "data_off": data_off,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched is not None else None,
                "config": cfg.__dict__,
                "run": {
                    "run_dir": str(layout["run_dir"]),
                    "checkpoint_dir": str(layout["checkpoints"]),
                    "inference_dir": str(layout["inference"]),
                    "log_mlx_tsv": str(layout["log_tsv"]),
                    "results_tsv": str(layout["results_tsv"]),
                },
            }
            torch.save(ck_payload, latest_path)
            ck_path = latest_path
            if save_numbered:
                torch.save(ck_payload, numbered_path)
                (ck_dir / f"codec_step{step}.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": (step + 1) / float(steps_per_epoch),
                            "total_steps": cfg.steps,
                            "resolved_epochs": resolved_epochs,
                            "steps_per_epoch": steps_per_epoch,
                            "data_off": data_off,
                            "run_dir": str(layout["run_dir"]),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                ck_path = numbered_path
            _write_run_state(
                layout=layout,
                latest_checkpoint=latest_path,
                step=step,
                total_steps=cfg.steps,
                steps_per_epoch=steps_per_epoch,
                data_off=data_off,
                resumed=resumed_from_checkpoint,
            )
            print_event(ansi, "latest", str(latest_path), color="cyan")
            if save_numbered:
                print_event(ansi, "checkpoint", str(numbered_path), color="cyan")
            if save_numbered and cfg.results_tsv_path:
                rp = Path(cfg.results_tsv_path)
                rp.parent.mkdir(parents=True, exist_ok=True)
                need_hdr = not rp.is_file()
                with rp.open("a", encoding="utf-8") as rf:
                    if need_hdr:
                        rf.write("cycle\tphase\thypothesis\tarch_id\tbitrate_bps\tpesq_est\tvisqol_est\tlatency_ms\tparams_M\tverdict\tkey_finding\tnext_action\n")
                    rf.write(f"0\tcuda_train\tplan_arch\tcuda_cfg\t{nom_kbps * 1000.0:.1f}\tna\tna\tna\t{n_params / 1e6:.2f}\ttrain\tstep={step}\tcheckpoint\n")

        viz_due = (
            cfg.spectrogram_every > 0
            and step > 0
            and (
                (((step % max(1, steps_per_epoch)) + 1) % cfg.spectrogram_every == 0)
                or epoch_boundary
            )
        )
        if viz_due:
            with torch.no_grad():
                if cfg.spectrogram_seconds > 0:
                    n_viz = max(int(cfg.spectrogram_seconds * cfg.sample_rate), cfg.segment)
                    batch_viz = load_audio_viz_clip(cfg, audio_paths, step, n_viz, device) if audio_paths is not None else synth_viz_clip(cfg, step + cfg.seed * 10007, n_viz, device)
                else:
                    batch_viz = last_batch
                if batch_viz is not None:
                    if cfg.use_bf16 and device.type == "cuda":
                        batch_viz = batch_viz.to(torch.bfloat16)
                    y_viz, _, _, _, _ = model.forward_full(batch_viz)
                    step_dir = Path(cfg.spectrogram_dir) / f"{step:08d}"
                    out = step_dir / "spectrogram.png"
                    if save_spectrogram_png(batch_viz[0, :, 0], y_viz[0, :, 0], cfg.sample_rate, out, step):
                        print_event(ansi, "spectrogram", str(out), color="cyan")
                    if cfg.save_audio:
                        stem = step_dir / "reconstruction"
                        if save_reconstruction_wavs(batch_viz[0, :, 0], y_viz[0, :, 0], cfg.sample_rate, stem):
                            print_event(ansi, "audio", f"{stem}_orig.wav  {stem}_recon.wav", color="cyan")
        profiler.add_since("misc", prof_t)

    if prefetch_ex is not None:
        prefetch_ex.shutdown(wait=False, cancel_futures=True)
    if row_ex is not None:
        row_ex.shutdown(wait=False, cancel_futures=True)
    ran = cfg.steps - start_step
    total = time.time() - t0
    final_step = cfg.steps - 1 if ran > 0 else start_step - 1
    _write_run_state(
        layout=layout,
        latest_checkpoint=ck_path,
        step=final_step,
        total_steps=cfg.steps,
        steps_per_epoch=steps_per_epoch,
        data_off=data_off,
        resumed=resumed_from_checkpoint,
    )
    if ran > 0:
        print()
        print_event(ansi, "done", f"updates {start_step}..{cfg.steps - 1} ({ran} updates) in {total:.1f}s", color="green")
    else:
        print_event(ansi, "done", "0 updates", color="green")


if __name__ == "__main__":
    main()
