"""Save mel + waveform comparison PNGs during training (headless Agg).

Defaults match ``tools/plot_pipeline_spectrogram.py``: 8s clip from manifest (or synthetic),
so mel time axis width matches reference ``spectrogram_pipeline.png``. Set
``spectrogram_viz_from_batch`` to use the current batch (shorter mel).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from sirencodec.core.train import (
    AudioCodec,
    CodecConfig,
    compute_abr_depth_heuristic,
    get_curriculum_codebooks,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_wave(path: Path, sr: int, seconds: float, device: torch.device) -> torch.Tensor:
    # soundfile avoids torchaudio>=2.9 routing through TorchCodec (optional torchcodec dep)
    import soundfile as sf

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
        wav = F.pad(wav, (0, seg - wav.shape[1]))
    return wav.unsqueeze(0).to(device)


def _synthetic_segment(sr: int, seconds: float, device: torch.device) -> torch.Tensor:
    t = int(sr * seconds)
    x = torch.linspace(0, 1, t, device=device)
    wave = 0.3 * torch.sin(2 * math.pi * (200 + 600 * x) * x * 3)
    wave = wave + 0.15 * torch.sin(2 * math.pi * 440 * x * 3)
    wave = wave + 0.05 * torch.randn(t, device=device)
    return wave.unsqueeze(0).unsqueeze(0)


def _viz_wave_like_plot_pipeline(
    cfg: CodecConfig,
    device: torch.device,
    wave_batch: torch.Tensor,
) -> torch.Tensor:
    """Same audio preparation as ``plot_pipeline_spectrogram`` (manifest or synthetic clip)."""
    if getattr(cfg, "spectrogram_viz_from_batch", False):
        return wave_batch[:1].to(device)

    seconds = float(getattr(cfg, "spectrogram_viz_seconds", 8.0))
    seed = int(getattr(cfg, "spectrogram_viz_seed", 42))
    manifest_rel = getattr(cfg, "spectrogram_viz_manifest", "data/master_manifest.jsonl")
    sr = cfg.sample_rate

    mp = Path(manifest_rel)
    if not mp.is_file():
        mp = Path(cfg.data_dir) / Path(manifest_rel).name
    if not mp.is_file():
        mp = _repo_root() / manifest_rel

    if mp.is_file():
        with open(mp) as f:
            entries = [json.loads(line) for line in f]
        rng = random.Random(seed)
        raw = rng.choice(entries)["path"]
        path = Path(raw)
        if not path.is_file():
            for base in (Path.cwd(), _repo_root()):
                cand = base / raw
                if cand.is_file():
                    path = cand
                    break
        if path.is_file():
            return _load_wave(path, sr, seconds, device)

    return _synthetic_segment(sr, seconds, device)


def _forward_recon(
    codec: torch.nn.Module,
    cfg: CodecConfig,
    wave: torch.Tensor,
    step: int,
    use_vq: bool,
) -> torch.Tensor:
    """Single forward matching train_pipeline (eval / no_grad)."""
    if not use_vq:
        z = codec.encoder(wave)
        x_hat = codec.decoder(z)
        return x_hat[..., : wave.size(-1)]
    n_cb = get_curriculum_codebooks(step, cfg.curriculum_schedule) if cfg.curriculum_enabled else cfg.n_codebooks
    if cfg.architecture == "arch-c-v1":
        x_hat, *_rest = codec(wave, step=step)
        return x_hat
    if cfg.architecture == "arch-d-v1":
        x_hat, *_rest = codec(wave)
        return x_hat
    if cfg.architecture in ("arch-a-spk", "arch-b-v1"):
        x_hat, *_rest = codec(wave)
        return x_hat
    out = codec(wave, n_codebooks=n_cb)
    return out[0]


def save_spectrogram_comparison_png(
    codec: torch.nn.Module,
    cfg: CodecConfig,
    wave_batch: torch.Tensor,
    step: int,
    out_dir: Path,
    device: torch.device,
    use_vq: bool,
    *,
    stage_label: str = "",
    no_depth_plot: bool = False,
) -> Path | None:
    """Write one PNG: waveforms + log-mel orig vs recon; optional ABR depth row for AudioCodec."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step_{step:08d}.png"

    wave = _viz_wave_like_plot_pipeline(cfg, device, wave_batch)
    was_training = codec.training
    codec.eval()
    try:
        with torch.no_grad():
            recon = _forward_recon(codec, cfg, wave, step, use_vq)
    finally:
        if was_training:
            codec.train()

    orig = wave.squeeze(0).cpu()
    deg = recon.squeeze(0).cpu()
    ml = min(orig.shape[-1], deg.shape[-1])
    orig = orig[..., :ml]
    deg = deg[..., :ml]

    sr = cfg.sample_rate
    dur_sec = float(ml) / float(sr)
    fig_w = min(24.0, max(14.0, 10.0 + dur_sec * 1.2))
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=50,
        f_max=sr // 2 - 1,
    )
    mo = mel_tf(orig)
    md = mel_tf(deg)
    mel_o = torch.log(mo.squeeze().clamp(min=1e-5)).numpy()
    mel_d = torch.log(md.squeeze().clamp(min=1e-5)).numpy()
    mel_vmin = float(min(mel_o.min(), mel_d.min()))
    mel_vmax = float(max(mel_o.max(), mel_d.max()))

    def peak_norm(w: torch.Tensor) -> torch.Tensor:
        p = w.abs().max().clamp(min=1e-8)
        return w / p

    ow = peak_norm(orig.squeeze()).numpy()
    dw = peak_norm(deg.squeeze()).numpy()

    depth_latent: np.ndarray | None = None
    if (
        not no_depth_plot
        and getattr(cfg, "abr_enabled", False)
        and isinstance(codec, AudioCodec)
    ):
        with torch.no_grad():
            z = codec.encoder(wave)
            if getattr(cfg, "abr_mode", "full") == "learned" and getattr(codec, "depth_policy", None) is not None:
                d, _, _ = codec.depth_policy(z, cfg, hard=True)
                depth_latent = d.float().cpu().numpy()[0]
            elif cfg.abr_mode == "heuristic":
                depth_latent = compute_abr_depth_heuristic(z, cfg).float().cpu().numpy()[0]
            else:
                depth_latent = np.full(z.size(2), float(cfg.n_codebooks), dtype=np.float32)

    if depth_latent is not None:
        fig = plt.figure(figsize=(fig_w, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 0.55], hspace=0.35, wspace=0.25)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        axd = fig.add_subplot(gs[2, :])
        axes = np.array([[ax00, ax01], [ax10, ax11]])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(fig_w, 8))
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

    title = f"step={step}"
    if stage_label:
        title += f" | {stage_label}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
