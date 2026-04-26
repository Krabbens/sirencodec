"""PyTorch STFT, mel, and entropy losses for CUDA training."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


def _as_bt(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x[..., 0]
    return x


@lru_cache(maxsize=64)
def _hann_cached(n: int, device_type: str, device_index: int | None, dtype_name: str) -> torch.Tensor:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    dtype = getattr(torch, dtype_name)
    return torch.hann_window(n, periodic=False, device=device, dtype=dtype)


def _window(n: int, x: torch.Tensor) -> torch.Tensor:
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    idx = x.device.index if x.device.type == "cuda" else None
    return _hann_cached(int(n), x.device.type, idx, str(dtype).split(".")[-1])


def stft_complex(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    x = _as_bt(x)
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if x.shape[1] < n_fft:
        x = F.pad(x, (0, n_fft - x.shape[1]))
    return torch.stft(
        x,
        n_fft=int(n_fft),
        hop_length=int(hop),
        win_length=int(n_fft),
        window=_window(int(n_fft), x),
        center=False,
        return_complex=True,
    )


def _stft_pair(pred: torch.Tensor, tgt: torch.Tensor, n_fft: int, hop: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = _as_bt(pred)
    q = _as_bt(tgt)
    b = p.shape[0]
    spec = stft_complex(torch.cat([p, q], dim=0), n_fft, hop)
    return spec[:b], spec[b:]


def _weighted_mean_abs(diff: torch.Tensor, hf_gamma: float) -> torch.Tensor:
    g = float(hf_gamma)
    if g <= 0 or diff.shape[1] <= 1:
        return diff.abs().mean()
    f = torch.linspace(0.0, 1.0, diff.shape[1], device=diff.device, dtype=diff.dtype)
    w1 = 1.0 + g * (f * f)
    w = w1.view(1, -1, 1)
    return (diff.abs() * w).sum() / (w1.sum() * diff.shape[0] * diff.shape[2])


def _weighted_mean_positive(value: torch.Tensor, hf_gamma: float) -> torch.Tensor:
    g = float(hf_gamma)
    if g <= 0 or value.shape[1] <= 1:
        return value.mean()
    f = torch.linspace(0.0, 1.0, value.shape[1], device=value.device, dtype=value.dtype)
    w1 = 1.0 + g * (f * f)
    w = w1.view(1, -1, 1)
    return (value * w).sum() / (w1.sum() * value.shape[0] * value.shape[2])


def _scale_weights(scales: tuple[tuple[int, int], ...], weights: tuple[float, ...] | None) -> list[float]:
    ws = [1.0] * len(scales) if weights is None else [float(w) for w in weights]
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    if sum(ws) <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    return ws


def multi_stft_loss(pred: torch.Tensor, tgt: torch.Tensor, scales: tuple[tuple[int, int], ...], *, hf_emphasis: float = 0.0, scale_weights: tuple[float, ...] | None = None) -> torch.Tensor:
    if not scales:
        return pred.new_zeros(())
    ws = _scale_weights(scales, scale_weights)
    total = pred.new_zeros((), dtype=torch.float32)
    for (n_fft, hop), w in zip(scales, ws):
        sp, sq = _stft_pair(pred, tgt, n_fft, hop)
        lp = torch.log(sp.abs() + 1e-5)
        lq = torch.log(sq.abs() + 1e-5)
        total = total + float(w) * _weighted_mean_abs(lp - lq, hf_emphasis)
    return total / float(sum(ws))


def stft_gradient_l1_from_log_mag(lp: torch.Tensor, lq: torch.Tensor, *, freq_weight: float, time_weight: float) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    weights: list[float] = []
    wf = max(0.0, float(freq_weight))
    wt = max(0.0, float(time_weight))
    if lp.shape[1] > 1 and wf > 0:
        terms.append(wf * (lp[:, 1:, :] - lp[:, :-1, :] - (lq[:, 1:, :] - lq[:, :-1, :])).abs().mean())
        weights.append(wf)
    if lp.shape[2] > 1 and wt > 0:
        terms.append(wt * (lp[:, :, 1:] - lp[:, :, :-1] - (lq[:, :, 1:] - lq[:, :, :-1])).abs().mean())
        weights.append(wt)
    if not terms:
        return lp.new_zeros(())
    return torch.stack(terms).sum() / float(sum(weights))


def stft_logmag_cosine_1m(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    p = lp.reshape(lp.shape[0], -1)
    q = lq.reshape(lq.shape[0], -1)
    return (1.0 - F.cosine_similarity(p, q, dim=1, eps=1e-8)).mean()


def multi_stft_spectral_terms(pred: torch.Tensor, tgt: torch.Tensor, scales: tuple[tuple[int, int], ...], *, with_grad: bool, with_cos_1m: bool, grad_freq_weight: float = 1.0, grad_time_weight: float = 1.0, hf_emphasis: float = 0.0, scale_weights: tuple[float, ...] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not scales:
        z = pred.new_zeros(())
        return z, z, z
    ws = _scale_weights(scales, scale_weights)
    total_mag = pred.new_zeros((), dtype=torch.float32)
    total_grad = pred.new_zeros((), dtype=torch.float32)
    total_cos = pred.new_zeros((), dtype=torch.float32)
    for (n_fft, hop), w in zip(scales, ws):
        sp, sq = _stft_pair(pred, tgt, n_fft, hop)
        lp = torch.log(sp.abs() + 1e-5)
        lq = torch.log(sq.abs() + 1e-5)
        total_mag = total_mag + float(w) * _weighted_mean_abs(lp - lq, hf_emphasis)
        if with_grad:
            total_grad = total_grad + float(w) * stft_gradient_l1_from_log_mag(
                lp,
                lq,
                freq_weight=grad_freq_weight,
                time_weight=grad_time_weight,
            )
        if with_cos_1m:
            total_cos = total_cos + float(w) * stft_logmag_cosine_1m(lp, lq)
    den = float(sum(ws))
    return total_mag / den, total_grad / den, total_cos / den


def multi_stft_all_terms(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    scales: tuple[tuple[int, int], ...],
    *,
    with_grad: bool,
    with_cos_1m: bool,
    with_linear: bool,
    with_sc: bool,
    with_complex: bool,
    with_excess: bool = False,
    grad_freq_weight: float = 1.0,
    grad_time_weight: float = 1.0,
    hf_emphasis: float = 0.0,
    excess_margin: float = 0.20,
    scale_weights: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all multi-STFT losses from one STFT pair per scale."""
    if not scales:
        z = pred.new_zeros(())
        return z, z, z, z, z, z, z
    ws = _scale_weights(scales, scale_weights)
    total_mag = pred.new_zeros((), dtype=torch.float32)
    total_grad = pred.new_zeros((), dtype=torch.float32)
    total_cos = pred.new_zeros((), dtype=torch.float32)
    total_linear = pred.new_zeros((), dtype=torch.float32)
    total_sc = pred.new_zeros((), dtype=torch.float32)
    total_complex = pred.new_zeros((), dtype=torch.float32)
    total_excess = pred.new_zeros((), dtype=torch.float32)
    for (n_fft, hop), w in zip(scales, ws):
        sp, sq = _stft_pair(pred, tgt, n_fft, hop)
        sp_abs = sp.abs()
        sq_abs = sq.abs()
        lp = torch.log(sp_abs + 1e-5)
        lq = torch.log(sq_abs + 1e-5)
        ww = float(w)
        total_mag = total_mag + ww * _weighted_mean_abs(lp - lq, hf_emphasis)
        if with_grad:
            total_grad = total_grad + ww * stft_gradient_l1_from_log_mag(
                lp,
                lq,
                freq_weight=grad_freq_weight,
                time_weight=grad_time_weight,
            )
        if with_cos_1m:
            total_cos = total_cos + ww * stft_logmag_cosine_1m(lp, lq)
        if with_linear:
            total_linear = total_linear + ww * _weighted_mean_abs(sp_abs - sq_abs, hf_emphasis)
        if with_sc:
            p = sp_abs.reshape(sp_abs.shape[0], -1)
            q = sq_abs.reshape(sq_abs.shape[0], -1)
            sc = torch.linalg.vector_norm(p - q, dim=1) / (torch.linalg.vector_norm(q, dim=1) + 1e-8)
            total_sc = total_sc + ww * sc.mean()
        if with_complex:
            l1 = 0.5 * ((sp.real - sq.real).abs().mean() + (sp.imag - sq.imag).abs().mean())
            total_complex = total_complex + ww * l1
        if with_excess:
            excess = torch.relu(lp - lq - float(excess_margin))
            total_excess = total_excess + ww * _weighted_mean_positive(excess, hf_emphasis)
    den = float(sum(ws))
    return (
        total_mag / den,
        total_grad / den,
        total_cos / den,
        total_linear / den,
        total_sc / den,
        total_complex / den,
        total_excess / den,
    )


def multi_stft_mag_l1_linear(pred: torch.Tensor, tgt: torch.Tensor, scales: tuple[tuple[int, int], ...], *, hf_emphasis: float = 0.0, scale_weights: tuple[float, ...] | None = None) -> torch.Tensor:
    if not scales:
        return pred.new_zeros(())
    ws = _scale_weights(scales, scale_weights)
    total = pred.new_zeros((), dtype=torch.float32)
    for (n_fft, hop), w in zip(scales, ws):
        sp, sq = _stft_pair(pred, tgt, n_fft, hop)
        total = total + float(w) * _weighted_mean_abs(sp.abs() - sq.abs(), hf_emphasis)
    return total / float(sum(ws))


def multi_stft_spectral_convergence(pred: torch.Tensor, tgt: torch.Tensor, scales: tuple[tuple[int, int], ...], *, scale_weights: tuple[float, ...] | None = None) -> torch.Tensor:
    if not scales:
        return pred.new_zeros(())
    ws = _scale_weights(scales, scale_weights)
    total = pred.new_zeros((), dtype=torch.float32)
    for (n_fft, hop), w in zip(scales, ws):
        sp, sq = _stft_pair(pred, tgt, n_fft, hop)
        p = sp.abs().reshape(sp.shape[0], -1)
        q = sq.abs().reshape(sq.shape[0], -1)
        sc = torch.linalg.vector_norm(p - q, dim=1) / (torch.linalg.vector_norm(q, dim=1) + 1e-8)
        total = total + float(w) * sc.mean()
    return total / float(sum(ws))


def multi_stft_complex_l1(pred: torch.Tensor, tgt: torch.Tensor, scales: tuple[tuple[int, int], ...], *, scale_weights: tuple[float, ...] | None = None) -> torch.Tensor:
    if not scales:
        return pred.new_zeros(())
    ws = _scale_weights(scales, scale_weights)
    total = pred.new_zeros((), dtype=torch.float32)
    for (n_fft, hop), w in zip(scales, ws):
        sp, sq = _stft_pair(pred, tgt, n_fft, hop)
        l1 = 0.5 * ((sp.real - sq.real).abs().mean() + (sp.imag - sq.imag).abs().mean())
        total = total + float(w) * l1
    return total / float(sum(ws))


def _mel_filterbank_numpy(*, n_fft: int, n_mels: int, sample_rate: float, fmin: float, fmax: float) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    fmax = min(float(fmax), float(sample_rate) * 0.5)
    fmin = max(0.0, float(fmin))
    if fmax <= fmin:
        raise ValueError(f"mel_fmax ({fmax}) must be > mel_fmin ({fmin})")

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.clip(np.floor((n_fft + 1) * hz_pts / float(sample_rate)).astype(np.int64), 0, n_freqs)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lo, peak, hi = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if peak <= lo:
            peak = lo + 1
        if hi <= peak:
            hi = peak + 1
        peak = min(peak, n_freqs)
        hi = min(hi, n_freqs)
        for k in range(lo, peak):
            fb[i, k] = (k - lo) / float(max(peak - lo, 1))
        for k in range(peak, hi):
            fb[i, k] = (hi - k) / float(max(hi - peak, 1))
    return fb


@lru_cache(maxsize=64)
def _mel_filterbank_cached(sample_rate: int, n_fft: int, n_mels: int, fmin_milli: int, fmax_milli: int) -> np.ndarray:
    return _mel_filterbank_numpy(
        n_fft=n_fft,
        n_mels=n_mels,
        sample_rate=float(sample_rate),
        fmin=fmin_milli / 1000.0,
        fmax=fmax_milli / 1000.0,
    )


def mel_filterbank_torch(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    fb = _mel_filterbank_cached(sample_rate, n_fft, n_mels, int(round(fmin * 1000.0)), int(round(fmax * 1000.0)))
    return torch.as_tensor(fb, device=device, dtype=dtype)


def mel_log_bin_losses(pred: torch.Tensor, tgt: torch.Tensor, mel_fb: torch.Tensor, n_fft: int, hop: int, *, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    sp, sq = _stft_pair(pred, tgt, n_fft, hop)
    mp = sp.abs()
    mq = sq.abs()
    mel_p = torch.matmul(mp.transpose(1, 2), mel_fb.t()).transpose(1, 2)
    mel_q = torch.matmul(mq.transpose(1, 2), mel_fb.t()).transpose(1, 2)
    d = torch.log(mel_p + eps) - torch.log(mel_q + eps)
    return d.abs().mean(), (d * d).mean()


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(torch.clamp(logits, -50.0, 50.0), dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()


def marginal_code_entropy_from_dist(dist: torch.Tensor, tau: float) -> torch.Tensor:
    p = torch.softmax(torch.clamp(-dist / float(tau), -50.0, 50.0), dim=-1)
    avg = p.mean(dim=(0, 1))
    return -(avg * torch.log(avg + 1e-8)).sum()
