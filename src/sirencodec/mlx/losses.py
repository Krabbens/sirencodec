"""STFT / mel / entropy helpers for MLX training losses."""
from __future__ import annotations

import math

import mlx.core as mx

def _hann(n: int) -> mx.array:
    i = mx.arange(n, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2.0 * math.pi * i / float(max(n - 1, 1)))


def stft_complex(x: mx.array, n_fft: int, hop: int) -> mx.array:
    """x: [B, T] real → complex STFT [B, F, n_frames] via one batched ``rfft`` over framed signal."""
    b, t = x.shape
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    win = _hann(n_fft)
    n_frames = 1 + max(0, (t - n_fft) // hop)
    if n_frames <= 0:
        return mx.zeros((b, n_fft // 2 + 1, 1))

    pad_total = (n_frames - 1) * hop + n_fft
    if pad_total > t:
        x = mx.pad(x, [(0, 0), (0, pad_total - t)])
    nf_i = int(n_fft)
    hop_i = int(hop)
    f_idx = mx.arange(n_frames, dtype=mx.int32)[:, None] * hop_i + mx.arange(nf_i, dtype=mx.int32)[None, :]
    flat = mx.reshape(f_idx, (-1,))
    taken = mx.take(x, flat, axis=1)
    frames = mx.reshape(taken, (b, n_frames, nf_i)) * win
    spec = mx.fft.rfft(frames, n=n_fft)
    return mx.transpose(spec, (0, 2, 1))


def stft_linear_mag(x: mx.array, n_fft: int, hop: int) -> mx.array:
    """x: [B, T] → linear magnitude STFT [B, F, n_frames] (``|stft_complex(x)|``)."""
    return mx.abs(stft_complex(x, n_fft, hop))


def stft_log_mag(x: mx.array, n_fft: int, hop: int) -> mx.array:
    """x: [B, T] → log magnitude STFT [B, F, n_frames]."""
    m = stft_linear_mag(x, n_fft, hop)
    return mx.log(m + 1e-5)


def _frame_signal_mx(x: mx.array, frame: int, hop: int) -> mx.array:
    """Frame ``x`` [B, T] into [B, N, frame], padding the tail when needed."""
    b, t = x.shape
    frame_i = max(1, int(frame))
    hop_i = max(1, int(hop))
    n_frames = 1 + max(0, (t - frame_i) // hop_i)
    pad_total = (n_frames - 1) * hop_i + frame_i
    if pad_total > t:
        x = mx.pad(x, [(0, 0), (0, pad_total - t)])
    idx = mx.arange(n_frames, dtype=mx.int32)[:, None] * hop_i + mx.arange(frame_i, dtype=mx.int32)[None, :]
    flat = mx.reshape(idx, (-1,))
    taken = mx.take(x, flat, axis=1)
    return mx.reshape(taken, (b, n_frames, frame_i))


def harmonic_f0_voicing_loss(
    pred_freq: mx.array | None,
    pred_amp: mx.array | None,
    tgt: mx.array,
    *,
    sample_rate: int,
    frame: int = 512,
    hop: int = 256,
    lags: tuple[int, ...] = (32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 143, 160, 180, 202, 226, 254),
    voicing_threshold: float = 0.30,
) -> tuple[mx.array, mx.array, mx.array]:
    """Supervise harmonic-source F0/amp with target waveform autocorrelation.

    The target is computed from the waveform itself: for each frame we pick the
    lag with strongest normalized autocorrelation, then train predicted F0 only
    on voiced frames. Amplitude is trained toward a soft voicing strength.
    """
    z = mx.array(0.0, dtype=mx.float32)
    if pred_freq is None or pred_amp is None:
        return z, z, z
    frame_i = max(1, int(frame))
    hop_i = max(1, int(hop))
    lags_i = tuple(int(l) for l in lags if 0 < int(l) < frame_i)
    if not lags_i:
        return z, z, z

    x = tgt[..., 0] if len(tgt.shape) == 3 else tgt
    x = x.astype(mx.float32)
    frames = _frame_signal_mx(x, frame_i, hop_i)
    win = _hann(frame_i)
    frames = frames * win
    frames = frames - mx.mean(frames, axis=2, keepdims=True)
    energy = mx.mean(frames * frames, axis=2)

    corrs = []
    for lag in lags_i:
        a = frames[:, :, :-lag]
        b = frames[:, :, lag:]
        num = mx.mean(a * b, axis=2)
        den = mx.sqrt(mx.mean(a * a, axis=2) * mx.mean(b * b, axis=2) + 1e-8)
        corrs.append(num / den)
    corr = mx.stack(corrs, axis=2)
    best_corr = mx.max(corr, axis=2)
    best_i = mx.argmax(corr, axis=2)
    lag_values = mx.array(lags_i, dtype=mx.float32)
    onehot = (best_i[:, :, None] == mx.arange(len(lags_i), dtype=mx.int32)[None, None, :]).astype(mx.float32)
    target_lag = mx.sum(onehot * lag_values[None, None, :], axis=2)
    target_f0 = float(sample_rate) / mx.maximum(target_lag, mx.array(1.0, dtype=mx.float32))

    threshold = float(voicing_threshold)
    denom = max(1.0 - threshold, 1e-3)
    strength = (best_corr - threshold) / denom
    strength = mx.minimum(mx.maximum(strength, mx.array(0.0, dtype=mx.float32)), mx.array(1.0, dtype=mx.float32))
    strength = strength * (energy > 1e-6).astype(mx.float32)

    pf = _frame_signal_mx(pred_freq[..., 0].astype(mx.float32), frame_i, hop_i)
    pa = _frame_signal_mx(pred_amp[..., 0].astype(mx.float32), frame_i, hop_i)
    pred_f0 = mx.mean(pf, axis=2)

    voiced_den = mx.sum(strength) + 1e-6
    f0_loss = mx.sum(strength * mx.abs(mx.log(pred_f0 + 1e-4) - mx.log(target_f0 + 1e-4))) / voiced_den
    amp_loss = mx.mean(mx.abs(pa - strength[:, :, None]))
    voiced_frac = mx.mean((strength > 0.0).astype(mx.float32))
    return f0_loss, amp_loss, voiced_frac


def _stft_complex_pair(pred: mx.array, tgt: mx.array, n_fft: int, hop: int) -> tuple[mx.array, mx.array]:
    """Aligned complex STFT ``[B, F, T]`` pair. One batched rfft over concat(pred, tgt)."""
    p = pred[..., 0]
    q = tgt[..., 0]
    b = p.shape[0]
    pq = mx.concatenate([p, q], axis=0)
    s = stft_complex(pq, n_fft, hop)
    return s[:b], s[b:]


def _stft_linear_mag_pair(pred: mx.array, tgt: mx.array, n_fft: int, hop: int) -> tuple[mx.array, mx.array]:
    """Aligned linear-mag STFT ``[B, F, T]`` pair via one batched rfft + ``mx.abs``."""
    sp, sq = _stft_complex_pair(pred, tgt, n_fft, hop)
    return mx.abs(sp), mx.abs(sq)


def _stft_log_mag_pair(pred: mx.array, tgt: mx.array, n_fft: int, hop: int) -> tuple[mx.array, mx.array]:
    """Aligned log-mag STFT tensors ``[B, F, T]`` for pred vs tgt."""
    mp, mq = _stft_linear_mag_pair(pred, tgt, n_fft, hop)
    return mx.log(mp + 1e-5), mx.log(mq + 1e-5)


def _mel_filterbank_numpy(
    *,
    n_fft: int,
    n_mels: int,
    sample_rate: float,
    fmin: float,
    fmax: float,
):
    """Triangular mel filter bank, shape ``[n_mels, n_fft // 2 + 1]`` float32 (NumPy only)."""
    import numpy as np

    n_freqs = n_fft // 2 + 1
    fmax = min(float(fmax), float(sample_rate) * 0.5)
    fmin = max(0.0, float(fmin))
    if fmax <= fmin:
        raise ValueError(f"mel_fmax ({fmax}) must be > mel_fmin ({fmin})")

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_lo = hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    mel_hi = hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2, dtype=np.float64)
    hz_pts = mel_to_hz(mel_pts)
    fft_bins = np.floor((n_fft + 1) * hz_pts / float(sample_rate)).astype(np.int64)
    fft_bins = np.clip(fft_bins, 0, n_freqs)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lo, peak, hi = int(fft_bins[i]), int(fft_bins[i + 1]), int(fft_bins[i + 2])
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


_MEL_FB_MX_CACHE: dict[tuple[int, int, int, int, int], mx.array] = {}
_BAND_SPLIT_FB_MX_CACHE: dict[tuple[int, int, int, int], mx.array] = {}


def _mel_cache_key(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> tuple[int, int, int, int, int]:
    return (sample_rate, n_fft, n_mels, int(round(fmin * 1000.0)), int(round(fmax * 1000.0)))


def mel_filterbank_mx(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> mx.array:
    """Cached ``[n_mels, n_fft//2+1]`` mel weights as MLX array."""
    key = _mel_cache_key(sample_rate, n_fft, n_mels, fmin, fmax)
    if key not in _MEL_FB_MX_CACHE:
        fb_np = _mel_filterbank_numpy(
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=float(sample_rate),
            fmin=fmin,
            fmax=fmax,
        )
        _MEL_FB_MX_CACHE[key] = mx.array(fb_np)
    return _MEL_FB_MX_CACHE[key]


def _fir_lowpass_kernel(sample_rate: int, cutoff_hz: float, taps: int) -> mx.array:
    sr = float(sample_rate)
    cutoff = min(max(float(cutoff_hz), 1.0), 0.499 * sr)
    n = mx.arange(taps, dtype=mx.float32)
    center = float(taps // 2)
    m = n - center
    x = 2.0 * cutoff / sr * m
    pix = math.pi * x
    sinc = mx.where(mx.abs(m) < 1e-6, mx.ones_like(x), mx.sin(pix) / (pix + 1e-20))
    h = (2.0 * cutoff / sr) * sinc
    win = 0.54 - 0.46 * mx.cos(2.0 * math.pi * n / float(max(taps - 1, 1)))
    h = h * win
    return h / (mx.sum(h) + 1e-12)


def band_split_filterbank_mx(
    sample_rate: int,
    cutoffs_hz: tuple[float, float] = (2500.0, 5000.0),
    taps: int = 129,
) -> mx.array:
    """Fixed low/mid/high FIR filterbank, shape ``[3, taps, 1]`` for ``mx.conv1d``."""
    taps_i = int(taps)
    if taps_i < 3 or taps_i % 2 == 0:
        raise ValueError("band_split_taps must be odd and >= 3")
    c1, c2 = float(cutoffs_hz[0]), float(cutoffs_hz[1])
    key = (int(sample_rate), int(round(c1 * 1000.0)), int(round(c2 * 1000.0)), taps_i)
    if key not in _BAND_SPLIT_FB_MX_CACHE:
        lp1 = _fir_lowpass_kernel(sample_rate, c1, taps_i)
        lp2 = _fir_lowpass_kernel(sample_rate, c2, taps_i)
        delta = (mx.arange(taps_i, dtype=mx.int32) == int(taps_i // 2)).astype(mx.float32)
        low = lp1
        mid = lp2 - lp1
        high = delta - lp2
        fb = mx.stack([low, mid, high], axis=0)
        _BAND_SPLIT_FB_MX_CACHE[key] = mx.reshape(fb, (3, taps_i, 1))
    return _BAND_SPLIT_FB_MX_CACHE[key]


def apply_band_split_filterbank(
    x: mx.array,
    *,
    sample_rate: int,
    cutoffs_hz: tuple[float, float] = (2500.0, 5000.0),
    taps: int = 129,
) -> mx.array:
    """Filter ``[B,T,1]`` into three bands or depthwise-filter ``[B,T,3]`` branch outputs."""
    fb = band_split_filterbank_mx(sample_rate, cutoffs_hz, taps)
    ch = int(x.shape[-1])
    if ch == 1:
        return mx.conv1d(x, fb, padding=int(taps) // 2)
    if ch == 3:
        return mx.conv1d(x, fb, padding=int(taps) // 2, groups=3)
    raise ValueError("band split filterbank expects 1 or 3 channels")


def band_l1_loss(
    pred: mx.array,
    tgt: mx.array,
    *,
    sample_rate: int,
    cutoffs_hz: tuple[float, float] = (2500.0, 5000.0),
    taps: int = 129,
    weights: tuple[float, float, float] = (0.25, 1.0, 1.5),
    floor: float = 0.015,
) -> mx.array:
    """Band-normalized L1: mean abs error per band divided by target band energy."""
    bp = apply_band_split_filterbank(pred, sample_rate=sample_rate, cutoffs_hz=cutoffs_hz, taps=taps)
    bt = apply_band_split_filterbank(tgt, sample_rate=sample_rate, cutoffs_hz=cutoffs_hz, taps=taps)
    err = mx.mean(mx.abs(bp - bt), axis=(0, 1))
    ref = mx.mean(mx.abs(bt), axis=(0, 1))
    w = mx.array(tuple(float(x) for x in weights), dtype=mx.float32)
    denom = ref + max(0.0, float(floor)) + 1e-8
    return mx.sum(w * (err / denom)) / (mx.sum(w) + 1e-8)


def band_normalized_l1(
    pred_bands: mx.array,
    target_bands: mx.array,
    *,
    weights: tuple[float, float, float] = (0.10, 1.0, 2.0),
    floor: float = 0.015,
) -> mx.array:
    """Band-normalized L1 between two ``[B,T,3]`` band tensors."""
    err = mx.mean(mx.abs(pred_bands - target_bands), axis=(0, 1))
    ref = mx.mean(mx.abs(target_bands), axis=(0, 1))
    w = mx.array(tuple(float(x) for x in weights), dtype=mx.float32)
    denom = ref + max(0.0, float(floor)) + 1e-8
    return mx.sum(w * (err / denom)) / (mx.sum(w) + 1e-8)


def band_branch_l1_loss(
    pred_bands: mx.array,
    tgt: mx.array,
    *,
    sample_rate: int,
    cutoffs_hz: tuple[float, float] = (2500.0, 5000.0),
    taps: int = 129,
    weights: tuple[float, float, float] = (0.10, 1.0, 2.0),
    floor: float = 0.015,
) -> mx.array:
    """Supervise decoder low/mid/high branches against the matching target waveform bands."""
    target_bands = apply_band_split_filterbank(
        tgt,
        sample_rate=sample_rate,
        cutoffs_hz=cutoffs_hz,
        taps=taps,
    )
    return band_normalized_l1(pred_bands, target_bands, weights=weights, floor=floor)


def _linear_mag_to_mel(mag_bft: mx.array, mel_fb: mx.array) -> mx.array:
    """``mag_bft`` [B, F, T], ``mel_fb`` [M, F] → log-mel scale linear magnitudes [B, M, T]."""
    x = mx.transpose(mag_bft, (0, 2, 1))
    fb_t = mx.transpose(mel_fb, (1, 0))
    y = mx.matmul(x, fb_t)
    return mx.transpose(y, (0, 2, 1))


def mel_log_bin_losses(
    pred: mx.array,
    tgt: mx.array,
    mel_fb: mx.array,
    n_fft: int,
    hop: int,
    *,
    eps: float = 1e-5,
) -> tuple[mx.array, mx.array]:
    """Mean L1 and mean L2 on **log** mel magnitudes (per-bin). Returns (l1, l2)."""
    mp, mq = _stft_linear_mag_pair(pred, tgt, n_fft, hop)
    mel_p = _linear_mag_to_mel(mp, mel_fb)
    mel_t = _linear_mag_to_mel(mq, mel_fb)
    lp = mx.log(mel_p + eps)
    lt = mx.log(mel_t + eps)
    d = lp - lt
    return mx.mean(mx.abs(d)), mx.mean(d * d)


def stft_loss(pred: mx.array, tgt: mx.array, n_fft: int, hop: int) -> mx.array:
    """pred, tgt: [B, T, 1] → scalar L1 on log-mags."""
    lp, lq = _stft_log_mag_pair(pred, tgt, n_fft, hop)
    return mx.mean(mx.abs(lp - lq))


def _mean_abs_logmag_l1_hf(lp: mx.array, lq: mx.array, hf_gamma: float) -> mx.array:
    """Mean |Δlogmag| with optional quadratically growing weights toward Nyquist (``lp``, ``lq``: [B,F,T])."""
    d = mx.abs(lp - lq)
    return _weighted_mean_hf(d, hf_gamma)


def _weighted_mean_hf(x: mx.array, hf_gamma: float) -> mx.array:
    """Mean of non-negative ``[B,F,T]`` values with optional HF emphasis."""
    g = float(hf_gamma)
    if g <= 0.0:
        return mx.mean(x)
    f_dim = int(x.shape[1])
    if f_dim <= 1:
        return mx.mean(x)
    fi = mx.arange(f_dim, dtype=mx.float32) / float(f_dim - 1)
    w1 = 1.0 + g * (fi * fi)
    w = mx.reshape(w1, (1, f_dim, 1))
    b, _, t = int(x.shape[0]), f_dim, int(x.shape[2])
    den = mx.sum(w1) * float(b * t)
    return mx.sum(x * w) / den


def stft_gradient_l1_from_log_mag(
    lp: mx.array,
    lq: mx.array,
    *,
    freq_weight: float = 1.0,
    time_weight: float = 1.0,
) -> mx.array:
    """Weighted mean L1 between ∂/∂f and ∂/∂t of log-mag (horizontal harmonics → emphasize freq axis)."""
    # lp, lq: [B, F, T] — use slices (``mx.diff`` not in all MLX versions)
    wf = max(0.0, float(freq_weight))
    wt = max(0.0, float(time_weight))
    terms: list[mx.array] = []
    weights: list[float] = []
    if lp.shape[1] > 1 and wf > 0:
        gp = lp[:, 1:, :] - lp[:, :-1, :]
        gq = lq[:, 1:, :] - lq[:, :-1, :]
        terms.append(wf * mx.mean(mx.abs(gp - gq)))
        weights.append(wf)
    if lp.shape[2] > 1 and wt > 0:
        gp = lp[:, :, 1:] - lp[:, :, :-1]
        gq = lq[:, :, 1:] - lq[:, :, :-1]
        terms.append(wt * mx.mean(mx.abs(gp - gq)))
        weights.append(wt)
    if not terms:
        return mx.array(0.0)
    wsum = float(sum(weights))
    return mx.sum(mx.stack(terms)) / (wsum if wsum > 0 else 1.0)


def _shift_freq_edge(x: mx.array, offset: int) -> mx.array:
    """Frequency-axis shift with edge replication; ``x`` is ``[B,F,T]``."""
    off = int(offset)
    if off == 0:
        return x
    f_dim = int(x.shape[1])
    if f_dim <= 1:
        return x
    n = min(abs(off), f_dim - 1)
    if off > 0:
        return mx.concatenate([x[:, :n, :], x[:, :-n, :]], axis=1)
    return mx.concatenate([x[:, n:, :], x[:, -n:, :]], axis=1)


def _target_hf_peak_mask_and_contrast(
    lq: mx.array,
    *,
    n_fft: int,
    sample_rate: int,
    min_hz: float,
    gate_db: float,
    radius: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Return target local-peak mask, neighborhood mean, and positive contrast for ``[B,F,T]`` log-mag."""
    f_dim = int(lq.shape[1])
    if f_dim <= 2:
        z = mx.zeros_like(lq)
        return z, z, z

    r = max(1, int(radius))
    q_neigh = mx.zeros_like(lq)
    n_terms = 0
    for k in range(1, r + 1):
        q_neigh = q_neigh + _shift_freq_edge(lq, k) + _shift_freq_edge(lq, -k)
        n_terms += 2
    q_neigh = q_neigh / float(max(1, n_terms))

    left = mx.concatenate([lq[:, :1, :] - 1.0e9, lq[:, :-1, :]], axis=1)
    right = mx.concatenate([lq[:, 1:, :], lq[:, -1:, :] - 1.0e9], axis=1)
    local_peak = mx.logical_and(lq >= left, lq >= right)

    sr = float(sample_rate)
    hz = mx.arange(f_dim, dtype=mx.float32) * (sr / float(max(1, int(n_fft))))
    hf = mx.reshape(hz >= max(0.0, float(min_hz)), (1, f_dim, 1))
    hf_f = hf.astype(mx.float32)
    lq_hf = lq * hf_f + (1.0 - hf_f) * mx.array(-1.0e9, dtype=mx.float32)
    peak = mx.max(lq_hf, axis=(1, 2), keepdims=True)
    gate_log = float(gate_db) * math.log(10.0) / 20.0
    target_gate = lq >= (peak + gate_log)

    target_contrast = mx.maximum(mx.array(0.0, dtype=mx.float32), lq - q_neigh)
    contrast_gate = target_contrast > 0.0
    mask = mx.logical_and(mx.logical_and(hf, target_gate), mx.logical_and(local_peak, contrast_gate))
    return mask.astype(mx.float32), q_neigh, target_contrast


def stft_peak_contrast_from_log_mag(
    lp: mx.array,
    lq: mx.array,
    *,
    n_fft: int,
    sample_rate: int,
    min_hz: float = 2500.0,
    gate_db: float = -18.0,
    radius: int = 3,
) -> mx.array:
    """Target-gated HF harmonic contrast loss on log-magnitude STFT.

    For target local maxima above ``min_hz``, compare peak-vs-neighborhood contrast
    between prediction and target. Broadband HF energy has near-zero local contrast,
    so this term prefers harmonic ridges over a raised spectral floor.
    """
    f_dim = int(lq.shape[1])
    if f_dim <= 2:
        return mx.array(0.0)
    r = max(1, int(radius))
    p_neigh = mx.zeros_like(lp)
    n_terms = 0
    for k in range(1, r + 1):
        p_neigh = p_neigh + _shift_freq_edge(lp, k) + _shift_freq_edge(lp, -k)
        n_terms += 2
    p_neigh = p_neigh / float(max(1, n_terms))

    mask, _, target_contrast = _target_hf_peak_mask_and_contrast(
        lq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        min_hz=min_hz,
        gate_db=gate_db,
        radius=radius,
    )
    pred_contrast = lp - p_neigh
    den = mx.sum(mask) + 1e-8
    return mx.sum(mx.abs(pred_contrast - target_contrast) * mask) / den


def stft_peak_logmag_under_from_log_mag(
    lp: mx.array,
    lq: mx.array,
    *,
    n_fft: int,
    sample_rate: int,
    min_hz: float = 2500.0,
    gate_db: float = -18.0,
    radius: int = 3,
) -> mx.array:
    """Target-gated HF local-peak under-loss on log magnitude.

    Only target local maxima are masked in, so this raises missing harmonic ridge
    bins without directly rewarding a broadband high-frequency floor.
    """
    mask, _, _ = _target_hf_peak_mask_and_contrast(
        lq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        min_hz=min_hz,
        gate_db=gate_db,
        radius=radius,
    )
    den = mx.sum(mask) + 1e-8
    under = mx.maximum(mx.array(0.0, dtype=mx.float32), lq - lp)
    return mx.sum(under * mask) / den


def stft_freq_autocorr_from_log_mag(
    lp: mx.array,
    lq: mx.array,
    *,
    n_fft: int,
    sample_rate: int,
    min_hz: float = 300.0,
    lags_hz: tuple[float, ...] = (60.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0),
) -> mx.array:
    """Normalized frequency-axis autocorrelation loss for harmonic spacing.

    Local peak losses can be satisfied with vertical impulses. This term compares
    per-frame autocorrelation of log spectra at plausible F0 lags, so it rewards
    harmonic comb spacing while staying independent of any semantic teacher.
    """
    f_dim = int(lq.shape[1])
    if f_dim <= 4:
        return mx.array(0.0)
    bin_hz = float(sample_rate) / float(max(1, int(n_fft)))
    hz = mx.arange(f_dim, dtype=mx.float32) * bin_hz
    band = mx.reshape(hz >= max(0.0, float(min_hz)), (1, f_dim, 1)).astype(mx.float32)
    den_band = mx.sum(band, axis=1, keepdims=True) + 1e-6
    p_mean = mx.sum(lp * band, axis=1, keepdims=True) / den_band
    q_mean = mx.sum(lq * band, axis=1, keepdims=True) / den_band
    xp = (lp - p_mean) * band
    xq = (lq - q_mean) * band
    p_var = mx.sum(xp * xp, axis=1) / mx.squeeze(den_band, axis=1) + 1e-5
    q_var = mx.sum(xq * xq, axis=1) / mx.squeeze(den_band, axis=1) + 1e-5
    total = mx.array(0.0)
    n_used = 0
    for lag_hz in lags_hz:
        lag = int(round(float(lag_hz) / bin_hz))
        if lag < 1 or lag >= f_dim - 1:
            continue
        pair_mask = band[:, :-lag, :] * band[:, lag:, :]
        den = mx.sum(pair_mask, axis=1) + 1e-6
        ap = mx.sum(xp[:, :-lag, :] * xp[:, lag:, :] * pair_mask, axis=1) / den
        aq = mx.sum(xq[:, :-lag, :] * xq[:, lag:, :] * pair_mask, axis=1) / den
        ap = ap / p_var
        aq = aq / q_var
        total = total + mx.mean(mx.abs(ap - aq))
        n_used += 1
    if n_used <= 0:
        return mx.array(0.0)
    return total / float(n_used)


def stft_stationary_line_excess_from_log_mag(
    lp: mx.array,
    lq: mx.array,
    *,
    n_fft: int,
    sample_rate: int,
    min_hz: float = 1000.0,
    radius: int = 5,
    margin: float = 0.08,
) -> mx.array:
    """Penalize narrow frequency lines that are more stationary than target lines.

    Mean-over-time log spectra expose horizontal fixed-frequency artifacts. We compare
    each frequency bin against its local frequency neighborhood and only penalize the
    prediction's excess local contrast above the target plus a margin.
    """
    f_dim = int(lp.shape[1])
    if f_dim <= 2:
        return mx.array(0.0)
    prof_p = mx.mean(lp, axis=2)
    prof_q = mx.mean(lq, axis=2)
    r = max(1, int(radius))
    neigh_p = mx.zeros_like(prof_p)
    neigh_q = mx.zeros_like(prof_q)
    n_terms = 0
    pp = prof_p[:, :, None]
    qq = prof_q[:, :, None]
    for k in range(1, r + 1):
        neigh_p = neigh_p + mx.squeeze(_shift_freq_edge(pp, k), axis=2) + mx.squeeze(_shift_freq_edge(pp, -k), axis=2)
        neigh_q = neigh_q + mx.squeeze(_shift_freq_edge(qq, k), axis=2) + mx.squeeze(_shift_freq_edge(qq, -k), axis=2)
        n_terms += 2
    neigh_p = neigh_p / float(max(1, n_terms))
    neigh_q = neigh_q / float(max(1, n_terms))
    pred_line = prof_p - neigh_p
    target_line = mx.maximum(mx.array(0.0, dtype=mx.float32), prof_q - neigh_q)
    hz = mx.arange(f_dim, dtype=mx.float32) * (float(sample_rate) / float(max(1, int(n_fft))))
    band = (hz >= max(0.0, float(min_hz))).astype(mx.float32)[None, :]
    excess = mx.maximum(mx.array(0.0, dtype=mx.float32), pred_line - target_line - float(margin))
    return mx.sum(excess * band) / (mx.sum(band) * float(lp.shape[0]) + 1e-8)


def multi_stft_stationary_line_loss(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    sample_rate: int,
    min_hz: float = 1000.0,
    radius: int = 5,
    margin: float = 0.08,
    scale_weights: tuple[float, ...] | None = None,
) -> mx.array:
    if not scales:
        return mx.array(0.0)
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    total = mx.array(0.0)
    w_den = float(sum(ws))
    for (n_fft, hop), w in zip(scales, ws):
        lp, lq = _stft_log_mag_pair(pred, tgt, n_fft, hop)
        total = total + float(w) * stft_stationary_line_excess_from_log_mag(
            lp,
            lq,
            n_fft=n_fft,
            sample_rate=sample_rate,
            min_hz=min_hz,
            radius=radius,
            margin=margin,
        )
    return total / max(w_den, 1e-8)


def multi_stft_loss(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    hf_emphasis: float = 0.0,
    scale_weights: tuple[float, ...] | None = None,
) -> mx.array:
    if not scales:
        return mx.array(0.0)
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    w_den = float(sum(ws))
    if w_den <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    total = mx.array(0.0)
    for (n_fft, hop), w in zip(scales, ws):
        lp, lq = _stft_log_mag_pair(pred, tgt, n_fft, hop)
        total = total + float(w) * _mean_abs_logmag_l1_hf(lp, lq, hf_emphasis)
    return total / w_den


def _mean_abs_mag_l1_hf(mp: mx.array, mq: mx.array, hf_gamma: float) -> mx.array:
    """Mean |Δmag| on **linear** magnitudes with optional HF emphasis (``mp``, ``mq``: [B,F,T])."""
    d = mx.abs(mp - mq)
    g = float(hf_gamma)
    if g <= 0.0:
        return mx.mean(d)
    f_dim = int(mp.shape[1])
    if f_dim <= 1:
        return mx.mean(d)
    fi = mx.arange(f_dim, dtype=mx.float32) / float(f_dim - 1)
    w1 = 1.0 + g * (fi * fi)
    w = mx.reshape(w1, (1, f_dim, 1))
    b, _, t = int(mp.shape[0]), f_dim, int(mp.shape[2])
    den = mx.sum(w1) * float(b * t)
    return mx.sum(d * w) / den


def multi_stft_mag_l1_linear(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    hf_emphasis: float = 0.0,
    scale_weights: tuple[float, ...] | None = None,
) -> mx.array:
    """Mean L1 on linear magnitudes (multi-scale), same weighting as ``multi_stft_loss``."""
    if not scales:
        return mx.array(0.0)
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    w_den = float(sum(ws))
    if w_den <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    total = mx.array(0.0)
    for (n_fft, hop), w in zip(scales, ws):
        mp, mq = _stft_linear_mag_pair(pred, tgt, n_fft, hop)
        total = total + float(w) * _mean_abs_mag_l1_hf(mp, mq, hf_emphasis)
    return total / w_den


def stft_logmag_cosine_1m(lp: mx.array, lq: mx.array) -> mx.array:
    """Mean ``(1 - cos)`` over batch, cosine between flattened log-mag spectra per sample. lp,lq: [B,F,T]."""
    b = lp.shape[0]
    p = lp.reshape(b, -1)
    q = lq.reshape(b, -1)
    dot = mx.sum(p * q, axis=1)
    np = mx.sqrt(mx.sum(p * p, axis=1) + 1e-8)
    nq = mx.sqrt(mx.sum(q * q, axis=1) + 1e-8)
    c = dot / (np * nq)
    return mx.mean(1.0 - c)


def multi_stft_spectral_terms(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    with_grad: bool,
    with_cos_1m: bool,
    grad_freq_weight: float = 1.0,
    grad_time_weight: float = 1.0,
    hf_emphasis: float = 0.0,
    scale_weights: tuple[float, ...] | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """One STFT per scale: weighted mean mag L1, weighted mean grad L1 or 0, weighted mean (1−cos) or 0."""
    if not scales:
        z = mx.array(0.0)
        return z, z, z
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    w_den = float(sum(ws))
    if w_den <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    tot_mag = mx.array(0.0)
    tot_grad = mx.array(0.0)
    tot_cos = mx.array(0.0)
    for (n_fft, hop), w in zip(scales, ws):
        wf = float(w)
        lp, lq = _stft_log_mag_pair(pred, tgt, n_fft, hop)
        tot_mag = tot_mag + wf * _mean_abs_logmag_l1_hf(lp, lq, hf_emphasis)
        if with_grad:
            tot_grad = tot_grad + wf * stft_gradient_l1_from_log_mag(
                lp,
                lq,
                freq_weight=grad_freq_weight,
                time_weight=grad_time_weight,
            )
        if with_cos_1m:
            tot_cos = tot_cos + wf * stft_logmag_cosine_1m(lp, lq)
    return tot_mag / w_den, tot_grad / w_den, tot_cos / w_den


def multi_stft_all_terms(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    with_grad: bool,
    with_cos_1m: bool,
    with_linear: bool,
    with_sc: bool,
    with_complex: bool,
    with_excess: bool = False,
    with_peak_contrast: bool = False,
    with_peak_mag: bool = False,
    with_freq_ac: bool = False,
    grad_freq_weight: float = 1.0,
    grad_time_weight: float = 1.0,
    hf_emphasis: float = 0.0,
    excess_margin: float = 0.20,
    sample_rate: int = 16000,
    peak_min_hz: float = 2500.0,
    peak_gate_db: float = -18.0,
    peak_radius: int = 3,
    freq_ac_min_hz: float = 300.0,
    freq_ac_lags_hz: tuple[float, ...] = (60.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0),
    scale_weights: tuple[float, ...] | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """One complex STFT pair per scale, then derive all enabled spectral losses.

    This avoids recomputing the same rFFT for log-mag, spectral convergence,
    complex L1, and optional linear-mag terms.
    """
    z = mx.array(0.0)
    if not scales:
        return z, z, z, z, z, z, z, z, z, z
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    w_den = float(sum(ws))
    if w_den <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    tot_mag = mx.array(0.0)
    tot_grad = mx.array(0.0)
    tot_cos = mx.array(0.0)
    tot_lin = mx.array(0.0)
    tot_sc = mx.array(0.0)
    tot_cx = mx.array(0.0)
    tot_excess = mx.array(0.0)
    tot_peak = mx.array(0.0)
    tot_peak_mag = mx.array(0.0)
    tot_freq_ac = mx.array(0.0)
    for (n_fft, hop), w in zip(scales, ws):
        wf = float(w)
        sp, sq = _stft_complex_pair(pred, tgt, n_fft, hop)
        mp = mx.abs(sp)
        mq = mx.abs(sq)
        lp = mx.log(mp + 1e-5)
        lq = mx.log(mq + 1e-5)
        tot_mag = tot_mag + wf * _mean_abs_logmag_l1_hf(lp, lq, hf_emphasis)
        if with_grad:
            tot_grad = tot_grad + wf * stft_gradient_l1_from_log_mag(
                lp,
                lq,
                freq_weight=grad_freq_weight,
                time_weight=grad_time_weight,
            )
        if with_cos_1m:
            tot_cos = tot_cos + wf * stft_logmag_cosine_1m(lp, lq)
        if with_linear:
            tot_lin = tot_lin + wf * _mean_abs_mag_l1_hf(mp, mq, hf_emphasis)
        if with_sc:
            tot_sc = tot_sc + wf * spectral_convergence(mp, mq)
        if with_complex:
            d_real = mx.real(sp) - mx.real(sq)
            d_imag = mx.imag(sp) - mx.imag(sq)
            tot_cx = tot_cx + wf * (0.5 * (mx.mean(mx.abs(d_real)) + mx.mean(mx.abs(d_imag))))
        if with_excess:
            excess = mx.maximum(mx.array(0.0, dtype=mx.float32), lp - lq - float(excess_margin))
            tot_excess = tot_excess + wf * _weighted_mean_hf(excess, hf_emphasis)
        if with_peak_contrast:
            tot_peak = tot_peak + wf * stft_peak_contrast_from_log_mag(
                lp,
                lq,
                n_fft=n_fft,
                sample_rate=sample_rate,
                min_hz=peak_min_hz,
                gate_db=peak_gate_db,
                radius=peak_radius,
            )
        if with_peak_mag:
            tot_peak_mag = tot_peak_mag + wf * stft_peak_logmag_under_from_log_mag(
                lp,
                lq,
                n_fft=n_fft,
                sample_rate=sample_rate,
                min_hz=peak_min_hz,
                gate_db=peak_gate_db,
                radius=peak_radius,
            )
        if with_freq_ac:
            tot_freq_ac = tot_freq_ac + wf * stft_freq_autocorr_from_log_mag(
                lp,
                lq,
                n_fft=n_fft,
                sample_rate=sample_rate,
                min_hz=freq_ac_min_hz,
                lags_hz=freq_ac_lags_hz,
            )
    return (
        tot_mag / w_den,
        tot_grad / w_den,
        tot_cos / w_den,
        tot_lin / w_den,
        tot_sc / w_den,
        tot_cx / w_den,
        tot_excess / w_den,
        tot_peak / w_den,
        tot_peak_mag / w_den,
        tot_freq_ac / w_den,
    )


def high_frequency_stft_terms(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    sample_rate: int,
    min_hz: float = 2500.0,
    gate_db: float = -18.0,
    under_margin: float = 0.05,
    peak_mask: bool = False,
    scale_weights: tuple[float, ...] | None = None,
) -> tuple[mx.array, mx.array]:
    """Target-gated HF under-energy and HF-only spectral convergence.

    The under-energy term only looks above ``min_hz`` and only where the target
    has meaningful energy relative to its per-sample peak. This avoids solving
    missing harmonics by raising broadband noise.
    """
    z = mx.array(0.0)
    if not scales:
        return z, z
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    w_den = float(sum(ws))
    if w_den <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    tot_under = mx.array(0.0)
    tot_sc = mx.array(0.0)
    sr = float(sample_rate)
    min_hz_f = max(0.0, float(min_hz))
    gate_log = float(gate_db) * math.log(10.0) / 20.0
    margin = max(0.0, float(under_margin))
    for (n_fft, hop), w in zip(scales, ws):
        wf = float(w)
        sp, sq = _stft_complex_pair(pred, tgt, n_fft, hop)
        mp = mx.abs(sp)
        mq = mx.abs(sq)
        lp = mx.log(mp + 1e-5)
        lq = mx.log(mq + 1e-5)

        f_dim = int(mq.shape[1])
        hz = mx.arange(f_dim, dtype=mx.float32) * (sr / float(max(1, n_fft)))
        hf = mx.reshape(hz >= min_hz_f, (1, f_dim, 1))
        peak = mx.max(lq, axis=(1, 2), keepdims=True)
        target_gate = lq >= (peak + gate_log)
        if peak_mask and f_dim > 2:
            left = mx.concatenate([lq[:, :1, :] - 1.0e9, lq[:, :-1, :]], axis=1)
            right = mx.concatenate([lq[:, 1:, :], lq[:, -1:, :] - 1.0e9], axis=1)
            target_gate = mx.logical_and(target_gate, mx.logical_and(lq >= left, lq >= right))
        mask = mx.logical_and(hf, target_gate).astype(mx.float32)
        den = mx.sum(mask) + 1e-8
        under = mx.maximum(mx.array(0.0, dtype=mx.float32), lq - lp - margin)
        tot_under = tot_under + wf * (mx.sum(under * mask) / den)

        hf_f = hf.astype(mx.float32)
        mp_h = mp * hf_f
        mq_h = mq * hf_f
        tot_sc = tot_sc + wf * spectral_convergence(mp_h, mq_h)
    return tot_under / w_den, tot_sc / w_den


def high_frequency_complex_stft_l1(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    sample_rate: int,
    min_hz: float = 2500.0,
    scale_weights: tuple[float, ...] | None = None,
) -> mx.array:
    """Complex STFT L1 averaged only over high-frequency bins.

    Unlike under-energy losses, this penalizes both missing target detail and
    unmatched broadband HF noise.
    """
    z = mx.array(0.0)
    if not scales:
        return z
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    w_den = float(sum(ws))
    if w_den <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    sr = float(sample_rate)
    min_hz_f = max(0.0, float(min_hz))
    total = mx.array(0.0)
    for (n_fft, hop), w in zip(scales, ws):
        sp, sq = _stft_complex_pair(pred, tgt, n_fft, hop)
        f_dim = int(sp.shape[1])
        hz = mx.arange(f_dim, dtype=mx.float32) * (sr / float(max(1, n_fft)))
        mask = mx.reshape((hz >= min_hz_f).astype(mx.float32), (1, f_dim, 1))
        d = 0.5 * (mx.abs(mx.real(sp) - mx.real(sq)) + mx.abs(mx.imag(sp) - mx.imag(sq)))
        denom = mx.sum(mask) * float(int(d.shape[0]) * int(d.shape[2])) + 1e-8
        total = total + float(w) * (mx.sum(d * mask) / denom)
    return total / w_den


def spectral_convergence(mp: mx.array, mq: mx.array, *, eps: float = 1e-8) -> mx.array:
    """Spectral convergence (Yamamoto et al., Parallel WaveGAN): ``‖|S(ŷ)|−|S(y)|‖_F / ‖|S(y)|‖_F``.

    Inputs are linear-mag STFTs ``[B, F, T]``. Returns batch-mean of per-sample SC.
    Scale-invariant, emphasizes relative error in high-energy bins (harmonic peaks).
    """
    b = int(mp.shape[0])
    p = mp.reshape(b, -1)
    q = mq.reshape(b, -1)
    num = mx.sqrt(mx.sum((p - q) * (p - q), axis=1) + eps)
    den = mx.sqrt(mx.sum(q * q, axis=1) + eps)
    return mx.mean(num / den)


def multi_stft_spectral_convergence(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    scale_weights: tuple[float, ...] | None = None,
) -> mx.array:
    """Multi-scale SC; same weighting scheme as ``multi_stft_loss``."""
    if not scales:
        return mx.array(0.0)
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    w_den = float(sum(ws))
    if w_den <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    total = mx.array(0.0)
    for (n_fft, hop), w in zip(scales, ws):
        mp, mq = _stft_linear_mag_pair(pred, tgt, n_fft, hop)
        total = total + float(w) * spectral_convergence(mp, mq)
    return total / w_den


def multi_stft_complex_l1(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    scale_weights: tuple[float, ...] | None = None,
) -> mx.array:
    """Mean L1 on complex STFT real + imag parts (multi-scale). Captures phase (cheap vs mag-only)."""
    if not scales:
        return mx.array(0.0)
    ws = [1.0] * len(scales) if scale_weights is None else list(scale_weights)
    if len(ws) != len(scales):
        raise ValueError("scale_weights length must match stft_scales")
    w_den = float(sum(ws))
    if w_den <= 0:
        raise ValueError("sum of STFT scale weights must be > 0")
    total = mx.array(0.0)
    for (n_fft, hop), w in zip(scales, ws):
        sp, sq = _stft_complex_pair(pred, tgt, n_fft, hop)
        d_real = mx.real(sp) - mx.real(sq)
        d_imag = mx.imag(sp) - mx.imag(sq)
        # 0.5*(L1(real)+L1(imag)) keeps same O(1) scale as magnitude-based L1 when energy split ≈ even.
        l1 = 0.5 * (mx.mean(mx.abs(d_real)) + mx.mean(mx.abs(d_imag)))
        total = total + float(w) * l1
    return total / w_den


def entropy_from_logits(logits: mx.array) -> mx.array:
    """Mean Shannon entropy of softmax(logits) over last dim (higher = flatter)."""
    logits = mx.clip(logits, -50.0, 50.0)
    m = mx.max(logits, axis=-1, keepdims=True)
    ex = mx.exp(logits - m)
    p = ex / (mx.sum(ex, axis=-1, keepdims=True) + 1e-8)
    return -mx.mean(mx.sum(p * mx.log(p + 1e-8), axis=-1))


def marginal_code_entropy_from_dist(dist: mx.array, tau: float) -> mx.array:
    """Shannon entropy of batch-marginal p̄ = mean_{b,t} softmax(-dist/τ). Differentiable; rewards code diversity."""
    logits = -dist / tau
    logits = mx.clip(logits, -50.0, 50.0)
    m = mx.max(logits, axis=-1, keepdims=True)
    ex = mx.exp(logits - m)
    p = ex / (mx.sum(ex, axis=-1, keepdims=True) + 1e-8)
    avg = mx.mean(p, axis=(0, 1))
    return -mx.sum(avg * mx.log(avg + 1e-8))
