"""STFT / mel / entropy helpers for MLX training losses."""
from __future__ import annotations

import math

import mlx.core as mx

def _hann(n: int) -> mx.array:
    i = mx.arange(n, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2.0 * math.pi * i / float(max(n - 1, 1)))


def _hann_periodic(n: int) -> mx.array:
    i = mx.arange(n, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2.0 * math.pi * i / float(max(n, 1)))


def _reflect_pad_time(x: mx.array, pad: int) -> mx.array:
    """Torch-style 1D reflect padding over ``axis=1`` for audio batches."""
    if pad <= 0:
        return x
    t = int(x.shape[1])
    if t <= 1:
        return mx.pad(x, [(0, 0), (pad, pad)])
    # PyTorch reflection excludes the edge sample:
    # [0, 1, 2, 3], pad=2 -> [2, 1, 0, 1, 2, 3, 2, 1].
    p = min(int(pad), t - 1)
    base = mx.arange(p, dtype=mx.int32)
    left_idx = p - base
    right_idx = (t - 2) - base
    left = mx.take(x, left_idx, axis=1)
    right = mx.take(x, right_idx, axis=1)
    y = mx.concatenate([left, x, right], axis=1)
    if p == pad:
        return y
    extra = pad - p
    return mx.pad(y, [(0, 0), (extra, extra)])


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


def stft_linear_mag_centered_reflect(x: mx.array, n_fft: int, hop: int) -> mx.array:
    """Torchaudio-style ``center=True, pad_mode='reflect'`` linear magnitude STFT."""
    b, t = x.shape
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    n_fft_i = int(n_fft)
    hop_i = int(hop)
    x = _reflect_pad_time(x, n_fft_i // 2)
    t_pad = int(x.shape[1])
    n_frames = 1 + max(0, (t_pad - n_fft_i) // hop_i)
    if n_frames <= 0:
        return mx.zeros((b, n_fft_i // 2 + 1, 1))
    f_idx = mx.arange(n_frames, dtype=mx.int32)[:, None] * hop_i + mx.arange(n_fft_i, dtype=mx.int32)[None, :]
    flat = mx.reshape(f_idx, (-1,))
    taken = mx.take(x, flat, axis=1)
    frames = mx.reshape(taken, (b, n_frames, n_fft_i)) * _hann_periodic(n_fft_i)
    spec = mx.fft.rfft(frames, n=n_fft_i)
    return mx.transpose(mx.abs(spec), (0, 2, 1))


def stft_log_mag(x: mx.array, n_fft: int, hop: int) -> mx.array:
    """x: [B, T] → log magnitude STFT [B, F, n_frames]."""
    m = stft_linear_mag(x, n_fft, hop)
    return mx.log(m + 1e-5)


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
    all_freqs = np.linspace(0.0, float(int(sample_rate) // 2), n_freqs, dtype=np.float64)
    f_diff = hz_pts[1:] - hz_pts[:-1]
    slopes = hz_pts[None, :] - all_freqs[:, None]
    down_slopes = (-slopes[:, :-2]) / np.maximum(f_diff[:-1], 1e-12)
    up_slopes = slopes[:, 2:] / np.maximum(f_diff[1:], 1e-12)
    fb_fm = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    return fb_fm.T.astype(np.float32, copy=False)


_MEL_FB_MX_CACHE: dict[tuple[int, int, int, int, int], mx.array] = {}


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


def _linear_mag_to_mel(mag_bft: mx.array, mel_fb: mx.array) -> mx.array:
    """``mag_bft`` [B, F, T], ``mel_fb`` [M, F] → log-mel scale linear magnitudes [B, M, T]."""
    x = mx.transpose(mag_bft, (0, 2, 1))
    fb_t = mx.transpose(mel_fb, (1, 0))
    y = mx.matmul(x, fb_t)
    return mx.transpose(y, (0, 2, 1))


def log_mel_spectrogram(
    wav: mx.array,
    mel_fb: mx.array,
    n_fft: int,
    hop: int,
    *,
    eps: float = 1e-7,
) -> mx.array:
    """Vocos-style log-mel features: waveform ``[B,T,1]`` → ``[B,frames,mels]``."""
    mag = stft_linear_mag_centered_reflect(wav[..., 0], n_fft, hop)
    mel = _linear_mag_to_mel(mag, mel_fb)
    return mx.transpose(mx.log(mel + eps), (0, 2, 1))


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
    g = float(hf_gamma)
    if g <= 0.0:
        return mx.mean(d)
    f_dim = int(lp.shape[1])
    if f_dim <= 1:
        return mx.mean(d)
    fi = mx.arange(f_dim, dtype=mx.float32) / float(f_dim - 1)
    w1 = 1.0 + g * (fi * fi)
    w = mx.reshape(w1, (1, f_dim, 1))
    b, _, t = int(lp.shape[0]), f_dim, int(lp.shape[2])
    den = mx.sum(w1) * float(b * t)
    return mx.sum(d * w) / den


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


def _mean_stationary_logmag_excess(
    lp: mx.array,
    lq: mx.array,
    *,
    sample_rate: int,
    margin: float,
    fmin: float,
    fmax: float,
    hf_emphasis: float = 0.0,
) -> mx.array:
    """Penalize excess log-mag energy that stays present across time at the same frequency."""
    excess = mx.maximum(mx.array(0.0, dtype=mx.float32), lp - lq - float(margin))
    # Horizontal bands are persistent per-frequency excess, so average over time first.
    stationary = mx.mean(excess, axis=2)
    stationary = stationary * stationary
    f_dim = int(lp.shape[1])
    if f_dim <= 1:
        return mx.mean(stationary)
    nyq = float(sample_rate) * 0.5
    hi = nyq if float(fmax) <= 0.0 else min(float(fmax), nyq)
    lo = max(0.0, float(fmin))
    if hi <= lo:
        return mx.array(0.0, dtype=mx.float32)
    freqs = mx.arange(f_dim, dtype=mx.float32) * (nyq / float(f_dim - 1))
    band = (freqs >= lo).astype(mx.float32) * (freqs <= hi).astype(mx.float32)
    if float(hf_emphasis) > 0.0:
        fi = mx.arange(f_dim, dtype=mx.float32) / float(f_dim - 1)
        band = band * (1.0 + float(hf_emphasis) * (fi * fi))
    den = mx.sum(band) * float(stationary.shape[0]) + 1e-8
    return mx.sum(stationary * mx.reshape(band, (1, f_dim))) / den


def multi_stft_band_excess(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    sample_rate: int,
    margin: float = 0.15,
    fmin: float = 1200.0,
    fmax: float = 0.0,
    hf_emphasis: float = 0.0,
    scale_weights: tuple[float, ...] | None = None,
) -> mx.array:
    """Multi-scale stationary excess penalty for artificial horizontal spectral bands."""
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
        total = total + float(w) * _mean_stationary_logmag_excess(
            lp,
            lq,
            sample_rate=sample_rate,
            margin=margin,
            fmin=fmin,
            fmax=fmax,
            hf_emphasis=hf_emphasis,
        )
    return total / w_den


def _mean_temporal_logmag_excess(
    lp: mx.array,
    lq: mx.array,
    *,
    sample_rate: int,
    margin: float,
    fmin: float,
    fmax: float,
) -> mx.array:
    """Penalize broadband per-frame target-relative log-mag excess.

    Unlike ``_mean_stationary_logmag_excess`` this averages over frequency first,
    so a frame is expensive only when many bins are above target together.
    """
    excess = mx.maximum(mx.array(0.0, dtype=mx.float32), lp - lq - float(margin))
    f_dim = int(lp.shape[1])
    if f_dim <= 1:
        frame_excess = mx.mean(excess, axis=1)
        return mx.mean(frame_excess * frame_excess)
    nyq = float(sample_rate) * 0.5
    hi = nyq if float(fmax) <= 0.0 else min(float(fmax), nyq)
    lo = max(0.0, float(fmin))
    if hi <= lo:
        return mx.array(0.0, dtype=mx.float32)
    freqs = mx.arange(f_dim, dtype=mx.float32) * (nyq / float(f_dim - 1))
    band = (freqs >= lo).astype(mx.float32) * (freqs <= hi).astype(mx.float32)
    den = mx.sum(band) + 1e-8
    frame_excess = mx.sum(excess * mx.reshape(band, (1, f_dim, 1)), axis=1) / den
    return mx.mean(frame_excess * frame_excess)


def multi_stft_temporal_stripe_excess(
    pred: mx.array,
    tgt: mx.array,
    scales: tuple[tuple[int, int], ...],
    *,
    sample_rate: int,
    margin: float = 0.08,
    fmin: float = 700.0,
    fmax: float = 0.0,
    scale_weights: tuple[float, ...] | None = None,
) -> mx.array:
    """Multi-scale broadband temporal excess penalty for vertical spectrogram stripes."""
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
        total = total + float(w) * _mean_temporal_logmag_excess(
            lp,
            lq,
            sample_rate=sample_rate,
            margin=margin,
            fmin=fmin,
            fmax=fmax,
        )
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
