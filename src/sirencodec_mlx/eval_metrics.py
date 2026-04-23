"""SI-SDR and optional PESQ for short validation clips."""

from __future__ import annotations


def si_sdr_db(reference, estimate) -> float:
    """Scale-invariant SDR in dB (numpy 1D float32)."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()
    n = min(ref.size, est.size)
    if n < 1:
        return -100.0
    ref = ref[:n]
    est = est[:n]
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    dot = np.dot(ref, est)
    s_target = dot * ref / (np.dot(ref, ref) + 1e-12)
    e_noise = est - s_target
    num = np.dot(s_target, s_target)
    den = np.dot(e_noise, e_noise) + 1e-12
    return float(10.0 * np.log10(num / den + 1e-12))


def pesq_wb_16k(reference, estimate) -> float | None:
    """Wideband PESQ @ 16 kHz if ``pesq`` is installed; else ``None``."""
    try:
        from pesq import pesq as _pesq  # type: ignore
        import numpy as np

        ref = np.asarray(reference, dtype=np.float32).ravel()
        est = np.asarray(estimate, dtype=np.float32).ravel()
        n = min(ref.size, est.size)
        if n < 256:
            return None
        return float(_pesq(16000, ref[:n], est[:n], "wb"))
    except Exception:
        return None


def stoi_16k(reference, estimate) -> float | None:
    """STOI @ 16 kHz if ``pystoi`` is installed; else ``None``."""
    try:
        from pystoi import stoi as _stoi  # type: ignore
        import numpy as np

        ref = np.asarray(reference, dtype=np.float32).ravel()
        est = np.asarray(estimate, dtype=np.float32).ravel()
        n = min(ref.size, est.size)
        if n < 256:
            return None
        return float(_stoi(ref[:n], est[:n], 16000, extended=False))
    except Exception:
        return None


def waveform_cosine(reference, estimate) -> float:
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()
    n = min(ref.size, est.size)
    if n < 1:
        return 0.0
    ref = ref[:n]
    est = est[:n]
    den = np.linalg.norm(ref) * np.linalg.norm(est) + 1e-12
    return float(np.dot(ref, est) / den)


def waveform_l1(reference, estimate) -> float:
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()
    n = min(ref.size, est.size)
    if n < 1:
        return 0.0
    return float(np.mean(np.abs(ref[:n] - est[:n])))


def log_spectral_distance_db(reference, estimate, *, n_fft: int = 512, hop: int = 128) -> float:
    """RMS distance between log-magnitude spectra in dB; lower is better."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()
    n = min(ref.size, est.size)
    if n < 1:
        return 100.0
    ref = ref[:n]
    est = est[:n]
    if n < n_fft:
        pad = n_fft - n
        ref = np.pad(ref, (0, pad))
        est = np.pad(est, (0, pad))
    win = np.hanning(n_fft)
    vals = []
    for start in range(0, max(1, ref.size - n_fft + 1), hop):
        r = ref[start : start + n_fft]
        e = est[start : start + n_fft]
        if r.size < n_fft:
            r = np.pad(r, (0, n_fft - r.size))
            e = np.pad(e, (0, n_fft - e.size))
        sr = np.abs(np.fft.rfft(r * win, n=n_fft))
        se = np.abs(np.fft.rfft(e * win, n=n_fft))
        dr = 20.0 * np.log10(np.maximum(sr, 1e-7))
        de = 20.0 * np.log10(np.maximum(se, 1e-7))
        vals.append(float(np.mean((dr - de) ** 2)))
    if not vals:
        return 100.0
    return float(np.sqrt(np.mean(vals)))


def quality_metrics_16k(reference, estimate) -> dict[str, float | None]:
    return {
        "si_sdr_db": si_sdr_db(reference, estimate),
        "pesq_wb": pesq_wb_16k(reference, estimate),
        "stoi": stoi_16k(reference, estimate),
        "lsd_db": log_spectral_distance_db(reference, estimate),
        "l1": waveform_l1(reference, estimate),
        "cos": waveform_cosine(reference, estimate),
    }
