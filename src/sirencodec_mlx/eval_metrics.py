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
