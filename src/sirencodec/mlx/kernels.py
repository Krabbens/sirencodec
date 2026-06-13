from __future__ import annotations

import mlx.core as mx


def batch_mean_cosine_metric(orig: mx.array, recon: mx.array) -> mx.array:
    """Return mean per-item cosine similarity for waveform batches."""
    batch = orig.shape[0]
    orig_flat = orig.reshape(batch, -1)
    recon_flat = recon.reshape(batch, -1)
    dot = mx.sum(orig_flat * recon_flat, axis=1)
    orig_norm = mx.sqrt(mx.sum(orig_flat * orig_flat, axis=1) + 1e-8)
    recon_norm = mx.sqrt(mx.sum(recon_flat * recon_flat, axis=1) + 1e-8)
    return mx.mean(dot / (orig_norm * recon_norm))
