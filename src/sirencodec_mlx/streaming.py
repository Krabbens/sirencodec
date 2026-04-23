"""Causal convolutions and chunked encode/decode helpers (low-latency / streaming)."""

from __future__ import annotations

import math

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = None  # type: ignore
    nn = None  # type: ignore


class CausalConv1d(nn.Module):
    """Left-padded Conv1d for ``[B, T, C]`` (NLC). Right padding is zero."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        self.left_pad = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.left_pad, 0), (0, 0)])
        return self.conv(x)


def benchmark_causal_latency_ms(
    *,
    sample_rate: int = 16_000,
    chunk_ms: float = 20.0,
    n_warmup: int = 2,
    n_trials: int = 5,
) -> float:
    """Rough wall-clock ms for one causal conv block on a chunk (CPU sync)."""
    import time

    if mx is None:
        return 0.0
    chunk = max(1, int(sample_rate * (chunk_ms / 1000.0)))
    layer = CausalConv1d(1, 16, kernel_size=7, stride=1)
    mx.eval(layer.parameters())
    x = mx.random.normal((1, chunk, 1))
    for _ in range(n_warmup):
        y = layer(x)
        mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        y = layer(x)
        mx.eval(y)
    return (time.perf_counter() - t0) / float(n_trials) * 1000.0


def encode_decode_chunk(model, x_chunk, state=None):
    """Run full codec on one chunk; ``state`` reserved for future streaming state (currently unused)."""
    if mx is None:
        raise RuntimeError("MLX is not installed")
    y, *_ = model.forward_full(x_chunk)
    return y, state
