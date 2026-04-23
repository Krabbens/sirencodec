#!/usr/bin/env python3
"""CUDA trainer entrypoint kept at the old ``train_mlx.py`` path for script compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sirencodec_mlx.config import (  # noqa: E402
    Config,
    effective_codebook_sizes,
    encoder_time_stride,
    nominal_rvq_kbps,
    parse_codebook_sizes_arg,
)
from sirencodec_mlx.torch_codec import CUDACodec  # noqa: E402
from sirencodec_mlx.train_cuda_main import main  # noqa: E402

try:  # Keep legacy MLX tests/helpers working where MLX is installed.
    from sirencodec_mlx.codec import MLXCodec as _CompatMLXCodec  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - expected on CUDA/Windows machines.
    _CompatMLXCodec = CUDACodec

MLXCodec = _CompatMLXCodec

__all__ = [
    "main",
    "Config",
    "CUDACodec",
    "MLXCodec",
    "effective_codebook_sizes",
    "encoder_time_stride",
    "nominal_rvq_kbps",
    "parse_codebook_sizes_arg",
]

if __name__ == "__main__":
    main()
