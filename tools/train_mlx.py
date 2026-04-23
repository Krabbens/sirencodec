#!/usr/bin/env python3
"""MLX trainer entrypoint — implementation lives in ``sirencodec_mlx.train_mlx_main``."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: ``src/sirencodec_mlx`` is the package root.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sirencodec_mlx.codec import MLXCodec  # noqa: E402
from sirencodec_mlx.config import (  # noqa: E402
    Config,
    effective_codebook_sizes,
    encoder_time_stride,
    nominal_rvq_kbps,
    parse_codebook_sizes_arg,
)
from sirencodec_mlx.train_mlx_main import main  # noqa: E402

__all__ = [
    "main",
    "Config",
    "MLXCodec",
    "effective_codebook_sizes",
    "encoder_time_stride",
    "nominal_rvq_kbps",
    "parse_codebook_sizes_arg",
]

if __name__ == "__main__":
    main()
