#!/usr/bin/env python3
"""MLX trainer entrypoint for Apple Silicon runs."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sirencodec.config import (  # noqa: E402
    Config,
    effective_codebook_sizes,
    encoder_time_stride,
    nominal_rvq_kbps,
    parse_codebook_sizes_arg,
)
from sirencodec.mlx.codec import MLXCodec  # noqa: E402
from sirencodec.mlx.train import (  # noqa: E402
    _active_stft_scales,
    _spectral_loss_batch,
    batch_mean_cosine,
    batch_multidelta_l1,
    batch_neg_log_si_sdr,
    batch_preemph_l1,
    make_train_fn,
    main,
)

__all__ = [
    "main",
    "Config",
    "MLXCodec",
    "batch_mean_cosine",
    "batch_multidelta_l1",
    "batch_neg_log_si_sdr",
    "batch_preemph_l1",
    "make_train_fn",
    "_active_stft_scales",
    "_spectral_loss_batch",
    "effective_codebook_sizes",
    "encoder_time_stride",
    "nominal_rvq_kbps",
    "parse_codebook_sizes_arg",
]

if __name__ == "__main__":
    main()
