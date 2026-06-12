#!/usr/bin/env python3
"""Repository entrypoint for the conditioned two-band MLX trainer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sirencodec.mlx.train_two_band import main  # noqa: E402


if __name__ == "__main__":
    main()
