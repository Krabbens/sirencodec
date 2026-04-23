"""Placeholders for adaptive frame rate, semantic split, and bandwidth extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class AdaptiveStubConfig:
    """Reserved for future adaptive FPS / BWE integration."""

    mode: str = "none"  # none | bwe_stub | fps_stub


def nominal_bitrate_multiplier(mode: str) -> float:
    """Optional nominal-bitrate bookkeeping multiplier (1.0 = no change)."""
    m = (mode or "none").strip().lower()
    if m == "bwe_stub":
        return 0.5  # conceptual halving when only low band is coded
    if m == "fps_stub":
        return 0.75
    return 1.0
