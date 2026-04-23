"""Empirical entropy and simple range-coding upper bounds for RVQ indices."""

from __future__ import annotations

import math
from typing import Sequence


def empirical_cross_entropy_bits_per_symbol(indices_flat: list[int], k: int) -> float:
    """``H(p̂)`` in bits/symbol from counts (natural estimator)."""
    if k < 2 or not indices_flat:
        return 0.0
    counts = [0] * k
    for i in indices_flat:
        if 0 <= int(i) < k:
            counts[int(i)] += 1
    n = float(len(indices_flat))
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    return h


def range_coded_size_upper_bound_bits(counts: Sequence[int], n_symbols: int) -> float:
    """Order-0 static range code length upper bound: ``sum -n log2(p_i)`` bits (ideal)."""
    n = int(n_symbols)
    if n <= 0:
        return 0.0
    tot = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / float(n)
        tot -= float(c) * math.log2(p)
    return tot
