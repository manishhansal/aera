"""Mean reversion z-score utilities."""
from __future__ import annotations

from typing import Optional, Sequence


def zscore_signal(prices: Sequence[float], current: float) -> Optional[float]:
    """Return the z-score of `current` against the trailing window `prices`."""
    if not prices or len(prices) < 5:
        return None
    n = len(prices)
    mean = sum(prices) / n
    var = sum((p - mean) ** 2 for p in prices) / n
    std = var**0.5
    if std == 0:
        return 0.0
    return (current - mean) / std
