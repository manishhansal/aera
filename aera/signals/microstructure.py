"""Microstructure features: rolling z-score, order-flow imbalance.

These features are the ones that empirically predict short-horizon prediction-
market price moves the best:

  * **Order Flow Imbalance (OFI)** — the net change in bid- vs. ask-side depth.
    A sudden jump in bid depth without a matching ask move is a strong signal
    that informed flow is on the buy side.

  * **Rolling z-score of mid** — the mean-reversion engine for micro-markets.
    Combined with a vol-implied fair value, it gives you a robust "buy when
    market is N sigma cheap" trigger.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class RollingZScore:
    window: int = 60
    _buf: Deque[float] = field(default_factory=deque)

    def update(self, x: float) -> Optional[float]:
        self._buf.append(x)
        while len(self._buf) > self.window:
            self._buf.popleft()
        if len(self._buf) < 5:
            return None
        n = len(self._buf)
        mean = sum(self._buf) / n
        var = sum((v - mean) ** 2 for v in self._buf) / n
        std = var**0.5
        if std == 0:
            return 0.0
        return (x - mean) / std


@dataclass
class OrderFlowImbalance:
    """Tracks (bid_depth - ask_depth) / (bid_depth + ask_depth) at top-of-book."""
    smoothing: float = 0.2          # EMA factor
    _ema: Optional[float] = None

    def update(self, bid_size: float, ask_size: float) -> float:
        total = bid_size + ask_size
        if total <= 0:
            return self._ema or 0.0
        raw = (bid_size - ask_size) / total
        if self._ema is None:
            self._ema = raw
        else:
            self._ema = self.smoothing * raw + (1 - self.smoothing) * self._ema
        return self._ema

    @property
    def value(self) -> float:
        return self._ema or 0.0
