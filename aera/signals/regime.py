"""Per-symbol market regime detection.

Classifies the current state of each market into one of a small set of
regimes so the :class:`AdaptiveBrain` can decide whether a given strategy
is *appropriate* for the current conditions:

* ``TREND_UP``     — strong upward drift; momentum/breakout strategies work
* ``TREND_DOWN``   — strong downward drift; momentum/breakout strategies work
* ``RANGE``        — low directional drift; mean-reversion strategies work
* ``HIGH_VOL``     — abnormally large bar-to-bar moves; reduce or skip
* ``NEWS_SPIKE``   — single-tick jump >> recent baseline; skip everything

The detector is intentionally light: a rolling window of mid prints fed
through a couple of EMA/ATR-style accumulators. No external libraries,
no candle aggregation (the bar-based strategies already do that). The
key insight is that you do not need a great regime classifier to lift
profitability — you only need *any* classifier that successfully refuses
to scalp during the worst 10% of conditions (news spikes, vol blowouts,
hard trends through mean-reversion levels).

Each :class:`RegimeDetector` instance keeps state for a single symbol;
the :class:`RegimeBook` wraps a dict of them keyed by ``market.id`` and
exposes a single ``observe()`` call from the engine loop.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional


class Regime(str, Enum):
    """Market regime label.

    Stored as a string Enum so it serialises cleanly through the dashboard
    state container.
    """

    UNKNOWN = "unknown"
    RANGE = "range"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    HIGH_VOL = "high_vol"
    NEWS_SPIKE = "news_spike"


@dataclass
class RegimeSnapshot:
    """Frozen view of a symbol's current regime estimate.

    Attributes
    ----------
    regime : Regime
        The classified label (most-restrictive wins: a news spike beats a
        trend, a trend beats range, etc.).
    trend_score : float
        Signed dimensionless score in roughly [-3, +3]. Positive = up.
    vol_ratio : float
        Short-window ATR / long-window ATR. > 1 means short-term is
        spikier than the baseline; > ``high_vol_ratio`` triggers HIGH_VOL.
    last_tick_bps : float
        Absolute size of the most recent mid move in basis points. Used
        for the NEWS_SPIKE gate.
    samples : int
        Number of mid observations the snapshot was computed from.
    """

    regime: Regime = Regime.UNKNOWN
    trend_score: float = 0.0
    vol_ratio: float = 1.0
    last_tick_bps: float = 0.0
    samples: int = 0

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "trend_score": float(self.trend_score),
            "vol_ratio": float(self.vol_ratio),
            "last_tick_bps": float(self.last_tick_bps),
            "samples": int(self.samples),
        }


@dataclass
class RegimeDetector:
    """Streaming regime classifier for a single symbol.

    Algorithm
    ---------
    On every ``observe(mid)`` call:

    1. Compute the per-tick return ``r = (mid - prev_mid) / prev_mid``.
    2. Update an EMA of ``r`` (drift) and an EMA of ``|r|`` (vol).
    3. Maintain short- and long-window ATR-style aggregates from the same
       per-tick |r| stream so we can ratio them for the vol regime.
    4. Compute the trend score as ``drift_ema / vol_ema`` — a Sharpe-like
       ratio that's >> 0 in clean trends and ≈ 0 in noisy chop.
    5. Classify:
       * NEWS_SPIKE if the latest |r| > ``news_tick_bps``,
       * HIGH_VOL  if vol_ratio > ``high_vol_ratio``,
       * TREND_UP  if trend_score >= ``trend_threshold``,
       * TREND_DOWN if trend_score <= ``-trend_threshold``,
       * RANGE otherwise.

    The detector is fully streaming (no full-buffer recompute) so each
    update is O(1). Memory is bounded by the two short/long deques.
    """

    short_window: int = 30
    long_window: int = 300
    trend_threshold: float = 0.30
    high_vol_ratio: float = 2.0
    news_tick_bps: float = 25.0
    drift_alpha: float = 0.05
    vol_alpha: float = 0.10
    _prev_mid: float = 0.0
    _drift_ema: float = 0.0
    _vol_ema: float = 0.0
    _short_abs: Deque[float] = field(default_factory=deque)
    _long_abs: Deque[float] = field(default_factory=deque)
    _last_tick_bps: float = 0.0
    _samples: int = 0

    def observe(self, mid: float) -> RegimeSnapshot:
        """Ingest a new mid print and return the current snapshot."""
        if mid <= 0:
            return self.snapshot()
        if self._prev_mid <= 0:
            self._prev_mid = mid
            self._samples += 1
            return self.snapshot()

        r = (mid - self._prev_mid) / self._prev_mid
        abs_r = abs(r)
        self._prev_mid = mid
        self._samples += 1
        self._last_tick_bps = abs_r * 1e4

        # EMAs of drift and vol (the drift one is *signed*, the vol one
        # is on |r|). Bootstrapping with the first observation keeps the
        # warm-up bias low.
        if self._samples == 2:
            self._drift_ema = r
            self._vol_ema = abs_r
        else:
            self._drift_ema = (
                self.drift_alpha * r + (1.0 - self.drift_alpha) * self._drift_ema
            )
            self._vol_ema = (
                self.vol_alpha * abs_r + (1.0 - self.vol_alpha) * self._vol_ema
            )

        self._short_abs.append(abs_r)
        while len(self._short_abs) > self.short_window:
            self._short_abs.popleft()
        self._long_abs.append(abs_r)
        while len(self._long_abs) > self.long_window:
            self._long_abs.popleft()

        return self.snapshot()

    def snapshot(self) -> RegimeSnapshot:
        """Return the latest classification without ingesting a new sample."""
        trend_score = 0.0
        if self._vol_ema > 0:
            trend_score = self._drift_ema / self._vol_ema

        short_atr = (
            sum(self._short_abs) / len(self._short_abs) if self._short_abs else 0.0
        )
        long_atr = (
            sum(self._long_abs) / len(self._long_abs) if self._long_abs else 0.0
        )
        vol_ratio = (short_atr / long_atr) if long_atr > 0 else 1.0

        # Classification ladder (most-restrictive first).
        if self._samples < 5:
            regime = Regime.UNKNOWN
        elif self._last_tick_bps >= self.news_tick_bps:
            regime = Regime.NEWS_SPIKE
        elif vol_ratio >= self.high_vol_ratio:
            regime = Regime.HIGH_VOL
        elif trend_score >= self.trend_threshold:
            regime = Regime.TREND_UP
        elif trend_score <= -self.trend_threshold:
            regime = Regime.TREND_DOWN
        else:
            regime = Regime.RANGE

        return RegimeSnapshot(
            regime=regime,
            trend_score=float(trend_score),
            vol_ratio=float(vol_ratio),
            last_tick_bps=float(self._last_tick_bps),
            samples=int(self._samples),
        )


class RegimeBook:
    """Dictionary of per-symbol :class:`RegimeDetector` instances.

    The engine calls :meth:`observe_markets` once per scan tick with the
    current ``Dict[str, Market]`` and the book lazily creates a detector
    for every previously-unseen symbol. Strategies / the AdaptiveBrain
    then ask for the current regime via :meth:`snapshot`.
    """

    def __init__(
        self,
        *,
        short_window: int = 30,
        long_window: int = 300,
        trend_threshold: float = 0.30,
        high_vol_ratio: float = 2.0,
        news_tick_bps: float = 25.0,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.trend_threshold = trend_threshold
        self.high_vol_ratio = high_vol_ratio
        self.news_tick_bps = news_tick_bps
        self._detectors: Dict[str, RegimeDetector] = {}

    def _detector_for(self, symbol: str) -> RegimeDetector:
        det = self._detectors.get(symbol)
        if det is None:
            det = RegimeDetector(
                short_window=self.short_window,
                long_window=self.long_window,
                trend_threshold=self.trend_threshold,
                high_vol_ratio=self.high_vol_ratio,
                news_tick_bps=self.news_tick_bps,
            )
            self._detectors[symbol] = det
        return det

    def observe_markets(self, markets) -> None:
        """Ingest the mid of every market into its detector."""
        if not markets:
            return
        for m in markets.values() if isinstance(markets, dict) else markets:
            outcome = next(iter(m.outcomes.values()), None)
            if outcome is None:
                continue
            bid = outcome.best_bid
            ask = outcome.best_ask
            if bid is None or ask is None or ask <= 0:
                continue
            mid = 0.5 * (bid + ask)
            if mid <= 0 or not math.isfinite(mid):
                continue
            self._detector_for(m.id).observe(mid)

    def snapshot(self, symbol: str) -> RegimeSnapshot:
        det = self._detectors.get(symbol)
        if det is None:
            return RegimeSnapshot()
        return det.snapshot()

    def snapshots(self) -> Dict[str, RegimeSnapshot]:
        return {sym: d.snapshot() for sym, d in self._detectors.items()}
