"""L2 order-book microstructure helpers for the Order Book Sniper.

Three pieces:

* :class:`DepthImbalanceSnapshot` and :func:`measure_depth_imbalance` —
  cumulative bid/ask size within ±N bps of mid, plus the bid/ask ratios.
* :class:`TapeInferrer` — infers aggressive taker buys/sells from
  successive top-of-book snapshots (so the sniper can do tape-confirmation
  without subscribing to a separate trades feed).
* :class:`WallTracker` — records the resting wall present at entry time
  and detects whether it persists for a configurable window (spoofing
  defense — exit at market if the wall vanishes within N seconds of entry).

These helpers are deliberately stateless / per-symbol so the strategy can
own one instance per market and the tests can drive them deterministically
without monkey-patching the clock.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from aera.markets.orderbook import OrderBook


@dataclass
class DepthImbalanceSnapshot:
    """Cumulative bid/ask depth measurement within ±``band_bps`` of mid.

    ``ratio = bid_size / ask_size`` (or ``+inf`` when ``ask_size == 0``).
    ``inverse_ratio = ask_size / bid_size`` is the mirror for short setups.
    """

    bid_size: float
    ask_size: float
    ratio: float
    inverse_ratio: float
    mid: float
    band_bps: float

    @property
    def bull(self) -> bool:
        """True when bid depth dominates ask depth in the band."""
        return self.bid_size > self.ask_size

    @property
    def bear(self) -> bool:
        return self.ask_size > self.bid_size


def measure_depth_imbalance(
    book: OrderBook,
    *,
    band_bps: float = 10.0,
    max_levels: int = 10,
) -> Optional[DepthImbalanceSnapshot]:
    """Sum sizes on each side within ±``band_bps`` of mid (top ``max_levels``).

    ``band_bps`` is in basis points: 10.0 → ±0.1% of mid, matching the
    Order Book Sniper spec. Levels outside the band are excluded from both
    totals so a thick wall five percent away never inflates the ratio.

    Returns ``None`` when the book is one-sided or empty — the strategy
    should treat that the same as "no signal" and move on.
    """
    bid = book.best_bid_price()
    ask = book.best_ask_price()
    if bid is None or ask is None or ask <= 0:
        return None
    mid = 0.5 * (bid + ask)
    if mid <= 0:
        return None
    band = mid * (band_bps / 1e4)
    bid_floor = mid - band
    ask_ceil = mid + band

    bids = book.bids_sorted()[: max(1, max_levels)]
    asks = book.asks_sorted()[: max(1, max_levels)]
    bid_size = sum(l.size for l in bids if l.price >= bid_floor)
    ask_size = sum(l.size for l in asks if l.price <= ask_ceil)
    if bid_size <= 0 and ask_size <= 0:
        return None

    eps = 1e-9
    ratio = bid_size / ask_size if ask_size > eps else float("inf")
    inverse = ask_size / bid_size if bid_size > eps else float("inf")
    return DepthImbalanceSnapshot(
        bid_size=bid_size,
        ask_size=ask_size,
        ratio=ratio,
        inverse_ratio=inverse,
        mid=mid,
        band_bps=band_bps,
    )


@dataclass
class TapeInferrer:
    """Infers aggressive market buys/sells from successive top-of-book frames.

    Without a separate trades-channel subscription we can still get a
    reasonable read on aggressive flow by diffing top-of-book between scan
    ticks:

    * An **aggressive buy** eats the resting ask. Best-ask SIZE shrinks
      while best-ask PRICE stays the same (or rises if the level was fully
      eaten and a higher level becomes top). New offers improving the ask
      drop the price; we deliberately don't count that as a buy.
    * Mirror logic for **aggressive sells** on the bid side.

    Events are timestamped and kept in a sliding window so callers can ask
    "how many taker buys in the last 2 seconds?" — the standard
    tape-confirmation pattern used by the sniper.

    The heuristic is intentionally conservative: when a level appears or
    withdraws by a large amount in a single tick we treat that as
    quote-revision noise and skip it (``max_step_fraction`` guard) so a
    spoofing pull doesn't masquerade as 1000 aggressive buys.
    """

    window_seconds: float = 2.0
    # If a single tick's "eaten" size is more than ``max_step_fraction`` of
    # the previous level, treat it as cancellation noise rather than a
    # market order. 0 disables the guard.
    max_step_fraction: float = 0.95

    _last_bid_px: Optional[float] = None
    _last_bid_sz: Optional[float] = None
    _last_ask_px: Optional[float] = None
    _last_ask_sz: Optional[float] = None
    _buys: Deque[Tuple[float, float]] = field(default_factory=deque)
    _sells: Deque[Tuple[float, float]] = field(default_factory=deque)

    def update(
        self,
        book: OrderBook,
        *,
        now: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Ingest a fresh book; return ``(taker_buys, taker_sells)`` in window.

        The returned counts are the *events* in the last ``window_seconds``
        (not the cumulative size eaten). Counts are what the spec asks for
        ("3+ consecutive aggressive market buys").
        """
        t = now if now is not None else time.time()
        bid_lvl = book.best_bid()
        ask_lvl = book.best_ask()
        if bid_lvl is None or ask_lvl is None:
            return self._window_counts(t)

        bid_px, bid_sz = bid_lvl.price, bid_lvl.size
        ask_px, ask_sz = ask_lvl.price, ask_lvl.size

        # Aggressive taker BUY: ask shrank without the level improving.
        # An improving ask (price moved down) means new sellers showed up;
        # that is the opposite signal we want to record.
        if (
            self._last_ask_px is not None
            and self._last_ask_sz is not None
            and ask_px >= self._last_ask_px
            and ask_sz < self._last_ask_sz
        ):
            eaten = self._last_ask_sz - ask_sz
            if self._accept(eaten, self._last_ask_sz):
                self._buys.append((t, eaten))

        # Aggressive taker SELL: bid shrank without the level improving.
        if (
            self._last_bid_px is not None
            and self._last_bid_sz is not None
            and bid_px <= self._last_bid_px
            and bid_sz < self._last_bid_sz
        ):
            eaten = self._last_bid_sz - bid_sz
            if self._accept(eaten, self._last_bid_sz):
                self._sells.append((t, eaten))

        self._last_bid_px, self._last_bid_sz = bid_px, bid_sz
        self._last_ask_px, self._last_ask_sz = ask_px, ask_sz
        return self._window_counts(t)

    def _accept(self, eaten: float, prior: float) -> bool:
        """Filter out one-tick-vanish events that look like cancellations."""
        if eaten <= 0 or prior <= 0:
            return False
        if self.max_step_fraction <= 0:
            return True
        return (eaten / prior) <= self.max_step_fraction

    def _window_counts(self, now: float) -> Tuple[int, int]:
        cutoff = now - self.window_seconds
        while self._buys and self._buys[0][0] < cutoff:
            self._buys.popleft()
        while self._sells and self._sells[0][0] < cutoff:
            self._sells.popleft()
        return len(self._buys), len(self._sells)

    def reset(self) -> None:
        """Forget last-tick state and clear the rolling event buffer."""
        self._last_bid_px = self._last_bid_sz = None
        self._last_ask_px = self._last_ask_sz = None
        self._buys.clear()
        self._sells.clear()


@dataclass
class WallSnapshot:
    """The resting wall observed at entry time on the favoured side.

    ``side`` is ``"BID"`` for a long entry, ``"ASK"`` for a short. ``price``
    pins the level; ``size`` is the size at signal-fire time. ``observed_at``
    is the wall-clock timestamp used for the persistence window.
    """

    side: str
    price: float
    size: float
    observed_at: float

    def current_size(self, book: OrderBook) -> float:
        """Read the size at ``price`` on ``side`` from a fresh ``book``.

        Returns 0.0 if the level has been pulled entirely. Reads from the
        raw dict so we don't allocate a sorted ladder just to look up one
        price (the sniper queries this on every tick).
        """
        levels = book.bids if self.side == "BID" else book.asks
        return float(levels.get(self.price, 0.0))

    def vanished(
        self,
        book: OrderBook,
        *,
        ratio_threshold: float = 0.5,
        now: Optional[float] = None,
        persist_seconds: float = 1.0,
    ) -> bool:
        """True when the wall has shrunk past the spoofing threshold in window.

        The check only fires while ``now − observed_at <= persist_seconds``;
        outside that window the strategy is on its own time-based exit, and
        residual wall shrinkage is just normal book evolution rather than
        spoofing.
        """
        t = now if now is not None else time.time()
        if t - self.observed_at > persist_seconds:
            return False
        remaining = self.current_size(book)
        if self.size <= 0:
            return False
        return remaining <= self.size * (1.0 - ratio_threshold)
