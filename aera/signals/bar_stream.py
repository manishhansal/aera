"""Per-symbol OHLC bar stream + swing-pivot detection.

A "bar" here is a fixed-duration aggregate of mid prints with inferred
per-bar volume (taker buys + taker sells) reconstructed from top-of-
book deltas, exactly like :class:`aera.signals.trade_tape.TradeTape`
does for individual prints. The buy / sell split lets downstream callers
read the bar's *delta* (signed aggressive flow), which is how the Stop
Hunt Reversal strategy distinguishes a genuine engineered sweep (delta
flips against the wick direction) from continuation chop.

Two complementary read paths:

* :meth:`BarStream.closed_bars` — list of completed bars in chronological
  order. The strategy snapshots this after each ``update`` to detect
  sweep candles against the recent swing high / low set.
* :meth:`BarStream.swing_pivots` — fractal pivot detection on the closed
  bars. A bar is a "swing high" when its high is strictly greater than
  ``pivot_strength`` bars on each side; mirror for "swing low". These
  are the canonical "key levels" the spec asks for.

Designed to be driven by the book-poll loop or the websocket feed (both
call ``update(book, now=...)`` from the strategy's ``scan``). Memory is
bounded by ``max_bars`` so per-symbol footprint stays small across many
symbols.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from aera.markets.orderbook import OrderBook


@dataclass
class Bar:
    """Single closed (or in-progress) OHLC bar.

    ``buy_volume`` / ``sell_volume`` are aggressive taker volumes
    inferred from book deltas during the bar. ``volume`` is their sum.
    ``delta`` is signed: ``buy_volume - sell_volume`` (> 0 = net buying
    pressure during the bar, < 0 = net selling). The spec uses this to
    confirm a bearish sweep ("delta flips red after the wick").
    """

    start: float
    end: float
    open: float
    high: float
    low: float
    close: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    @property
    def volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def range(self) -> float:
        return max(0.0, self.high - self.low)

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def body_ratio(self) -> float:
        """``body / range``. ``0`` when the bar has no range (1-print bar)."""
        r = self.range
        return (self.body / r) if r > 0 else 0.0

    @property
    def upper_wick(self) -> float:
        return max(0.0, self.high - max(self.open, self.close))

    @property
    def lower_wick(self) -> float:
        return max(0.0, min(self.open, self.close) - self.low)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


@dataclass
class BarStream:
    """Time-bucketed OHLC bars + swing-pivot accessors for one symbol.

    Parameters
    ----------
    bar_seconds : float
        Duration of each closed bar. Spec uses 1 s bars; longer durations
        give noisier candles with more body-vs-wick signal.
    max_bars : int
        Cap on the rolling buffer of closed bars. Bounded memory across
        many symbols. Spec's "3 recent swing highs/lows on 1m chart" with
        1 s bars means ~60 bars of lookback is plenty; default 300 covers
        5 minutes.
    inference_max_step_fraction : float
        Same noise guard as :class:`TradeTape` — drop one-tick book
        collapses larger than this fraction of the prior level (those
        typically reflect order cancellations, not trades). ``0``
        disables.
    """

    bar_seconds: float = 1.0
    max_bars: int = 300
    inference_max_step_fraction: float = 0.95

    _bars: Deque[Bar] = field(default_factory=deque)
    _cur: Optional[Bar] = None
    _cur_index: float = 0.0
    _last_bid_px: Optional[float] = None
    _last_bid_sz: Optional[float] = None
    _last_ask_px: Optional[float] = None
    _last_ask_sz: Optional[float] = None

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    def update(self, book: OrderBook, *, now: Optional[float] = None) -> Optional[Bar]:
        """Push a fresh top-of-book; rotate the bar if a boundary passed.

        Returns the bar that just closed (if this update crossed a bar
        boundary), or ``None`` otherwise. The newly-open bar is then
        seeded with the latest mid as its open / high / low / close.

        The taker buy / sell split is inferred from the change in best-
        bid / best-ask since the previous update (same heuristic as
        :class:`aera.signals.trade_tape.TradeTape`) and credited to
        the bar that's *currently open* (which contains ``now``).
        """
        t = now if now is not None else time.time()
        bid_lvl = book.best_bid()
        ask_lvl = book.best_ask()
        if bid_lvl is None or ask_lvl is None or ask_lvl.price <= 0:
            return None

        mid = 0.5 * (bid_lvl.price + ask_lvl.price)
        if mid <= 0:
            return None

        buy_vol, sell_vol = self._infer_taker_volumes(bid_lvl, ask_lvl)

        bar_seconds = max(1e-3, float(self.bar_seconds))
        bucket_index = t // bar_seconds
        bucket_start = bucket_index * bar_seconds
        bucket_end = bucket_start + bar_seconds

        closed: Optional[Bar] = None

        if self._cur is None:
            # cold start — open the first bar at this bucket
            self._cur = Bar(
                start=bucket_start,
                end=bucket_end,
                open=mid,
                high=mid,
                low=mid,
                close=mid,
            )
            self._cur_index = bucket_index
        elif bucket_index != self._cur_index:
            # bar boundary crossed — close the in-progress bar (apply
            # this tick's inferred flow to the OLD bar, since the flow
            # consumed liquidity that was resting before the boundary)
            self._cur.close = self._cur.close  # close already tracks last mid
            self._cur.buy_volume += max(0.0, buy_vol)
            self._cur.sell_volume += max(0.0, sell_vol)
            closed = self._cur
            self._bars.append(closed)
            while len(self._bars) > self.max_bars:
                self._bars.popleft()
            # open the new bar at the latest mid
            self._cur = Bar(
                start=bucket_start,
                end=bucket_end,
                open=mid,
                high=mid,
                low=mid,
                close=mid,
            )
            self._cur_index = bucket_index
        else:
            # same bar — update OHLC + accumulate volume
            if mid > self._cur.high:
                self._cur.high = mid
            if mid < self._cur.low:
                self._cur.low = mid
            self._cur.close = mid
            self._cur.buy_volume += max(0.0, buy_vol)
            self._cur.sell_volume += max(0.0, sell_vol)

        # Always refresh the last-observed snapshot — the next update
        # diffs against these to re-infer eaten liquidity.
        self._last_bid_px, self._last_bid_sz = bid_lvl.price, bid_lvl.size
        self._last_ask_px, self._last_ask_sz = ask_lvl.price, ask_lvl.size
        return closed

    def _infer_taker_volumes(self, bid_lvl, ask_lvl) -> Tuple[float, float]:
        """Estimate (taker_buy, taker_sell) sizes since the last update.

        Aggressive BUY = ask side ate (price held + size dropped, or
        price climbed = level fully cleared, charge prior size).
        Aggressive SELL = mirror on bids. Returns 0 on cold start or
        when a side improved (= new liquidity, not aggression).
        """
        buy_vol = 0.0
        sell_vol = 0.0

        if self._last_ask_px is not None and self._last_ask_sz is not None:
            if ask_lvl.price == self._last_ask_px and ask_lvl.size < self._last_ask_sz:
                eaten = self._last_ask_sz - ask_lvl.size
                if self._accept(eaten, self._last_ask_sz):
                    buy_vol = eaten
            elif ask_lvl.price > self._last_ask_px:
                buy_vol = max(0.0, self._last_ask_sz)

        if self._last_bid_px is not None and self._last_bid_sz is not None:
            if bid_lvl.price == self._last_bid_px and bid_lvl.size < self._last_bid_sz:
                eaten = self._last_bid_sz - bid_lvl.size
                if self._accept(eaten, self._last_bid_sz):
                    sell_vol = eaten
            elif bid_lvl.price < self._last_bid_px:
                sell_vol = max(0.0, self._last_bid_sz)

        return buy_vol, sell_vol

    def _accept(self, eaten: float, prior: Optional[float]) -> bool:
        """Same noise guard as :class:`TradeTape._accept`."""
        if eaten <= 0 or prior is None or prior <= 0:
            return False
        if self.inference_max_step_fraction <= 0:
            return True
        return (eaten / prior) <= self.inference_max_step_fraction

    # ------------------------------------------------------------------
    # read-only accessors
    # ------------------------------------------------------------------

    @property
    def closed_bars(self) -> List[Bar]:
        """Defensive copy of the closed-bar buffer (oldest first)."""
        return list(self._bars)

    @property
    def current_bar(self) -> Optional[Bar]:
        """In-progress bar (or ``None`` before the first update)."""
        return self._cur

    @property
    def last_closed(self) -> Optional[Bar]:
        return self._bars[-1] if self._bars else None

    def avg_volume(self, lookback: int) -> Optional[float]:
        """Mean per-bar volume over the trailing ``lookback`` closed bars.

        Returns ``None`` when the buffer hasn't accumulated enough bars
        yet (cold start) so callers don't divide by zero.
        """
        if lookback <= 0 or not self._bars:
            return None
        recent = list(self._bars)[-lookback:]
        if not recent:
            return None
        return sum(b.volume for b in recent) / len(recent)

    def swing_pivots(
        self,
        *,
        pivot_strength: int = 2,
        lookback_bars: int = 60,
    ) -> Tuple[List[Bar], List[Bar]]:
        """Detect ``(swing_highs, swing_lows)`` via fractal pivot rule.

        A closed bar at index ``i`` (within the lookback slice) is a
        swing high when its high is **strictly greater** than the highs
        of the ``pivot_strength`` bars on each side. Mirror for swing
        lows on the bar's low.

        Only the latest ``lookback_bars`` closed bars are inspected.
        Returns the pivots in chronological order, oldest first. The
        most recent ``pivot_strength`` bars cannot be pivots yet
        (they're not flanked on the right), so this is naturally
        confirmation-only — a level becomes a swing once the market
        has moved ``pivot_strength`` bars past it without exceeding.
        """
        if pivot_strength < 1 or not self._bars:
            return [], []
        recent = list(self._bars)[-max(1, lookback_bars):]
        if len(recent) < (2 * pivot_strength + 1):
            return [], []
        highs: List[Bar] = []
        lows: List[Bar] = []
        for i in range(pivot_strength, len(recent) - pivot_strength):
            window = recent[i - pivot_strength: i + pivot_strength + 1]
            center = recent[i]
            if all(center.high > b.high for j, b in enumerate(window) if j != pivot_strength):
                highs.append(center)
            if all(center.low < b.low for j, b in enumerate(window) if j != pivot_strength):
                lows.append(center)
        return highs, lows

    def recent_swing_highs(
        self,
        *,
        pivot_strength: int = 2,
        lookback_bars: int = 60,
        max_count: int = 3,
    ) -> List[float]:
        """The ``max_count`` most-recent swing-high *price levels*.

        Convenience wrapper over :meth:`swing_pivots` that returns the
        distinct high prices, newest first. Spec uses 3 levels.
        """
        highs, _ = self.swing_pivots(
            pivot_strength=pivot_strength,
            lookback_bars=lookback_bars,
        )
        out: List[float] = []
        for bar in reversed(highs):
            price = float(bar.high)
            if not any(abs(price - p) < 1e-12 for p in out):
                out.append(price)
            if len(out) >= max_count:
                break
        return out

    def recent_swing_lows(
        self,
        *,
        pivot_strength: int = 2,
        lookback_bars: int = 60,
        max_count: int = 3,
    ) -> List[float]:
        """The ``max_count`` most-recent swing-low *price levels*."""
        _, lows = self.swing_pivots(
            pivot_strength=pivot_strength,
            lookback_bars=lookback_bars,
        )
        out: List[float] = []
        for bar in reversed(lows):
            price = float(bar.low)
            if not any(abs(price - p) < 1e-12 for p in out):
                out.append(price)
            if len(out) >= max_count:
                break
        return out

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._bars.clear()
        self._cur = None
        self._cur_index = 0.0
        self._last_bid_px = self._last_bid_sz = None
        self._last_ask_px = self._last_ask_sz = None
