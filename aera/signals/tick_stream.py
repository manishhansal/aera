"""Per-symbol tick stream for tick-exhaustion / reversal strategies.

A "tick" here is any scan iteration where the mid price changes; the
inferred direction is ``+1`` for an uptick and ``−1`` for a downtick. The
"size" of a tick is the amount of liquidity that disappeared from the
leading side of the book — that's the closest proxy to "trade size on
that tick" we can get without a separate trades-channel subscription.

The buffer is deliberately small (default 200 ticks) so the per-symbol
memory footprint stays bounded even when 50+ symbols are streamed. All
windowed queries (volume rate, news-spike, S/R extreme) walk the buffer
linearly; that's O(n) per scan but n is tiny.

Designed to be driven by either the REST-poll loop or the websocket book
feed — both ultimately call ``TickStream.update(book, now=...)`` from
the strategy's ``scan`` method.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from aera.markets.orderbook import OrderBook


@dataclass
class Tick:
    """One observed mid-price change.

    Recorded only when the mid actually moves between successive
    ``TickStream.update`` calls — flat scans do not produce ticks. This
    means the tick rate naturally tracks market activity rather than the
    bot's scan cadence.

    ``prev_mid`` captures the immediately-preceding mid (including the
    cold-start mid set on the very first scan), so windowed accessors
    like :meth:`TickStream.max_tick_move_bps` can compute a per-tick
    magnitude even for the buffer's first entry. Without it the first
    recorded tick's move size is lost (there's nothing in the buffer to
    diff against), which would silently invalidate the news-spike
    filter on any session whose very first observation IS a spike.
    """

    timestamp: float
    mid: float
    direction: int          # +1 (up), −1 (down). 0 ticks are not stored.
    size: float             # liquidity eaten on the leading side (>= 0)
    spread: float
    bid_size: float
    ask_size: float
    prev_mid: float = 0.0


@dataclass
class TickStream:
    """Rolling tick history + windowed feature accessors for one symbol.

    All windowed queries (``volume_in_window``, ``max_tick_move_bps``,
    ``recent_extreme``) read the buffer as it stands; they never mutate
    state. Callers can therefore safely poll them multiple times per scan
    without altering the inferred streak.

    The spread EMA is the only continuously-updated derived field — it's
    smoothed across every update so a single wide-spread tick doesn't
    blow the "spread > N × normal" filter on the next scan.
    """

    max_ticks: int = 200
    spread_ema_alpha: float = 0.05
    volume_short_window_seconds: float = 5.0
    volume_long_window_seconds: float = 60.0

    _ticks: Deque[Tick] = field(default_factory=deque)
    _last_mid: Optional[float] = None
    _last_bid_px: Optional[float] = None
    _last_bid_sz: Optional[float] = None
    _last_ask_px: Optional[float] = None
    _last_ask_sz: Optional[float] = None
    _spread_ema: Optional[float] = None

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    def update(self, book: OrderBook, *, now: Optional[float] = None) -> Optional[Tick]:
        """Ingest a fresh book; return a new Tick if mid moved, else None.

        Always updates the spread EMA — even on a flat tick we want to
        smooth the prevailing spread so the filter has a baseline once
        the market starts moving.
        """
        t = now if now is not None else time.time()
        bid_lvl = book.best_bid()
        ask_lvl = book.best_ask()
        if bid_lvl is None or ask_lvl is None or ask_lvl.price <= 0:
            return None

        mid = 0.5 * (bid_lvl.price + ask_lvl.price)
        spread = ask_lvl.price - bid_lvl.price
        if self._spread_ema is None:
            self._spread_ema = spread
        else:
            self._spread_ema = (
                self.spread_ema_alpha * spread
                + (1.0 - self.spread_ema_alpha) * self._spread_ema
            )

        tick: Optional[Tick] = None
        if self._last_mid is not None and mid != self._last_mid:
            direction = 1 if mid > self._last_mid else -1
            eaten = self._infer_eaten_size(direction, bid_lvl, ask_lvl)
            tick = Tick(
                timestamp=t,
                mid=mid,
                direction=direction,
                size=max(0.0, eaten),
                spread=spread,
                bid_size=bid_lvl.size,
                ask_size=ask_lvl.size,
                prev_mid=self._last_mid,
            )
            self._ticks.append(tick)
            while len(self._ticks) > self.max_ticks:
                self._ticks.popleft()

        # Always refresh the last-observed snapshot — even on a flat tick
        # so the next directional change has the latest sizes to compare
        # against when inferring eaten liquidity.
        self._last_mid = mid
        self._last_bid_px, self._last_bid_sz = bid_lvl.price, bid_lvl.size
        self._last_ask_px, self._last_ask_sz = ask_lvl.price, ask_lvl.size
        return tick

    def _infer_eaten_size(self, direction: int, bid_lvl, ask_lvl) -> float:
        """Approximate the size eaten on the leading side this tick.

        Uptick — asks were consumed. Two cases:
            * Ask PRICE held but SIZE dropped: ``eaten = prior − current``.
            * Ask PRICE rose: prior best level was likely cleared
              entirely; charge ``prior_size`` as the eaten amount (a
              conservative upper bound — the actual trade may have been
              larger and crossed multiple levels, but we lack the data).

        Symmetric for downticks on the bid side. Returns ``0.0`` if we
        can't make a sensible inference (e.g. price improved, meaning new
        liquidity entered the leading side rather than being consumed).
        """
        if direction > 0:
            if self._last_ask_px is None or self._last_ask_sz is None:
                return 0.0
            if ask_lvl.price == self._last_ask_px:
                return max(0.0, self._last_ask_sz - ask_lvl.size)
            if ask_lvl.price > self._last_ask_px:
                return float(self._last_ask_sz)
            return 0.0  # ask improved (price dropped) — no eat
        else:
            if self._last_bid_px is None or self._last_bid_sz is None:
                return 0.0
            if bid_lvl.price == self._last_bid_px:
                return max(0.0, self._last_bid_sz - bid_lvl.size)
            if bid_lvl.price < self._last_bid_px:
                return float(self._last_bid_sz)
            return 0.0

    # ------------------------------------------------------------------
    # streak detection
    # ------------------------------------------------------------------

    def current_streak(self) -> Tuple[int, int, List[float]]:
        """Return ``(direction, length, sizes)`` of the trailing run.

        ``direction`` is ``+1`` or ``−1``; ``length`` is how many
        same-direction ticks end the buffer; ``sizes`` is the per-tick
        eaten sizes in chronological order. Returns ``(0, 0, [])`` if
        the buffer is empty.
        """
        if not self._ticks:
            return (0, 0, [])
        direction = self._ticks[-1].direction
        sizes: List[float] = []
        length = 0
        for tick in reversed(self._ticks):
            if tick.direction != direction:
                break
            sizes.insert(0, tick.size)
            length += 1
        return (direction, length, sizes)

    @staticmethod
    def size_decay(sizes: List[float]) -> float:
        """Total fractional decay across a streak's sizes.

        Returns ``1 − last / first``. A return >= ``threshold`` means
        the streak's per-tick size has dropped by at least ``threshold``
        end-to-end (e.g. ``0.20`` = 20% smaller). ``0.0`` when not
        applicable (fewer than 2 ticks or non-positive first size).
        """
        if len(sizes) < 2:
            return 0.0
        first = sizes[0]
        last = sizes[-1]
        if first <= 0:
            return 0.0
        return max(0.0, 1.0 - (last / first))

    # ------------------------------------------------------------------
    # windowed accessors used by entry filters
    # ------------------------------------------------------------------

    def recent_extreme(self, direction: int, lookback: int) -> Optional[float]:
        """Local extreme over the last ``lookback`` ticks.

        ``direction`` ``+1`` returns the max mid (resistance proxy for a
        short setup); ``−1`` returns the min mid (support proxy for a
        long setup). Returns ``None`` if the buffer is empty.
        """
        if not self._ticks or lookback <= 0:
            return None
        slice_ = list(self._ticks)[-lookback:]
        mids = [t.mid for t in slice_]
        return max(mids) if direction > 0 else min(mids)

    @property
    def spread_ema(self) -> Optional[float]:
        return self._spread_ema

    def current_spread(self) -> Optional[float]:
        return self._ticks[-1].spread if self._ticks else None

    def current_spread_multiple(self) -> Optional[float]:
        """Most recent spread divided by the smoothed average spread."""
        sp = self.current_spread()
        if sp is None or self._spread_ema is None or self._spread_ema <= 0:
            return None
        return sp / self._spread_ema

    def volume_in_window(
        self, seconds: float, *, now: Optional[float] = None
    ) -> float:
        """Sum of ``Tick.size`` for ticks newer than ``now − seconds``."""
        if not self._ticks or seconds <= 0:
            return 0.0
        t = now if now is not None else self._ticks[-1].timestamp
        cutoff = t - seconds
        return sum(tick.size for tick in self._ticks if tick.timestamp >= cutoff)

    def volume_spike_ratio(
        self,
        *,
        short_seconds: Optional[float] = None,
        long_seconds: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Optional[float]:
        """Short-window volume RATE divided by long-window volume rate.

        Compares per-second eaten volume in the trailing
        ``short_seconds`` to the same metric over ``long_seconds``. A
        value of ``5.0`` means "the last 5 s is moving 5× faster than
        the 60 s baseline" — typical news-event behavior the spec wants
        to filter out. Returns ``None`` if the long window has no
        recorded volume (i.e. cold start).
        """
        ss = short_seconds or self.volume_short_window_seconds
        ls = long_seconds or self.volume_long_window_seconds
        if ls <= ss or ss <= 0:
            return None
        long_vol = self.volume_in_window(ls, now=now)
        if long_vol <= 0:
            return None
        short_vol = self.volume_in_window(ss, now=now)
        short_rate = short_vol / ss
        long_rate = long_vol / ls
        if long_rate <= 0:
            return None
        return short_rate / long_rate

    def max_tick_move_bps(
        self, lookback_seconds: float, *, now: Optional[float] = None
    ) -> float:
        """Largest single-tick |mid change| in bps over the window.

        Used as the news-spike proxy: a single tick > 50 bps is a
        flash-crash / news signature, not a normal scalp setup.
        Reads ``Tick.prev_mid`` so the buffer's first entry is included
        — without that the first recorded tick's magnitude would be
        silently dropped (it has no preceding buffer entry to diff).
        """
        if not self._ticks or lookback_seconds <= 0:
            return 0.0
        t = now if now is not None else self._ticks[-1].timestamp
        cutoff = t - lookback_seconds
        max_move = 0.0
        for tick in self._ticks:
            if tick.timestamp < cutoff:
                continue
            if tick.prev_mid > 0:
                move = abs(tick.mid - tick.prev_mid) / tick.prev_mid * 1e4
                if move > max_move:
                    max_move = move
        return max_move

    def depth_trend(self, side: str, *, lookback: int) -> int:
        """Return +1/0/−1 for the bid (or ask) depth trend.

        Compares ``side`` size at the most recent tick vs. ``lookback``
        ticks earlier. ``+1`` = depth grew, ``−1`` = depth shrank, ``0``
        = flat or not enough history. ``side`` is ``"bid"`` or
        ``"ask"``.
        """
        if len(self._ticks) < max(2, lookback):
            return 0
        recent = list(self._ticks)[-lookback:]
        if not recent:
            return 0
        start = recent[0].bid_size if side == "bid" else recent[0].ask_size
        end = recent[-1].bid_size if side == "bid" else recent[-1].ask_size
        if end > start:
            return 1
        if end < start:
            return -1
        return 0

    # ------------------------------------------------------------------
    # housekeeping
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> List[Tick]:
        """Defensive copy of the rolling buffer (for tests / debug)."""
        return list(self._ticks)

    def reset(self) -> None:
        self._ticks.clear()
        self._last_mid = None
        self._last_bid_px = self._last_bid_sz = None
        self._last_ask_px = self._last_ask_sz = None
        self._spread_ema = None
