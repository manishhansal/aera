"""Per-symbol micro-VWAP stream for the Micro VWAP Reversion Sniper.

A rolling, time-windowed VWAP that tracks ``Σ(price × volume) / Σ(volume)``
over the trailing ``window_seconds`` (default 60 s). Volume is sourced
the same way :class:`TradeTape` sources it — by inferring aggressive
trades from successive top-of-book snapshots — so the stream works on a
book-only feed without needing a separate trades-channel subscription.

The stream exposes two read patterns the strategy needs:

* :meth:`VWAPStream.vwap` — VWAP over a configurable time window. Used
  to compute the "deviation from micro-VWAP" trigger and to snapshot the
  VWAP-at-entry static target.
* :meth:`VWAPStream.volume_ratio` — short-window per-second volume rate
  divided by a long-window rate. Used to filter for the "volume drop-off
  while price extends" pattern the spec describes — exhausted sellers
  (the opposite of the spike conditions the other scalpers veto on).

Like :class:`TradeTape`, the buffer is capped (``max_trades``) so memory
stays bounded across many symbols. All read accessors are non-mutating.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from aera.markets.orderbook import OrderBook


@dataclass
class _VolumeSample:
    """One inferred taker print.

    ``side`` is recorded for symmetry with :class:`Trade` even though
    the VWAP math doesn't care about aggressor side — useful for tests
    that want to assert balanced ingestion.
    """

    timestamp: float
    price: float
    size: float
    side: str


@dataclass
class VWAPStream:
    """Rolling micro-VWAP + volume-rate accessors for one symbol.

    Parameters
    ----------
    window_seconds : float
        Default VWAP window in seconds. Spec asks for 60 s (a 1-minute
        rolling VWAP). Callers can override per query via ``vwap``.
    max_trades : int
        Hard cap on the rolling sample buffer. The default (5000) covers
        a full 5-minute volume window even on chatty perps. Memory ≈
        ``max_trades × 48 bytes`` per symbol.
    inference_max_step_fraction : float
        Mirror of :class:`TradeTape`'s noise filter. Drops one-tick size
        collapses larger than this fraction of the prior level (usually
        an order cancellation, not a trade). ``0`` disables. ``0.95``
        accepts up to 95% of the prior level eaten in a single tick.
    """

    window_seconds: float = 60.0
    max_trades: int = 5000
    inference_max_step_fraction: float = 0.95

    _samples: Deque[_VolumeSample] = field(default_factory=deque)
    _last_bid_px: Optional[float] = None
    _last_bid_sz: Optional[float] = None
    _last_ask_px: Optional[float] = None
    _last_ask_sz: Optional[float] = None
    _last_mid: Optional[float] = None

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    def update(self, book: OrderBook, *, now: Optional[float] = None) -> int:
        """Ingest a fresh top-of-book snapshot.

        Infers aggressive trades from the diff against the last snapshot
        — same heuristic as :meth:`TradeTape.infer_from_book` — and
        appends one ``_VolumeSample`` per inferred event. A single tick
        can produce up to two samples (one BUY + one SELL when liquidity
        is taken from both sides simultaneously).

        Returns the number of samples appended this call (handy for
        instrumentation / tests).
        """
        t = now if now is not None else time.time()
        bid_lvl = book.best_bid()
        ask_lvl = book.best_ask()
        if bid_lvl is None or ask_lvl is None or ask_lvl.price <= 0:
            return 0

        bid_px, bid_sz = bid_lvl.price, bid_lvl.size
        ask_px, ask_sz = ask_lvl.price, ask_lvl.size
        appended = 0

        # Aggressive BUY: ask consumed. Either size shrank at the same
        # price (apply noise filter) or the prior level cleared and the
        # ask price climbed. New offers improving the ask aren't a trade.
        if self._last_ask_px is not None and self._last_ask_sz is not None:
            if ask_px == self._last_ask_px and ask_sz < self._last_ask_sz:
                eaten = self._last_ask_sz - ask_sz
                if self._accept(eaten, self._last_ask_sz):
                    self._append(t, self._last_ask_px, eaten, "BUY")
                    appended += 1
            elif ask_px > self._last_ask_px:
                eaten = self._last_ask_sz
                if eaten > 0:
                    self._append(t, self._last_ask_px, eaten, "BUY")
                    appended += 1

        # Aggressive SELL: bid consumed. Mirror of the BUY branch above.
        if self._last_bid_px is not None and self._last_bid_sz is not None:
            if bid_px == self._last_bid_px and bid_sz < self._last_bid_sz:
                eaten = self._last_bid_sz - bid_sz
                if self._accept(eaten, self._last_bid_sz):
                    self._append(t, self._last_bid_px, eaten, "SELL")
                    appended += 1
            elif bid_px < self._last_bid_px:
                eaten = self._last_bid_sz
                if eaten > 0:
                    self._append(t, self._last_bid_px, eaten, "SELL")
                    appended += 1

        # Always refresh the snapshot so the next tick has fresh sizes
        # to diff against (mirrors TradeTape semantics).
        self._last_bid_px, self._last_bid_sz = bid_px, bid_sz
        self._last_ask_px, self._last_ask_sz = ask_px, ask_sz
        self._last_mid = 0.5 * (bid_px + ask_px)
        return appended

    def record(
        self,
        *,
        price: float,
        size: float,
        side: str,
        now: Optional[float] = None,
    ) -> bool:
        """Push a trade directly (for tests / real trades-channel feeds)."""
        if size <= 0 or price <= 0:
            return False
        side_norm = side.upper()
        if side_norm not in ("BUY", "SELL"):
            return False
        self._append(
            now if now is not None else time.time(),
            float(price),
            float(size),
            side_norm,
        )
        return True

    def _append(self, ts: float, price: float, size: float, side: str) -> None:
        self._samples.append(_VolumeSample(ts, price, size, side))
        while len(self._samples) > self.max_trades:
            self._samples.popleft()

    def _accept(self, eaten: float, prior: Optional[float]) -> bool:
        if eaten <= 0 or prior is None or prior <= 0:
            return False
        if self.inference_max_step_fraction <= 0:
            return True
        return (eaten / prior) <= self.inference_max_step_fraction

    # ------------------------------------------------------------------
    # read-only accessors
    # ------------------------------------------------------------------

    @property
    def total_count(self) -> int:
        return len(self._samples)

    @property
    def last_mid(self) -> Optional[float]:
        return self._last_mid

    def vwap(
        self,
        window_seconds: Optional[float] = None,
        *,
        now: Optional[float] = None,
    ) -> Optional[float]:
        """Volume-weighted average price over the trailing window.

        Returns ``None`` when no samples exist in the window — strategies
        must guard against this on cold start. The window defaults to
        ``self.window_seconds``; pass an explicit value to query a
        different horizon (e.g. a 5-minute VWAP for context).
        """
        if not self._samples:
            return None
        w = window_seconds if window_seconds is not None else self.window_seconds
        if w <= 0:
            return None
        t = now if now is not None else self._samples[-1].timestamp
        cutoff = t - w
        num = 0.0
        denom = 0.0
        for s in self._samples:
            if s.timestamp < cutoff:
                continue
            num += s.price * s.size
            denom += s.size
        if denom <= 0:
            return None
        return num / denom

    def volume_in_window(
        self,
        seconds: float,
        *,
        now: Optional[float] = None,
    ) -> float:
        """Sum of inferred sizes in the last ``seconds`` of the buffer."""
        if not self._samples or seconds <= 0:
            return 0.0
        t = now if now is not None else self._samples[-1].timestamp
        cutoff = t - seconds
        return sum(s.size for s in self._samples if s.timestamp >= cutoff)

    def volume_ratio(
        self,
        *,
        short_seconds: float,
        long_seconds: float,
        now: Optional[float] = None,
    ) -> Optional[float]:
        """``short_rate / long_rate`` where rate = volume / window.

        Returns ``None`` when the long window has no recorded volume
        (cold start) — the caller can then treat "ratio unknown" as a
        soft veto (we don't fire entries without baseline context).
        Values < 1 mean the short window is quieter than the long
        baseline — the "volume drop-off" the spec wants to see.
        """
        if short_seconds <= 0 or long_seconds <= 0 or long_seconds <= short_seconds:
            return None
        long_vol = self.volume_in_window(long_seconds, now=now)
        if long_vol <= 0:
            return None
        short_vol = self.volume_in_window(short_seconds, now=now)
        short_rate = short_vol / short_seconds
        long_rate = long_vol / long_seconds
        if long_rate <= 0:
            return None
        return short_rate / long_rate

    def reset(self) -> None:
        self._samples.clear()
        self._last_bid_px = self._last_bid_sz = None
        self._last_ask_px = self._last_ask_sz = None
        self._last_mid = None
