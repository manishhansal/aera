"""Per-symbol trade tape for taker-flow strategies.

A "tape" here is a rolling buffer of aggressive trades (market orders
that lifted the ask or hit the bid). The tape exposes two complementary
ingestion paths so a strategy can run against either a real trades feed
or a book-only feed:

* :meth:`TradeTape.record` — push a trade with explicit
  ``(price, size, side)``. Used when the venue exposes a separate
  trades channel (e.g. Delta's ``all_trades`` websocket).
* :meth:`TradeTape.infer_from_book` — derive trade events from
  successive top-of-book snapshots. Identical heuristic to the existing
  :class:`aera.signals.order_book.TapeInferrer`, but stores the
  *size* of each inferred trade so downstream callers can run the
  "whale ≥ N× average trade size" math the Flow Scalp strategy needs.

The tape's read API is built around two patterns:

* ``avg_size(n)`` — rolling N-trade mean size, the baseline against
  which "whale" prints are compared.
* ``latest_whale(...)`` / ``count_aggressive_since(...)`` — the
  detection / confirmation primitives used by the flow scalper.

All windowed queries are read-only; multiple strategies can share one
tape per symbol without stepping on each other.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from aera.markets.orderbook import OrderBook


@dataclass
class Trade:
    """One aggressive (taker) trade.

    ``side`` is the aggressor side:
        * ``"BUY"``  — taker lifted the ask (paid the offer)
        * ``"SELL"`` — taker hit the bid (sold into the bid)

    ``size`` is in the venue's native size unit (contracts on Delta).
    For inverse perps, 1 contract = $1 of underlying notional, so the
    "average trade size" comparison is meaningful as long as the unit
    is consistent for the symbol.
    """

    timestamp: float
    price: float
    size: float
    side: str


@dataclass
class TradeTape:
    """Rolling tape of aggressive trades + whale-detection helpers.

    Designed for the "Tape Reading Momentum (Flow Scalp)" strategy:
    detect a single taker trade ≥ ``whale_multiple`` × average trade
    size, then look for a same-direction confirmation within a short
    window before entering.

    Parameters
    ----------
    max_trades : int
        Cap on the rolling buffer. Bounded memory across many symbols.
    avg_window : int
        How many most-recent trades to average when computing the
        "average trade size" baseline. Spec default = 100.
    inference_max_step_fraction : float
        Same guard as :class:`TapeInferrer` — drop one-tick size
        collapses larger than this fraction of the prior level, which
        usually mean a level was cancelled rather than traded through.
        ``0`` disables the guard. ``0.95`` accepts up to 95%.
    """

    max_trades: int = 500
    avg_window: int = 100
    inference_max_step_fraction: float = 0.95

    _trades: Deque[Trade] = field(default_factory=deque)
    _last_bid_px: Optional[float] = None
    _last_bid_sz: Optional[float] = None
    _last_ask_px: Optional[float] = None
    _last_ask_sz: Optional[float] = None
    _last_mid: Optional[float] = None

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        price: float,
        size: float,
        side: str,
        now: Optional[float] = None,
    ) -> Optional[Trade]:
        """Append a single taker trade. ``size`` must be > 0.

        Returns the recorded :class:`Trade` (handy for tests / callers
        that want to log the event), or ``None`` if the trade was
        rejected as malformed.
        """
        if size <= 0 or price <= 0:
            return None
        side_norm = side.upper()
        if side_norm not in ("BUY", "SELL"):
            return None
        t = now if now is not None else time.time()
        trade = Trade(timestamp=t, price=float(price), size=float(size), side=side_norm)
        self._trades.append(trade)
        while len(self._trades) > self.max_trades:
            self._trades.popleft()
        return trade

    def infer_from_book(
        self,
        book: OrderBook,
        *,
        now: Optional[float] = None,
    ) -> List[Trade]:
        """Derive taker trade(s) from a fresh top-of-book snapshot.

        Mirrors the heuristic used by
        :class:`aera.signals.order_book.TapeInferrer`, but stores
        the size of every inferred trade so the avg-size baseline math
        works:

            * **Aggressive BUY** — ask size shrinks while ask price
              holds (someone took part of the offer at the same level)
              or ask price climbs (the prior best level was cleared,
              charge ``last_ask_sz`` as a conservative trade size).
            * **Aggressive SELL** — mirror on the bid.

        Both events can fire on a single tick (a tick that takes
        liquidity from *both* sides records one BUY and one SELL).
        ``inference_max_step_fraction`` filters out one-tick collapses
        larger than the configured fraction of the prior level — those
        typically reflect order cancellations rather than trades.

        Returns the newly-recorded :class:`Trade`\\ s. The returned
        list is empty on the first call (nothing to diff against) and
        on flat ticks where no eaten liquidity is detected.
        """
        t = now if now is not None else time.time()
        bid_lvl = book.best_bid()
        ask_lvl = book.best_ask()
        if bid_lvl is None or ask_lvl is None or ask_lvl.price <= 0:
            return []

        bid_px, bid_sz = bid_lvl.price, bid_lvl.size
        ask_px, ask_sz = ask_lvl.price, ask_lvl.size
        recorded: List[Trade] = []

        # Aggressive BUY: ask got hit. Either size shrank at the same
        # price (apply the cancellation-noise filter) or the prior
        # level was cleared entirely and the ask price climbed (this
        # is by definition a trade — cancelled orders don't walk the
        # price — so the noise filter is bypassed). New offers
        # improving the ask (lower price) are NOT a trade — that's
        # fresh liquidity, not aggression.
        if self._last_ask_px is not None and self._last_ask_sz is not None:
            if ask_px == self._last_ask_px and ask_sz < self._last_ask_sz:
                eaten = self._last_ask_sz - ask_sz
                if self._accept(eaten, self._last_ask_sz):
                    trade = self.record(
                        price=self._last_ask_px, size=eaten, side="BUY", now=t,
                    )
                    if trade is not None:
                        recorded.append(trade)
            elif ask_px > self._last_ask_px:
                eaten = self._last_ask_sz
                if eaten > 0:
                    trade = self.record(
                        price=self._last_ask_px, size=eaten, side="BUY", now=t,
                    )
                    if trade is not None:
                        recorded.append(trade)

        # Aggressive SELL: bid got hit. Mirror of the BUY logic above.
        if self._last_bid_px is not None and self._last_bid_sz is not None:
            if bid_px == self._last_bid_px and bid_sz < self._last_bid_sz:
                eaten = self._last_bid_sz - bid_sz
                if self._accept(eaten, self._last_bid_sz):
                    trade = self.record(
                        price=self._last_bid_px, size=eaten, side="SELL", now=t,
                    )
                    if trade is not None:
                        recorded.append(trade)
            elif bid_px < self._last_bid_px:
                eaten = self._last_bid_sz
                if eaten > 0:
                    trade = self.record(
                        price=self._last_bid_px, size=eaten, side="SELL", now=t,
                    )
                    if trade is not None:
                        recorded.append(trade)

        # Always refresh the snapshot — even on a flat tick the next
        # change needs the latest sizes to diff against.
        self._last_bid_px, self._last_bid_sz = bid_px, bid_sz
        self._last_ask_px, self._last_ask_sz = ask_px, ask_sz
        self._last_mid = 0.5 * (bid_px + ask_px)
        return recorded

    def _accept(self, eaten: float, prior: Optional[float]) -> bool:
        """Mirror of :class:`TapeInferrer._accept`."""
        if eaten <= 0 or prior is None or prior <= 0:
            return False
        if self.inference_max_step_fraction <= 0:
            return True
        return (eaten / prior) <= self.inference_max_step_fraction

    # ------------------------------------------------------------------
    # read-only accessors
    # ------------------------------------------------------------------

    @property
    def trades(self) -> List[Trade]:
        """Defensive copy of the rolling buffer (newest last)."""
        return list(self._trades)

    @property
    def total_count(self) -> int:
        return len(self._trades)

    def avg_size(self, n: Optional[int] = None) -> Optional[float]:
        """Mean ``size`` over the trailing ``n`` trades (whole buffer).

        Defaults to ``avg_window``. Useful as a generic "current
        baseline" snapshot — but for whale-detection math, prefer
        :meth:`avg_size_at` so the candidate trade doesn't inflate
        the baseline it's being compared against.

        Returns ``None`` when the buffer is empty so callers don't
        accidentally compute a ratio against zero.
        """
        if not self._trades:
            return None
        window = n if n is not None and n > 0 else self.avg_window
        recent = list(self._trades)[-window:]
        if not recent:
            return None
        return sum(t.size for t in recent) / len(recent)

    def avg_size_at(self, ts: float, n: Optional[int] = None) -> Optional[float]:
        """Mean ``size`` over the last ``n`` trades strictly older than ``ts``.

        Used to compute a pre-trade baseline so a whale doesn't
        inflate the average it's being compared to (a 5× whale on a
        20-trade window would otherwise lift the mean 25% on its own
        and fail its own threshold). Default ``n = avg_window``.

        Returns ``None`` when no qualifying prior trades exist.
        """
        if not self._trades:
            return None
        window = n if n is not None and n > 0 else self.avg_window
        prior = [t for t in self._trades if t.timestamp < ts]
        if not prior:
            return None
        prior = prior[-window:]
        return sum(t.size for t in prior) / len(prior)

    def latest_whale(
        self,
        *,
        multiple: float,
        lookback_seconds: float,
        side: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Optional[Trade]:
        """Most recent trade with ``size >= multiple × pre-trade avg`` in window.

        ``side`` (``"BUY"`` / ``"SELL"``) filters by aggressor side
        when set; ``None`` returns the latest whale in either
        direction. The avg-size baseline is computed *as of the
        candidate trade's timestamp* (see :meth:`avg_size_at`) so the
        whale never artificially raises its own threshold.

        Returns ``None`` if no qualifying trade exists in the
        ``lookback_seconds`` window, or if the pre-trade baseline
        isn't computable yet (cold tape).
        """
        if multiple <= 0:
            return None
        cutoff = (now if now is not None else time.time()) - max(0.0, lookback_seconds)
        side_filter = side.upper() if side else None
        for trade in reversed(self._trades):
            if trade.timestamp < cutoff:
                return None
            if side_filter is not None and trade.side != side_filter:
                continue
            avg = self.avg_size_at(trade.timestamp)
            if avg is None or avg <= 0:
                continue
            if trade.size >= avg * multiple:
                return trade
        return None

    def count_aggressive_since(
        self,
        *,
        side: str,
        multiple: float,
        since_ts: float,
        now: Optional[float] = None,
    ) -> int:
        """Trades on ``side`` after ``since_ts`` with ``size >= multiple × pre-baseline avg``.

        The avg-size baseline is taken *as of* ``since_ts`` — i.e.
        the trades being counted are NOT themselves in the baseline.
        That keeps the confirmation threshold stable as confirms come
        in (otherwise each new confirm would raise the bar for the
        next one).

        ``now`` is only used as an upper bound when callers want to
        cap the window; defaults to "no upper bound" (i.e. include
        every trade with timestamp > since_ts).
        """
        if multiple <= 0:
            return 0
        avg = self.avg_size_at(since_ts)
        if avg is None or avg <= 0:
            return 0
        threshold = avg * multiple
        side_norm = side.upper()
        upper = now if now is not None else float("inf")
        count = 0
        for trade in self._trades:
            if trade.timestamp <= since_ts:
                continue
            if trade.timestamp > upper:
                continue
            if trade.side != side_norm:
                continue
            if trade.size >= threshold:
                count += 1
        return count

    def reset(self) -> None:
        self._trades.clear()
        self._last_bid_px = self._last_bid_sz = None
        self._last_ask_px = self._last_ask_sz = None
        self._last_mid = None
