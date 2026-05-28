"""Bid-Ask Spread Fade — market-making lite for Delta perpetuals.

A high-frequency micro-profit strategy that posts simultaneous resting
limit orders on both sides of the spread and earns the bid-ask spread
as carry. Designed for illiquidity spikes on liquid majors where the
spread widens transiently — the strategy quotes inside the wide spread,
captures a configurable fraction of it, and rolls inventory net-flat.

Spec-mapped behaviour
---------------------

1. **Quote.** Every ``refresh_rate_ms`` (default 500 ms), post bid at
   ``mid − half_spread_capture`` and ask at ``mid + half_spread_capture``
   where ``half_spread_capture = spread × capture_target / 2``. Quotes
   are only emitted when ``spread / mid ≥ min_spread_pct`` (default
   0.03%).
2. **Fee gate.** Reject the cycle if projected capture is not net-
   positive after two maker fees: ``capture × mid − 2 × maker_fee_bps``
   must exceed ``min_net_edge_bps``. Delta's maker fee is 0.02% so the
   net edge target of 0.04% keeps the strategy above its breakeven
   line.
3. **Inventory skew.** If long inventory > ``inventory_skew_threshold_usd``
   (default $10), shift both quotes DOWN by ``inventory_skew_ticks``
   ticks to bias selling. Mirror for short inventory. Inventory cap is
   ``max_inventory_usd`` (default ±$15) — past that, the offending side
   is suppressed so a runaway one-sided fill can't pile up.
4. **Kill switch.** Track mid prints over the last
   ``kill_window_seconds`` (default 5s). If ``(max − min) / oldest_mid``
   exceeds ``kill_move_pct`` (default 0.08%), suspend quoting until
   volatility normalises. Existing inventory rides the regime change;
   no forced unwind.
5. **Fills.** When one side fills the opposite quote becomes the profit-
   lock for the next cycle — the inventory skew naturally biases
   subsequent quoting toward flattening. If both sides fill within one
   cycle the spread is booked as realised PnL on the next tick when
   the position closes.

Implementation notes
--------------------

The strategy emits **two separate single-leg Signals** per quote cycle
(one BUY, one SELL) rather than a multi-leg signal. The Executor
processes signals atomically and unwinds partial fills, which is
exactly what we DON'T want for market making — independent fills are
the entire point. By emitting them as separate signals the engine
processes them serially, each through its own risk vet and fill path,
without unwinding the other if only one trades.

On the live exchange, true market making requires resting limit orders
(``post_only=true`` + ``time_in_force="gtc"``). The default
``DeltaLiveExchange`` uses ``ioc`` so live mode will see most quotes
expire unfilled — flip to ``post_only=True`` on the exchange when
running this strategy live.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Deque, Dict, Iterable, List, Optional, Tuple

from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market

from .base import Leg, Signal, Strategy

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution -> strategies)
    from aera.core import Portfolio


log = get_logger(__name__)


@dataclass
class _SymbolState:
    """Per-symbol working state owned by the spread-fade strategy.

    Holds the rolling mid window used by the kill switch, the timestamp
    of the last emitted quote cycle (for the refresh-rate gate), and
    bookkeeping for the most recent quoted bid/ask so debug output can
    explain skips.
    """

    mid_history: Deque[Tuple[float, float]] = field(default_factory=deque)
    last_quote_time: float = 0.0
    last_quote_bid: float = 0.0
    last_quote_ask: float = 0.0
    suspended_until: float = 0.0  # set after a kill-switch trip


class BidAskSpreadFade(Strategy):
    """Symmetric two-sided market making with inventory + volatility guards.

    Parameters
    ----------
    min_spread_pct : float
        Floor on ``spread / mid`` required to quote. ``0.0003`` = 0.03%.
        Tighter spreads make the round-trip net-negative after fees.
    capture_target : float
        Fraction of the spread the strategy attempts to capture (split
        evenly across the two sides). ``0.60`` = ``mid ± 30% of spread``.
        Higher values mean wider quotes (less aggressive) — more passive
        and more selective.
    quote_size_usd : float
        Reference notional per side. The Executor's ``trade_size_fraction``
        still scales this against buying power, identical to the other
        Delta strategies. Defaults to a $5 per side floor.
    max_inventory_usd : float
        Hard cap on absolute open-position notional from this strategy.
        Once exceeded on a given side the strategy suppresses that side's
        quote until inventory shrinks back inside the cap.
    inventory_skew_threshold_usd : float
        Inventory level past which the strategy starts skewing quotes to
        bias offloading. Always ``< max_inventory_usd``.
    inventory_skew_ticks : int
        Number of ticks to shift quotes when skewing. Shifts both bid and
        ask in the same direction (down when long, up when short) so the
        net effect is to make the favoured side more aggressive.
    refresh_rate_ms : float
        Minimum interval between re-quotes (per symbol), in milliseconds.
        The Delta engine's scan loop ticks much faster than this, so the
        gate determines the actual MM cadence. Spec default = 500 ms.
    kill_move_pct : float
        Mid-price move (peak-to-trough) over ``kill_window_seconds``
        that triggers the kill switch. ``0.0008`` = 0.08%.
    kill_window_seconds : float
        Lookback window for the kill-switch volatility measurement.
    maker_fee_bps : float
        Assumed maker fee per side, in basis points. Used only by the
        net-edge gate — the actual fee charged on fills comes from the
        execution config. Spec uses 0.02% = 2 bps.
    min_net_edge_bps : float
        Minimum projected net edge after two maker fees for a quote
        cycle to be allowed. Spec target = 0.04% = 4 bps.
    leverage_override : float, optional
        If set, stamps this leverage on every emitted leg instead of
        reading from the market metadata. The spec asks for ``1.0``
        (cash-style sizing). When ``None`` the strategy follows the
        venue leverage from the market metadata (consistent with the
        other Delta strategies).
    min_edge : float
        Floor on the emitted ``Signal.edge`` so quotes don't sort below
        noise in the engine's signal queue.
    portfolio : Portfolio, optional
        Live portfolio used to read open-position notional for the
        inventory skew and cap. The strategy never mutates it.
    clock : callable, optional
        Overridable time source so tests can drive refresh-rate and
        kill-switch behaviour without sleeping.
    """

    name = "bid_ask_spread_fade"

    def __init__(
        self,
        *,
        min_spread_pct: float = 0.0003,
        capture_target: float = 0.60,
        quote_size_usd: float = 5.0,
        max_inventory_usd: float = 15.0,
        inventory_skew_threshold_usd: float = 10.0,
        inventory_skew_ticks: int = 1,
        refresh_rate_ms: float = 500.0,
        kill_move_pct: float = 0.0008,
        kill_window_seconds: float = 5.0,
        maker_fee_bps: float = 2.0,
        min_net_edge_bps: float = 4.0,
        leverage_override: Optional[float] = None,
        min_edge: float = 0.0004,
        portfolio: Optional["Portfolio"] = None,
        enabled: bool = True,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.min_spread_pct = max(0.0, float(min_spread_pct))
        self.capture_target = min(1.0, max(0.0, float(capture_target)))
        self.quote_size_usd = max(0.0, float(quote_size_usd))
        self.max_inventory_usd = max(0.0, float(max_inventory_usd))
        # The skew threshold should never exceed the cap — if it did we'd
        # never start biasing before suppressing the side outright.
        self.inventory_skew_threshold_usd = min(
            self.max_inventory_usd,
            max(0.0, float(inventory_skew_threshold_usd)),
        )
        self.inventory_skew_ticks = max(0, int(inventory_skew_ticks))
        self.refresh_rate_ms = max(0.0, float(refresh_rate_ms))
        self.kill_move_pct = max(0.0, float(kill_move_pct))
        self.kill_window_seconds = max(0.1, float(kill_window_seconds))
        self.maker_fee_bps = max(0.0, float(maker_fee_bps))
        self.min_net_edge_bps = max(0.0, float(min_net_edge_bps))
        self.leverage_override = (
            None if leverage_override is None else max(1.0, float(leverage_override))
        )
        self.min_edge = max(0.0, float(min_edge))
        self.portfolio = portfolio
        self._clock = clock or time.time
        self._state: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # public scan loop
    # ------------------------------------------------------------------

    def scan(self, markets: Iterable[Market]) -> List[Signal]:
        signals: List[Signal] = []
        now = float(self._clock())
        for m in markets:
            if m.venue != "delta":
                continue
            outcome = next(iter(m.outcomes.values()), None)
            if outcome is None or outcome.label != DELTA_OUTCOME_LABEL:
                continue

            book = outcome.book
            bid = book.best_bid_price()
            ask = book.best_ask_price()
            if bid is None or ask is None or bid <= 0 or ask <= 0 or ask <= bid:
                continue
            mid = 0.5 * (bid + ask)
            spread = ask - bid
            if mid <= 0 or spread <= 0:
                continue

            st = self._state_for(m.id)
            # Always feed the kill-switch buffer, even on cycles where we
            # won't be quoting — the vol estimate must keep up with the
            # market or we'll re-enter quoting prematurely after a spike.
            self._record_mid(st, mid, now)

            # 1. Kill-switch: short-window vol over the configured window.
            #    Returns the move ratio so the reason string is useful.
            kill_move = self._kill_move(st, now)
            if self.kill_move_pct > 0 and kill_move > self.kill_move_pct:
                # Don't quote — but DO keep updating the buffer (already done).
                # We do NOT force-unwind: the spec says "wait for volatility
                # to normalise", so existing inventory rides through.
                log.debug(
                    "spread-fade %s KILL-SWITCH move=%.4f%% > %.4f%% — skip quoting",
                    m.id, kill_move * 100.0, self.kill_move_pct * 100.0,
                )
                continue

            # 2. Refresh-rate gate.
            if self.refresh_rate_ms > 0:
                ms_since = (now - st.last_quote_time) * 1000.0
                if st.last_quote_time > 0 and ms_since < self.refresh_rate_ms:
                    continue

            # 3. Min spread gate.
            spread_pct = spread / mid
            if spread_pct < self.min_spread_pct:
                continue

            # 4. Net-edge gate. The fraction of the spread we expect to
            #    capture, minus two maker fees, must exceed min_net_edge.
            capture_bps = spread_pct * self.capture_target * 1e4
            net_bps = capture_bps - 2.0 * self.maker_fee_bps
            if net_bps < self.min_net_edge_bps:
                continue

            # 5. Quote prices: mid ± (spread × capture / 2). The /2 splits
            #    the captured spread evenly between the two sides.
            half_capture = spread * self.capture_target * 0.5
            quote_bid = mid - half_capture
            quote_ask = mid + half_capture

            # 6. Inventory skew: read live position notional from the
            #    portfolio (single source of truth) and shift quotes if
            #    we're outside the neutral band.
            inv_usd = self._inventory_usd(m, outcome, mid)
            tick = m.minimum_tick if m.minimum_tick and m.minimum_tick > 0 else mid * 1e-4
            skew_shift = self._inventory_skew_shift(inv_usd, tick)
            quote_bid += skew_shift
            quote_ask += skew_shift

            # 7. Don't let the (potentially skewed) quotes cross the
            #    spread the wrong way and turn into taker orders. The
            #    maker constraint is:
            #       quote_bid < best_ask  (else we'd lift the ask)
            #       quote_ask > best_bid  (else we'd hit the bid)
            #    The strategy quotes INSIDE the spread by design, so
            #    quote_bid being above best_bid (and quote_ask below
            #    best_ask) is expected — that's how the maker becomes
            #    the new best bid/ask.
            if quote_bid >= ask:
                quote_bid = ask - tick
            if quote_ask <= bid:
                quote_ask = bid + tick

            # 8. Inventory cap: suppress the side that would push past it.
            quote_buy = inv_usd < self.max_inventory_usd
            quote_sell = inv_usd > -self.max_inventory_usd

            try:
                leverage = float(m.metadata.get("leverage", 1.0) or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0
            if self.leverage_override is not None:
                leverage = self.leverage_override

            # Edge: scaled by the gross spread capture in bps so wider
            # spreads naturally sort higher. Capped at 1% so MM quotes
            # don't dominate the engine's signal queue when scalper +
            # MM are both running on the same tick.
            edge = max(self.min_edge, min(0.01, capture_bps / 1e4))
            reason_base = (
                f"spread={spread:.4f} ({spread_pct*1e4:.1f}bps) "
                f"capture={capture_bps:.1f}bps net={net_bps:.1f}bps "
                f"inv=${inv_usd:+.2f}"
            )

            quoted_any = False
            if quote_buy:
                signals.append(
                    self._build_quote_signal(
                        m, outcome,
                        side="BUY",
                        limit_price=float(quote_bid),
                        leverage=leverage,
                        edge=edge,
                        reason_base=reason_base,
                        capture_bps=capture_bps,
                        net_bps=net_bps,
                        inv_usd=inv_usd,
                        mid=mid,
                        spread=spread,
                    )
                )
                quoted_any = True
            if quote_sell:
                signals.append(
                    self._build_quote_signal(
                        m, outcome,
                        side="SELL",
                        limit_price=float(quote_ask),
                        leverage=leverage,
                        edge=edge,
                        reason_base=reason_base,
                        capture_bps=capture_bps,
                        net_bps=net_bps,
                        inv_usd=inv_usd,
                        mid=mid,
                        spread=spread,
                    )
                )
                quoted_any = True

            if quoted_any:
                st.last_quote_time = now
                st.last_quote_bid = quote_bid
                st.last_quote_ask = quote_ask
                log.debug(
                    "spread-fade %s QUOTE bid=%.4f ask=%.4f spread=%.4f "
                    "(%.1fbps) inv=$%+.2f skew=%+.4f",
                    m.id, quote_bid, quote_ask, spread, spread_pct * 1e4,
                    inv_usd, skew_shift,
                )

        return signals

    # ------------------------------------------------------------------
    # state plumbing
    # ------------------------------------------------------------------

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState()
            self._state[symbol] = st
        return st

    def _record_mid(self, st: _SymbolState, mid: float, now: float) -> None:
        """Append the latest mid + evict points outside the kill window."""
        st.mid_history.append((now, mid))
        cutoff = now - self.kill_window_seconds
        while st.mid_history and st.mid_history[0][0] < cutoff:
            st.mid_history.popleft()

    def _kill_move(self, st: _SymbolState, now: float) -> float:
        """Compute peak-to-trough mid move over the kill window, as a fraction.

        Anchored on the oldest mid in the window so a flash up-then-down
        round trip still trips the kill switch (the trough vs the start,
        or the peak vs the start, whichever is larger).
        """
        if len(st.mid_history) < 2:
            return 0.0
        oldest = st.mid_history[0][1]
        if oldest <= 0:
            return 0.0
        mids = [m for _, m in st.mid_history]
        hi = max(mids)
        lo = min(mids)
        return max(hi - oldest, oldest - lo, hi - lo) / oldest

    def _inventory_usd(self, market: Market, outcome, mid: float) -> float:
        """Signed open-position notional for this market, in USD.

        Reads the live portfolio if attached. ``+`` is long, ``−`` is
        short. Falls back to 0.0 (neutral) when no portfolio is wired
        in — useful for hand-built tests and the "stateless" preview
        path. Marks at ``mid`` so the cap doesn't oscillate with each
        tick of the bid/ask.
        """
        if self.portfolio is None:
            return 0.0
        from aera.core import Portfolio  # local import; avoid cycle

        key = Portfolio._key(market.id, outcome.id)
        pos = self.portfolio.positions.get(key)
        if pos is None or pos.shares == 0:
            return 0.0
        return float(pos.shares) * float(mid)

    def _inventory_skew_shift(self, inv_usd: float, tick: float) -> float:
        """Return signed tick offset to add to both quotes.

        Long inventory above the threshold → shift DOWN (negative) by
        ``inventory_skew_ticks`` ticks to bias selling. Short → shift
        UP. Inside the neutral band, shift = 0.
        """
        if self.inventory_skew_ticks <= 0 or tick <= 0:
            return 0.0
        if inv_usd > self.inventory_skew_threshold_usd:
            return -self.inventory_skew_ticks * tick
        if inv_usd < -self.inventory_skew_threshold_usd:
            return +self.inventory_skew_ticks * tick
        return 0.0

    # ------------------------------------------------------------------
    # signal construction
    # ------------------------------------------------------------------

    def _build_quote_signal(
        self,
        market: Market,
        outcome,
        *,
        side: str,
        limit_price: float,
        leverage: float,
        edge: float,
        reason_base: str,
        capture_bps: float,
        net_bps: float,
        inv_usd: float,
        mid: float,
        spread: float,
    ) -> Signal:
        leg = Leg(
            market_id=market.id,
            outcome_id=outcome.id,
            side=side,
            limit_price=float(limit_price),
            size_usd=self.quote_size_usd,
            reason=f"mm-quote {side} {reason_base}",
            leverage=leverage,
            # Maker-only: post a resting limit order inside the spread
            # so Delta rebates the maker fee. ``post_only=True`` makes
            # the venue refuse the order if it would cross (= take
            # liquidity); ``gtc`` keeps it on the book until filled or
            # cancelled. Without these the IOC default would expire
            # the inside-spread quote on the very next book scan and
            # the strategy could never accumulate a position — which
            # is exactly why it produced 0 fires before this wiring.
            time_in_force="gtc",
            post_only=True,
        )
        return Signal(
            strategy=self.name,
            # Confidence scales with the slice of the spread we're
            # capturing — bigger capture → higher confidence in MM-mode.
            confidence=min(1.0, self.capture_target),
            edge=edge,
            legs=[leg],
            metadata={
                "symbol": market.id,
                "quote_side": side,
                "mid": float(mid),
                "spread": float(spread),
                "capture_bps": float(capture_bps),
                "net_bps_after_fees": float(net_bps),
                "inventory_usd": float(inv_usd),
                "limit_price": float(limit_price),
            },
        )
