"""Tape Reading Momentum (Flow Scalp) — HFT taker-flow scalper.

Reads the live trade tape for *whale* aggressive prints (single taker
orders ≥ ``whale_multiple`` × the rolling average trade size), waits for
a fast same-direction confirmation print, then front-runs the continued
flow. Pure order-flow signal — no indicators, no mean reversion.

Spec → code mapping
-------------------

1. **Detect whale.** A single taker trade with ``size ≥ whale_multiple ×
   avg_size`` over the trailing ``avg_window`` trades. Spec default
   = 5× over 100 trades. Direction = aggressor side (``BUY`` taker
   lifted the ask; ``SELL`` taker hit the bid).

2. **Confirm.** Within ``confirm_window_seconds`` (spec: 3 s), at least
   ``confirm_count`` additional same-direction trades with
   ``size ≥ confirm_multiple × avg_size`` (spec: 1 more trade ≥ 2× avg)
   must print. Filters single hedge whales that don't carry follow-on.

3. **Enter.** Market entry at the touch (best ask for BUY, best bid for
   SELL). Live execution rides on the executor's IOC / market order
   path; paper exchange fills against the slippage model.

4. **Ride.**

   * Hard TP at ``+take_profit_pct`` from entry mid (spec: 0.08%).
   * Hard SL at ``−stop_loss_pct`` from entry mid (spec: 0.04%).
   * Trailing stop activates the moment ``mid > entry`` for longs
     (mirror for shorts). Stop = ``best_mid_since_entry × (1 −
     trailing_stop_pct)`` for longs. Spec: 0.02% trail.
   * The *effective* stop on a long is ``max(hard_sl, trailing_stop)``
     once trailing is armed, so a profitable run never gives back
     more than ``trailing_stop_pct`` from its high while still capped
     by the hard floor.

5. **Time exit.** Force a market close ``max_hold_seconds`` after entry
   (spec: 60 s) if neither TP, SL, nor trail tripped — whale flow
   doesn't persist, overstaying kills edge.

Data sourcing
-------------

The strategy keeps a :class:`TradeTape` per symbol. It can be fed
either by an external trades feed (``record_trade``) or by inferring
trades from the order book between scans (``infer_from_book``). When
no external feed is wired up, the inference path runs automatically
on every scan — same heuristic the Order Book Sniper already uses for
its tape confirmation, just with the *size* of each inferred trade
preserved so the whale math works.

Like every other strategy in the codebase, FlowScalp only emits
``Signal``s. Sizing, risk vetting, and order submission live in the
executor and risk manager.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market
from aera.signals.trade_tape import Trade, TradeTape

from .base import Leg, Signal, Strategy

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution -> strategies)
    from aera.core import Portfolio


log = get_logger(__name__)


@dataclass
class _PendingWhale:
    """A whale print waiting for confirmation before it can fire an entry."""

    side: str           # "BUY" or "SELL" — aggressor of the whale
    timestamp: float    # whale trade timestamp; confirmation must come after
    size: float
    price: float


@dataclass
class _SymbolState:
    """Per-symbol working state for the flow scalper."""

    tape: TradeTape
    pending: Optional[_PendingWhale] = None
    position_side: Optional[str] = None       # "LONG", "SHORT", or None
    entry_mid: float = 0.0
    entry_size_usd: float = 0.0
    entry_time: float = 0.0
    # Best mid in the direction of profit since entry — drives the
    # trailing stop. For LONG this is the running max; for SHORT the
    # running min. Initialised to entry_mid on fill.
    best_mid: float = 0.0
    last_signal_mid: float = 0.0


class FlowScalp(Strategy):
    """Whale-print + confirmation front-runner.

    Parameters
    ----------
    whale_multiple : float
        A single taker trade must be at least this many times the
        rolling ``avg_window``-trade mean size to qualify as a whale.
        Spec default = 5×.
    confirm_multiple : float
        Follow-on trades count toward confirmation when their size is
        ≥ this × avg_size. Spec default = 2×.
    confirm_count : int
        How many confirmation trades after the whale are required
        before the entry fires. Spec default = 1 (one more print).
    confirm_window_seconds : float
        Maximum time between the whale and its qualifying confirmation
        print, in seconds. Spec default = 3 s. Pending whales expire
        when no confirmation arrives in window.
    avg_window : int
        Trailing trades used for the avg-size baseline. Spec = 100.
    tape_max_trades : int
        Cap on the per-symbol tape buffer. Bounded memory across many
        symbols.
    notional_usd : float
        Reference notional. The executor's ``trade_size_fraction``
        typically overrides this — identical to the other strategies.
    take_profit_pct : float
        Hard take-profit, fraction of entry mid. Spec: 0.0008 (8 bps).
    stop_loss_pct : float
        Hard stop-loss, fraction of entry mid. Spec: 0.0004 (4 bps).
    trailing_stop_pct : float
        Trailing-stop distance from the running best mid since entry,
        fraction of best mid. Spec: 0.0002 (2 bps). 0 disables the
        trail and the strategy falls back to the hard TP / SL only.
    take_profit_usd, stop_loss_usd : float
        USD-P&L exits. When both > 0 and ``portfolio`` is attached
        these take precedence over the % thresholds (same contract as
        the other strategies).
    max_hold_seconds : float
        Force a market exit this many seconds after entry. Spec: 60 s.
        0 disables the time exit.
    rearm_distance_bps : float
        Don't refire on the same symbol until mid moves at least this
        many bps from the previous firing mid. Cheap debouncer.
    leverage_override : float, optional
        Stamp this leverage on every emitted leg, overriding the
        venue's. Spec asks for 5×. ``None`` inherits from market
        metadata (Delta's account leverage).
    min_edge : float
        Floor on emitted ``Signal.edge`` so flow signals sort
        sensibly against the other strategies' queue.
    auto_infer_from_book : bool
        When ``True`` (default), every ``scan`` call also runs
        :meth:`TradeTape.infer_from_book` on the latest book so the
        tape stays warm even without an external trades feed. Disable
        when wiring up a real trades-channel subscription that calls
        :meth:`record_trade` directly.
    portfolio : Portfolio, optional
        Live portfolio used by the USD-P&L exit path.
    clock : callable, optional
        Overridable time source for deterministic tests.
    """

    name = "flow_scalp"

    def __init__(
        self,
        *,
        whale_multiple: float = 5.0,
        confirm_multiple: float = 2.0,
        confirm_count: int = 1,
        confirm_window_seconds: float = 3.0,
        avg_window: int = 100,
        tape_max_trades: int = 500,
        notional_usd: float = 1000.0,
        take_profit_pct: float = 0.0008,
        stop_loss_pct: float = 0.0004,
        trailing_stop_pct: float = 0.0002,
        take_profit_usd: float = 0.0,
        stop_loss_usd: float = 0.0,
        max_hold_seconds: float = 60.0,
        rearm_distance_bps: float = 5.0,
        leverage_override: Optional[float] = 5.0,
        min_edge: float = 0.0008,
        auto_infer_from_book: bool = True,
        portfolio: Optional["Portfolio"] = None,
        enabled: bool = True,
        clock: Optional[callable] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.whale_multiple = max(1.0, float(whale_multiple))
        self.confirm_multiple = max(0.0, float(confirm_multiple))
        self.confirm_count = max(0, int(confirm_count))
        self.confirm_window_seconds = max(0.0, float(confirm_window_seconds))
        self.avg_window = max(1, int(avg_window))
        self.tape_max_trades = max(self.avg_window * 2, int(tape_max_trades))
        self.notional_usd = max(0.0, float(notional_usd))
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.trailing_stop_pct = max(0.0, float(trailing_stop_pct))
        self.take_profit_usd = max(0.0, float(take_profit_usd))
        self.stop_loss_usd = max(0.0, float(stop_loss_usd))
        self.max_hold_seconds = max(0.0, float(max_hold_seconds))
        self.rearm_distance_bps = max(0.0, float(rearm_distance_bps))
        self.leverage_override = (
            float(leverage_override) if leverage_override is not None else None
        )
        self.min_edge = max(0.0, float(min_edge))
        self.auto_infer_from_book = bool(auto_infer_from_book)
        self.portfolio = portfolio
        self._clock = clock or time.time
        self._state: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # external trade ingestion (for future websocket trades feed)
    # ------------------------------------------------------------------

    def record_trade(
        self,
        symbol: str,
        *,
        price: float,
        size: float,
        side: str,
        now: Optional[float] = None,
    ) -> Optional[Trade]:
        """Push a real taker trade onto the symbol's tape.

        Intended for callers that subscribe to a venue's trades
        channel and route prints into the strategy. Returns the
        recorded :class:`Trade` for logging convenience, or ``None``
        if the trade was rejected (zero / negative size).
        """
        st = self._state_for(symbol)
        return st.tape.record(price=price, size=size, side=side, now=now)

    # ------------------------------------------------------------------
    # scan loop
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
            bid_lvl = book.best_bid()
            ask_lvl = book.best_ask()
            if bid_lvl is None or ask_lvl is None or ask_lvl.price <= 0:
                continue
            mid = 0.5 * (bid_lvl.price + ask_lvl.price)
            if mid <= 0:
                continue

            st = self._state_for(m.id)
            # Reconcile internal position state with the live portfolio
            # so we don't fire phantom TP/SL closes after the brain or
            # risk vet vetoed an entry signal.
            self.sync_position_state(st, self.portfolio, m.id, outcome.id)

            # Keep the tape fresh even when positioned — the avg-size
            # baseline must keep tracking so the next setup post-exit
            # uses an up-to-date reference.
            if self.auto_infer_from_book:
                st.tape.infer_from_book(book, now=now)

            # Exit path always wins on the same tick a fresh setup
            # might want to fire.
            close = self._maybe_emit_close(
                st, m, outcome, bid_lvl.price, ask_lvl.price, mid, now,
            )
            if close is not None:
                signals.append(close)
                continue

            if st.position_side is not None:
                continue

            # Expire stale pending whales.
            if (
                st.pending is not None
                and self.confirm_window_seconds > 0
                and (now - st.pending.timestamp) > self.confirm_window_seconds
            ):
                log.debug(
                    "flow-scalp %s pending whale expired side=%s age=%.2fs",
                    m.id, st.pending.side, now - st.pending.timestamp,
                )
                st.pending = None

            # Refresh the pending whale if a newer one prints.
            whale = st.tape.latest_whale(
                multiple=self.whale_multiple,
                lookback_seconds=self.confirm_window_seconds or 60.0,
                now=now,
            )
            if whale is not None and (
                st.pending is None or whale.timestamp > st.pending.timestamp
            ):
                # Adopt the newer print only when it's actually fresh —
                # latest_whale returns the latest qualifying trade; if
                # it's the same one we already had, ignore.
                if st.pending is None or whale.timestamp != st.pending.timestamp:
                    st.pending = _PendingWhale(
                        side=whale.side,
                        timestamp=whale.timestamp,
                        size=whale.size,
                        price=whale.price,
                    )
                    log.debug(
                        "flow-scalp %s whale detected side=%s size=%g (avg=%g)",
                        m.id, whale.side, whale.size,
                        st.tape.avg_size() or 0.0,
                    )

            if st.pending is None:
                continue

            # Rearm debounce.
            if st.last_signal_mid > 0:
                move_bps = abs(mid - st.last_signal_mid) / st.last_signal_mid * 1e4
                if move_bps < self.rearm_distance_bps:
                    continue

            # Confirmation: at least confirm_count same-direction prints
            # AFTER the whale.
            confirms = st.tape.count_aggressive_since(
                side=st.pending.side,
                multiple=self.confirm_multiple,
                since_ts=st.pending.timestamp,
                now=now,
            )
            if confirms < self.confirm_count:
                continue

            # Fire entry on the aggressor side.
            entry_side = st.pending.side
            limit_price = ask_lvl.price if entry_side == "BUY" else bid_lvl.price
            leverage = self._leg_leverage(m)

            avg = st.tape.avg_size() or 0.0
            whale_multiple_obs = (st.pending.size / avg) if avg > 0 else 0.0
            # Edge: scale with how far the whale exceeded threshold.
            # Capped at 1% so it sorts above the other scalpers without
            # dominating the queue.
            strength = min(1.0, whale_multiple_obs / max(self.whale_multiple * 2.0, 1e-9))
            edge = max(self.min_edge, strength * 0.01)

            reason = (
                f"whale={whale_multiple_obs:.2f}x avg={avg:g} "
                f"confirms={confirms} side={entry_side}"
            )
            leg = Leg(
                market_id=m.id,
                outcome_id=outcome.id,
                side=entry_side,
                limit_price=float(limit_price),
                size_usd=self.notional_usd,
                reason=reason,
                leverage=leverage,
            )
            signals.append(
                Signal(
                    strategy=self.name,
                    confidence=min(1.0, strength + 0.1),
                    edge=edge,
                    legs=[leg],
                    metadata={
                        "symbol": m.id,
                        "whale_size": float(st.pending.size),
                        "whale_multiple": float(whale_multiple_obs),
                        "avg_trade_size": float(avg),
                        "confirms": int(confirms),
                        "mid": float(mid),
                        "tape_total": int(st.tape.total_count),
                    },
                )
            )

            st.last_signal_mid = mid
            st.position_side = "LONG" if entry_side == "BUY" else "SHORT"
            st.entry_mid = mid
            st.entry_size_usd = self.notional_usd
            st.entry_time = now
            st.best_mid = mid
            st.pending = None
            log.info(
                "flow-scalp FIRE %s %s mid=%.4f whale=%.2fx confirms=%d",
                m.id, entry_side, mid, whale_multiple_obs, confirms,
            )
        return signals

    # ------------------------------------------------------------------
    # state plumbing
    # ------------------------------------------------------------------

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState(
                tape=TradeTape(
                    max_trades=self.tape_max_trades,
                    avg_window=self.avg_window,
                ),
            )
            self._state[symbol] = st
        return st

    def _leg_leverage(self, market: Market) -> float:
        if self.leverage_override is not None:
            return max(1.0, self.leverage_override)
        try:
            return float(market.metadata.get("leverage", 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0

    # ------------------------------------------------------------------
    # exit emission
    # ------------------------------------------------------------------

    def _maybe_emit_close(
        self,
        st: _SymbolState,
        market: Market,
        outcome,
        bid: float,
        ask: float,
        mid: float,
        now: float,
    ) -> Optional[Signal]:
        """Emit a reduce-only flatten on hold-timeout / SL / TP / trail.

        Priority (highest first):
            1. Hold-timeout — ``now − entry_time > max_hold_seconds``.
               Whale flow doesn't persist; bail before it decays.
            2. USD or % stop-loss (hard).
            3. Trailing stop — only when armed (best_mid past entry)
               and ``trailing_stop_pct > 0``.
            4. USD or % take-profit.
        """
        if st.position_side is None or st.entry_mid <= 0:
            return None

        side = st.position_side
        close_side = "SELL" if side == "LONG" else "BUY"
        # Close at the opposite touch — that's where a market order
        # would actually fill.
        limit_price = bid if side == "LONG" else ask
        leverage = self._leg_leverage(market)

        # Update best mid since entry (drives the trailing stop).
        if side == "LONG":
            if mid > st.best_mid:
                st.best_mid = mid
        else:
            if mid < st.best_mid or st.best_mid == 0:
                st.best_mid = mid

        # --- 1. hold-timeout ----------------------------------------
        if self.max_hold_seconds > 0 and (now - st.entry_time) > self.max_hold_seconds:
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind="hold-timeout",
                reason=f"held {now - st.entry_time:.2f}s > {self.max_hold_seconds:.2f}s",
                extra_metadata={"hold_seconds": now - st.entry_time},
            )

        # --- 2 + 4: USD or % SL / TP ---------------------------------
        usd_active = (
            (self.take_profit_usd > 0 or self.stop_loss_usd > 0)
            and self.portfolio is not None
        )
        hit_sl = False
        hit_tp = False
        pnl_usd: Optional[float] = None

        if usd_active:
            from aera.core import Portfolio  # local import; avoid cycle

            key = Portfolio._key(market.id, outcome.id)
            pos = self.portfolio.positions.get(key) if self.portfolio else None
            if pos is None or pos.shares == 0:
                self._reset_position(st)
                return None
            mark = limit_price
            pnl_usd = (mark - pos.avg_cost) * pos.shares
            if self.stop_loss_usd > 0 and pnl_usd <= -self.stop_loss_usd:
                hit_sl = True
            if self.take_profit_usd > 0 and pnl_usd >= self.take_profit_usd:
                hit_tp = True
        else:
            tp = self.take_profit_pct
            sl = self.stop_loss_pct
            entry = st.entry_mid
            if side == "LONG":
                if sl > 0 and mid <= entry * (1.0 - sl):
                    hit_sl = True
                if tp > 0 and mid >= entry * (1.0 + tp):
                    hit_tp = True
            else:
                if sl > 0 and mid >= entry * (1.0 + sl):
                    hit_sl = True
                if tp > 0 and mid <= entry * (1.0 - tp):
                    hit_tp = True

        if hit_sl:
            kind = "stop-loss"
            move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
            pnl_label = f" pnl=${pnl_usd:+.2f}" if pnl_usd is not None else ""
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind=kind,
                reason=f"entry={st.entry_mid:.4f} mid={mid:.4f} ({move_bps:+.1f}bps){pnl_label}",
                extra_metadata=({"pnl_usd": float(pnl_usd)} if pnl_usd is not None else {}),
            )

        # --- 3. trailing stop (only when armed and SL hasn't tripped) -
        if self.trailing_stop_pct > 0:
            armed = (
                (side == "LONG" and st.best_mid > st.entry_mid)
                or (side == "SHORT" and 0 < st.best_mid < st.entry_mid)
            )
            if armed:
                if side == "LONG":
                    trail_level = st.best_mid * (1.0 - self.trailing_stop_pct)
                    if mid <= trail_level:
                        give_back_bps = (st.best_mid - mid) / st.best_mid * 1e4
                        run_bps = (st.best_mid - st.entry_mid) / st.entry_mid * 1e4
                        return self._build_close(
                            st, market, outcome, side, close_side, limit_price, leverage,
                            kind="trailing-stop",
                            reason=(
                                f"best={st.best_mid:.4f} (+{run_bps:.1f}bps) "
                                f"mid={mid:.4f} (-{give_back_bps:.1f}bps from best)"
                            ),
                            extra_metadata={
                                "best_mid": float(st.best_mid),
                                "run_bps": float(run_bps),
                                "give_back_bps": float(give_back_bps),
                            },
                        )
                else:
                    trail_level = st.best_mid * (1.0 + self.trailing_stop_pct)
                    if mid >= trail_level:
                        give_back_bps = (mid - st.best_mid) / st.best_mid * 1e4
                        run_bps = (st.entry_mid - st.best_mid) / st.entry_mid * 1e4
                        return self._build_close(
                            st, market, outcome, side, close_side, limit_price, leverage,
                            kind="trailing-stop",
                            reason=(
                                f"best={st.best_mid:.4f} (+{run_bps:.1f}bps) "
                                f"mid={mid:.4f} (+{give_back_bps:.1f}bps from best)"
                            ),
                            extra_metadata={
                                "best_mid": float(st.best_mid),
                                "run_bps": float(run_bps),
                                "give_back_bps": float(give_back_bps),
                            },
                        )

        if hit_tp:
            kind = "take-profit"
            move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
            pnl_label = f" pnl=${pnl_usd:+.2f}" if pnl_usd is not None else ""
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind=kind,
                reason=f"entry={st.entry_mid:.4f} mid={mid:.4f} ({move_bps:+.1f}bps){pnl_label}",
                extra_metadata=({"pnl_usd": float(pnl_usd)} if pnl_usd is not None else {}),
            )

        return None

    def _build_close(
        self,
        st: _SymbolState,
        market: Market,
        outcome,
        position_side: str,
        close_side: str,
        limit_price: float,
        leverage: float,
        *,
        kind: str,
        reason: str,
        extra_metadata: Optional[Dict[str, object]] = None,
    ) -> Signal:
        entry_mid = st.entry_mid
        entry_size = st.entry_size_usd
        leg = Leg(
            market_id=market.id,
            outcome_id=outcome.id,
            side=close_side,
            limit_price=float(limit_price),
            size_usd=entry_size,
            reason=f"{kind}: {reason}",
            leverage=leverage,
            reduce_only=True,
        )
        meta: Dict[str, object] = {
            "symbol": market.id,
            "exit": kind,
            "position_side": position_side,
            "entry_mid": float(entry_mid),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        sig = Signal(
            strategy=self.name,
            confidence=1.0,
            edge=max(self.min_edge, 0.01),
            legs=[leg],
            metadata=meta,
        )
        self._reset_position(st)
        log.info(
            "flow-scalp EXIT %s %s side=%s entry=%.4f close=%.4f (%s)",
            market.id, kind.upper(), position_side, entry_mid, limit_price, reason,
        )
        return sig

    @staticmethod
    def _reset_position(st: _SymbolState) -> None:
        st.position_side = None
        st.entry_mid = 0.0
        st.entry_size_usd = 0.0
        st.entry_time = 0.0
        st.best_mid = 0.0
