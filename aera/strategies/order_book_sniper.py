"""Order Book Sniper — L2 depth-imbalance scalp (DOM scalp).

A high-frequency micro-profit strategy for Delta perpetuals. The mental
model is "front-run the wall": when one side of the depth-of-market is
significantly stacked over the other AND the recent tape agrees with
that direction, the resting wall is providing real support/resistance.
Enter on the favoured side with a tight limit, ride a 5-bp move, and bail
the moment the wall is pulled (spoofing defense).

Spec-mapped behaviour
---------------------

1. **Scan L2 book.** Cumulative bid size within ``imbalance_band_bps``
   of mid must exceed cumulative ask size by ``imbalance_ratio``×
   (or vice-versa for shorts). Top-N levels only.
2. **Confirm tape.** At least ``tape_min_count`` aggressive taker buys
   (sells) inferred from the order book deltas over the last
   ``tape_window_seconds`` seconds in the same direction as the
   imbalance. The tape inferrer lives in :mod:`aera.signals.order_book`.
3. **Enter.** Limit price at best_bid + ``entry_tick_offset`` ticks for a
   buy (best_ask − N ticks for a sell). The paper exchange fills as soon
   as the slippage model allows; live execution rides on ``time_in_force``
   from the executor's exchange config.
4. **Exit.** Tight TP (default +0.05% from entry mid) + tight SL
   (default −0.03%). USD-P&L thresholds are also supported with the same
   precedence rules as the mean-reversion scalper. ``max_hold_seconds``
   forces a market exit if neither band hits.
5. **Spoof defense.** Record the wall size on the entry side at fire
   time. If the wall shrinks by more than ``spoof_vanish_ratio`` within
   ``spoof_persist_seconds`` of entry, market-exit regardless of P&L.

Like the other strategies, the sniper only emits ``Signal``s — actual
order submission, sizing, and risk-vetting all live in the executor and
risk manager. This is the highest-leverage strategy in the codebase per
the spec ("10× max safe"), but it sizes through the same
``trade_size_fraction × buying_power`` knob the rest of the bot uses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market
from aera.signals.order_book import (
    DepthImbalanceSnapshot,
    TapeInferrer,
    WallSnapshot,
    measure_depth_imbalance,
)

from .base import Leg, Signal, Strategy

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution -> strategies)
    from aera.core import Portfolio


log = get_logger(__name__)


@dataclass
class _SymbolState:
    """Per-symbol working state for the sniper.

    Owns the tape inferrer for that symbol, the open-position snapshot, the
    wall snapshot used for the spoofing exit, and the rearm-debounce mid.
    """

    tape: TapeInferrer
    last_signal_mid: float = 0.0
    position_side: Optional[str] = None       # "LONG", "SHORT", or None
    entry_mid: float = 0.0
    entry_size_usd: float = 0.0
    entry_time: float = 0.0
    wall: Optional[WallSnapshot] = None


class OrderBookSniper(Strategy):
    """High-frequency DOM-scalp on Delta perpetuals.

    Parameters
    ----------
    imbalance_ratio : float
        Required cumulative depth ratio in the favoured direction within
        the band. ``3.0`` means "bids in band ≥ 3× asks in band → BUY".
    imbalance_band_bps : float
        Band width around mid, in basis points. Spec default = 10 bps
        (i.e. ±0.1% of mid).
    imbalance_max_levels : int
        Top-N levels considered when summing band depth.
    tape_min_count : int
        Minimum inferred aggressive takers in the matching direction
        over the last ``tape_window_seconds``. Spec default = 3.
    tape_window_seconds : float
        Sliding tape window. Spec default = 2.0 s.
    notional_usd : float
        Reference USD notional emitted on each fire (the executor's
        ``trade_size_fraction`` typically overrides this).
    take_profit_pct, stop_loss_pct : float
        Percent-of-entry-mid TP / SL. Spec defaults: +0.0005 / −0.0003.
    take_profit_usd, stop_loss_usd : float
        USD-P&L TP / SL (require ``portfolio`` to be active).
    max_hold_seconds : float
        Force a market exit after this many seconds since entry. Set to 0
        to disable the time-based exit.
    spoof_min_wall_contracts : float
        Only protect against walls at least this big at entry time. Use
        the underlying's contract sizing — e.g. 50 for ``50 BTC`` on
        BTCUSD. ``0`` enables protection for any wall.
    spoof_persist_seconds : float
        How long after entry to keep watching the wall. Outside this
        window normal wall evolution is no longer treated as spoofing.
    spoof_vanish_ratio : float
        The wall counts as "vanished" once its size drops below
        ``original_size × (1 − vanish_ratio)``. ``0.5`` = "half pulled".
    entry_tick_offset : int
        Ticks above best_bid (below best_ask) to post the entry limit at.
    rearm_distance_bps : float
        Don't re-fire on the same symbol until mid moves at least this
        far from the previous firing mid. Cheap debouncer.
    min_edge : float
        Floor on the emitted ``Signal.edge`` so it sorts above noise.
    portfolio : Portfolio, optional
        Live portfolio for the USD-P&L exit path. The strategy never
        mutates it — same contract as the mean-reversion scalper.
    """

    name = "order_book_sniper"

    def __init__(
        self,
        *,
        imbalance_ratio: float = 3.0,
        imbalance_band_bps: float = 10.0,
        imbalance_max_levels: int = 10,
        tape_min_count: int = 3,
        tape_window_seconds: float = 2.0,
        notional_usd: float = 1000.0,
        take_profit_pct: float = 0.0005,
        stop_loss_pct: float = 0.0003,
        take_profit_usd: float = 0.0,
        stop_loss_usd: float = 0.0,
        max_hold_seconds: float = 10.0,
        spoof_min_wall_contracts: float = 0.0,
        spoof_persist_seconds: float = 1.0,
        spoof_vanish_ratio: float = 0.5,
        entry_tick_offset: int = 1,
        rearm_distance_bps: float = 3.0,
        min_edge: float = 0.0005,
        portfolio: Optional["Portfolio"] = None,
        enabled: bool = True,
        clock: Optional[callable] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.imbalance_ratio = max(1.0, float(imbalance_ratio))
        self.imbalance_band_bps = max(0.1, float(imbalance_band_bps))
        self.imbalance_max_levels = max(1, int(imbalance_max_levels))
        self.tape_min_count = max(0, int(tape_min_count))
        self.tape_window_seconds = max(0.1, float(tape_window_seconds))
        self.notional_usd = max(0.0, float(notional_usd))
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.take_profit_usd = max(0.0, float(take_profit_usd))
        self.stop_loss_usd = max(0.0, float(stop_loss_usd))
        self.max_hold_seconds = max(0.0, float(max_hold_seconds))
        self.spoof_min_wall_contracts = max(0.0, float(spoof_min_wall_contracts))
        self.spoof_persist_seconds = max(0.0, float(spoof_persist_seconds))
        # Clamp the vanish ratio to (0, 1] so the math always makes sense.
        self.spoof_vanish_ratio = min(1.0, max(0.0, float(spoof_vanish_ratio)))
        self.entry_tick_offset = max(0, int(entry_tick_offset))
        self.rearm_distance_bps = max(0.0, float(rearm_distance_bps))
        self.min_edge = max(0.0, float(min_edge))
        self.portfolio = portfolio
        # The clock is overridable so tests can drive entry/exit timing
        # without sleeping. Defaults to ``time.time``.
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
            if bid is None or ask is None or ask <= 0:
                continue
            mid = 0.5 * (bid + ask)
            if mid <= 0:
                continue

            st = self._state_for(m.id)
            # Reconcile internal position state with the live portfolio
            # so we don't fire phantom TP/SL closes after the brain or
            # risk vet vetoed an entry signal.
            self.sync_position_state(st, self.portfolio, m.id, outcome.id)
            # Always feed the tape inferrer, even if we won't fire this
            # tick — it builds the rolling window we'll consult next time.
            taker_buys, taker_sells = st.tape.update(book, now=now)

            # Exit path first: a tripped TP / SL / spoof / hold-timeout
            # always wins over opening a new position on the same tick.
            close = self._maybe_emit_close(
                st, m, outcome, bid, ask, mid, now,
            )
            if close is not None:
                signals.append(close)
                continue

            # Don't open while already positioned in the same direction.
            if st.position_side is not None:
                # Tape and rearm baseline are still updated above; the
                # position must close (via TP/SL/spoof/hold) before we
                # consider stacking another fire on the same symbol.
                continue

            # Rearm debounce (skip when there's no previous fire yet).
            if st.last_signal_mid > 0:
                move_bps = abs(mid - st.last_signal_mid) / st.last_signal_mid * 1e4
                if move_bps < self.rearm_distance_bps:
                    continue

            imb = measure_depth_imbalance(
                book,
                band_bps=self.imbalance_band_bps,
                max_levels=self.imbalance_max_levels,
            )
            if imb is None:
                continue

            side, wall_side, wall_price, wall_size = self._direction_from_imbalance(
                imb, bid, ask, book,
            )
            if side is None:
                continue

            # Tape confirmation in the SAME direction.
            if side == "BUY" and taker_buys < self.tape_min_count:
                continue
            if side == "SELL" and taker_sells < self.tape_min_count:
                continue

            # Tick-aware entry limit. minimum_tick of 0 is meaningless;
            # default to a sensible 1 bp of mid in that case so we don't
            # cross the spread by accident.
            tick = m.minimum_tick if m.minimum_tick and m.minimum_tick > 0 else mid * 1e-4
            if side == "BUY":
                limit_price = bid + self.entry_tick_offset * tick
            else:
                limit_price = ask - self.entry_tick_offset * tick
            # Don't let the rounding cross the spread the wrong way.
            if side == "BUY" and limit_price >= ask:
                limit_price = max(bid, ask - tick)
            if side == "SELL" and limit_price <= bid:
                limit_price = min(ask, bid + tick)

            try:
                leverage = float(m.metadata.get("leverage", 1.0) or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0

            edge = max(
                self.min_edge,
                min(imb.ratio if side == "BUY" else imb.inverse_ratio,
                    self.imbalance_ratio * 2.0) / (self.imbalance_ratio * 2.0) * 0.01,
            )
            reason = (
                f"imbalance={imb.ratio if side == 'BUY' else imb.inverse_ratio:.2f} "
                f"band={self.imbalance_band_bps:.1f}bps "
                f"tape={taker_buys if side == 'BUY' else taker_sells} "
                f"wall@{wall_price:.4f}={wall_size:g}"
            )
            leg = Leg(
                market_id=m.id,
                outcome_id=outcome.id,
                side=side,
                limit_price=float(limit_price),
                size_usd=self.notional_usd,
                reason=reason,
                leverage=leverage,
            )
            signals.append(
                Signal(
                    strategy=self.name,
                    confidence=min(
                        1.0,
                        (imb.ratio if side == "BUY" else imb.inverse_ratio)
                        / max(self.imbalance_ratio, 1e-9),
                    ),
                    edge=edge,
                    legs=[leg],
                    metadata={
                        "symbol": m.id,
                        "imbalance_ratio": float(
                            imb.ratio if side == "BUY" else imb.inverse_ratio
                        ),
                        "bid_band_size": float(imb.bid_size),
                        "ask_band_size": float(imb.ask_size),
                        "tape_count": int(
                            taker_buys if side == "BUY" else taker_sells
                        ),
                        "wall_side": wall_side,
                        "wall_price": float(wall_price),
                        "wall_size": float(wall_size),
                        "mid": float(mid),
                    },
                )
            )

            st.last_signal_mid = mid
            st.position_side = "LONG" if side == "BUY" else "SHORT"
            st.entry_mid = mid
            st.entry_size_usd = self.notional_usd
            st.entry_time = now
            st.wall = WallSnapshot(
                side=wall_side, price=wall_price, size=wall_size, observed_at=now,
            )
            log.debug(
                "sniper FIRE %s %s mid=%.4f imb=%.2f tape=%d wall=%g@%.4f",
                m.id, side, mid,
                imb.ratio if side == "BUY" else imb.inverse_ratio,
                taker_buys if side == "BUY" else taker_sells,
                wall_size, wall_price,
            )
        return signals

    # ------------------------------------------------------------------
    # state plumbing
    # ------------------------------------------------------------------

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState(
                tape=TapeInferrer(window_seconds=self.tape_window_seconds),
            )
            self._state[symbol] = st
        return st

    def _direction_from_imbalance(
        self,
        imb: DepthImbalanceSnapshot,
        bid: float,
        ask: float,
        book,
    ) -> tuple[Optional[str], str, float, float]:
        """Pick BUY / SELL / None from an imbalance snapshot.

        Returns ``(side, wall_side, wall_price, wall_size)`` where
        ``wall_*`` describes the largest resting level on the favoured
        side (used for the spoofing exit). ``side is None`` when neither
        direction crosses the configured ratio.
        """
        if imb.ratio >= self.imbalance_ratio and imb.bid_size >= imb.ask_size:
            # The "wall" is the largest bid in the band — that's the
            # support we're front-running.
            wall_price, wall_size = self._largest_level(
                book.bids_sorted()[: self.imbalance_max_levels],
                floor=imb.mid - imb.mid * (self.imbalance_band_bps / 1e4),
                side="BID",
            )
            return "BUY", "BID", wall_price or bid, wall_size
        if imb.inverse_ratio >= self.imbalance_ratio and imb.ask_size >= imb.bid_size:
            wall_price, wall_size = self._largest_level(
                book.asks_sorted()[: self.imbalance_max_levels],
                ceiling=imb.mid + imb.mid * (self.imbalance_band_bps / 1e4),
                side="ASK",
            )
            return "SELL", "ASK", wall_price or ask, wall_size
        return None, "", 0.0, 0.0

    @staticmethod
    def _largest_level(
        levels,
        *,
        floor: Optional[float] = None,
        ceiling: Optional[float] = None,
        side: str = "BID",
    ) -> tuple[Optional[float], float]:
        """Return ``(price, size)`` of the largest in-band level on ``side``.

        Caller pre-filters with ``bids_sorted``/``asks_sorted``; we just
        clamp to the price band and pick the max-size row. Returns
        ``(None, 0.0)`` when no level survives the band filter.
        """
        best_price: Optional[float] = None
        best_size: float = 0.0
        for lvl in levels:
            if side == "BID" and floor is not None and lvl.price < floor:
                continue
            if side == "ASK" and ceiling is not None and lvl.price > ceiling:
                continue
            if lvl.size > best_size:
                best_size = lvl.size
                best_price = lvl.price
        return best_price, best_size

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
        """Emit a flattening signal if any exit rule is tripped.

        Priority (highest first):
            1. Spoofing — wall on the favoured side vanished within
               ``spoof_persist_seconds`` of entry.
            2. Hold-time elapsed — ``now − entry_time > max_hold_seconds``.
            3. USD or % stop-loss.
            4. USD or % take-profit.

        Spoofing wins over the price-based exits because the structural
        thesis (the wall is real support) has been falsified — there's no
        reason to wait for a 3-bp move when the cause for being long is
        gone.
        """
        if st.position_side is None or st.entry_mid <= 0:
            return None

        side = st.position_side
        close_side = "SELL" if side == "LONG" else "BUY"
        limit_price = bid if side == "LONG" else ask
        try:
            leverage = float(market.metadata.get("leverage", 1.0) or 1.0)
        except (TypeError, ValueError):
            leverage = 1.0

        # --- 1. spoofing ----------------------------------------------
        if (
            st.wall is not None
            and self.spoof_persist_seconds > 0
            and st.wall.size >= self.spoof_min_wall_contracts
            and st.wall.vanished(
                outcome.book,
                ratio_threshold=self.spoof_vanish_ratio,
                now=now,
                persist_seconds=self.spoof_persist_seconds,
            )
        ):
            remaining = st.wall.current_size(outcome.book)
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind="spoof-exit",
                reason=(
                    f"wall {st.wall.size:g}->{remaining:g} @ {st.wall.price:.4f} "
                    f"shrunk past {self.spoof_vanish_ratio:.0%} in "
                    f"{now - st.wall.observed_at:.2f}s"
                ),
                extra_metadata={
                    "wall_initial": st.wall.size,
                    "wall_remaining": remaining,
                    "wall_age_seconds": now - st.wall.observed_at,
                },
            )

        # --- 2. hold-time --------------------------------------------
        if self.max_hold_seconds > 0 and (now - st.entry_time) > self.max_hold_seconds:
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind="hold-timeout",
                reason=(
                    f"held {now - st.entry_time:.2f}s > "
                    f"{self.max_hold_seconds:.2f}s"
                ),
                extra_metadata={
                    "hold_seconds": now - st.entry_time,
                },
            )

        # --- 3 + 4: TP / SL -------------------------------------------
        # USD-P&L path mirrors the mean-reversion scalper: requires a live
        # portfolio attached and at least one USD threshold > 0.
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
                # Position is already flat (e.g. fill never landed); reset
                # internal book and exit silently.
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

        if not (hit_sl or hit_tp):
            return None

        kind = "stop-loss" if hit_sl else "take-profit"
        move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
        pnl_label = f" pnl=${pnl_usd:+.2f}" if pnl_usd is not None else ""
        return self._build_close(
            st, market, outcome, side, close_side, limit_price, leverage,
            kind=kind,
            reason=(
                f"entry={st.entry_mid:.4f} mid={mid:.4f} ({move_bps:+.1f}bps){pnl_label}"
            ),
            extra_metadata=(
                {"pnl_usd": float(pnl_usd)} if pnl_usd is not None else {}
            ),
        )

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
        """Build a reduce-only flattening signal and reset internal state."""
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
            "sniper EXIT %s %s side=%s entry=%.4f close=%.4f (%s)",
            market.id, kind.upper(), position_side, entry_mid, limit_price, reason,
        )
        return sig

    @staticmethod
    def _reset_position(st: _SymbolState) -> None:
        st.position_side = None
        st.entry_mid = 0.0
        st.entry_size_usd = 0.0
        st.entry_time = 0.0
        st.wall = None
