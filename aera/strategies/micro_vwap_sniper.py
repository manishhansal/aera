"""Micro VWAP Reversion Sniper — fade short-term deviations from micro-VWAP.

Quiet, intraday mean-reversion play around a 1-minute rolling VWAP.
When price strays > ``deviation_pct`` from the rolling micro-VWAP AND
short-window volume is meaningfully below the longer baseline, that
combo reads as exhausted aggressors: the move ran without follow-on
volume and tends to snap back to the VWAP value at the time of the
deviation. Targets the VWAP-at-entry as a static snapshot (so the bar
doesn't drift with the very mean we're trying to revert to), stops on
extension, and force-exits on a hard hold timeout.

Spec → code mapping
-------------------

1. **Calculate.** Rolling 1-minute VWAP, recomputed every scan from a
   :class:`VWAPStream` that infers per-print volume from book deltas
   (no separate trades-channel subscription required). Formula:
   ``Σ(price × size) / Σ(size)`` over the last ``vwap_window_seconds``
   (default 60 s).
2. **Signal.** ``|mid − vwap| / vwap > deviation_pct`` (spec: 0.12%)
   **AND** ``vol_rate_short / vol_rate_long < volume_ratio_max``
   (spec: < 70% — short window is *quieter* than the long baseline,
   meaning the run drained aggression and follow-on flow is absent).
3. **Enter.** LONG when mid is below VWAP by ``> deviation_pct``;
   SHORT when above. Limit at the current touch (best ask for BUY,
   best bid for SELL); the executor's live path already submits with
   ``time_in_force=ioc``, matching the spec's 1 s IOC.
4. **Take profit.** Primary target = ``vwap_at_entry`` (snapshot at
   the fire time — never re-computed). Mid returning to that level
   from below (long) / above (short) closes the position. A secondary
   "VWAP + ``tp_extra_bps``" target is also supported as a stretch
   exit; ``0`` disables it.
5. **Stop loss.** Fixed % stop relative to entry mid (``stop_loss_pct``).
   The spec phrases it as "0.05% extended dev from VWAP" which, after
   a 0.12% entry deviation, lines up with ~0.05% further extension on
   mid. We model the simpler "% off entry mid" form for symmetry with
   the other scalpers.
6. **Time exit.** ``max_hold_seconds`` (default 90 s) hard stop. Past
   that, VWAP reversions either happen or fade — overstaying decays
   the edge.
7. **Avoid windows.**

   * Skip the first ``hour_skip_seconds`` (default 300 s) of each
     wall-clock hour. The rolling-VWAP buffer drains across that
     boundary and produces transient false deviations.
   * Skip when the current spread / mid exceeds ``max_spread_pct``
     (spec: 0.05%). Wider spreads can't clear the round-trip cost.

Sizing / leverage / risk vetting all live in the executor; the
strategy only emits ``Signal``\\ s.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market
from aera.signals.vwap_stream import VWAPStream

from .base import Leg, Signal, Strategy

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution -> strategies)
    from aera.core import Portfolio


log = get_logger(__name__)


@dataclass
class _SymbolState:
    """Per-symbol working state for the VWAP sniper."""

    stream: VWAPStream
    position_side: Optional[str] = None       # "LONG", "SHORT", or None
    entry_mid: float = 0.0
    entry_vwap: float = 0.0
    entry_size_usd: float = 0.0
    entry_time: float = 0.0
    last_signal_mid: float = 0.0


class MicroVWAPSniper(Strategy):
    """Mean-revert short-term deviations from a 1-minute micro-VWAP.

    Parameters
    ----------
    vwap_window_seconds : float
        Window over which the micro-VWAP is computed. Spec: 60 s.
    deviation_pct : float
        Required ``|mid − vwap| / vwap`` to fire an entry. Spec: 0.0012
        (0.12%). Larger values trade fewer but cleaner reversions.
    volume_short_seconds, volume_long_seconds : float
        Windows for the volume drop-off check. Spec: 10 s vs 300 s
        (5 min). The 5-minute window must contain volume for the ratio
        to compute — strategies cold-start with no signal.
    volume_ratio_max : float
        ``short_rate / long_rate`` must be ``< volume_ratio_max`` for
        the setup to fire (i.e. the recent window is *quieter*).
        Spec: 0.70.
    take_profit_pct : float
        Floor / safety TP relative to entry mid. Used only when the
        VWAP snap-back path can't fire (e.g. entry-VWAP got polluted).
        ``0`` disables.
    tp_extra_bps : float
        Stretch target beyond VWAP-at-entry, in bps in the favourable
        direction (spec: 3 bps = 0.03%). When > 0 the take-profit
        target is shifted past VWAP by this much. ``0`` snaps exactly
        at the VWAP-at-entry, matching the spec primary TP.
    stop_loss_pct : float
        Hard % stop relative to entry mid. Spec: 0.0005 (0.05%).
    take_profit_usd, stop_loss_usd : float
        USD-PnL exits. When both > 0 and ``portfolio`` is attached
        they take precedence over the % thresholds (matches the other
        strategies' contract).
    max_hold_seconds : float
        Force a reduce-only close ``max_hold_seconds`` after entry.
        Spec: 90 s. ``0`` disables.
    max_spread_pct : float
        Skip an entry when ``spread / mid`` exceeds this fraction.
        Spec: 0.0005 (0.05%). ``0`` disables.
    hour_skip_seconds : float
        Skip entries during the first N seconds of each wall-clock
        hour. Spec: 300 s (the first 5 minutes). ``0`` disables. The
        check uses ``now mod 3600`` so it's deterministic against the
        injected clock — perfect for tests.
    leverage_override : float, optional
        Stamp this leverage on every emitted leg (spec: 5×). ``None``
        falls through to the venue's account leverage in
        ``market.metadata``.
    notional_usd : float
        Reference notional. The executor's ``trade_size_fraction``
        typically overrides this.
    rearm_distance_bps : float
        Don't refire on the same symbol until mid has moved at least
        this many bps from the previous firing mid. Cheap debouncer.
    min_edge : float
        Floor on emitted ``Signal.edge`` so VWAP signals sort sensibly
        against the other strategies' queue.
    portfolio : Portfolio, optional
        Live portfolio used by the USD-PnL exit path.
    clock : callable, optional
        Overridable time source. Defaults to ``time.time``. Tests use
        a manually-advanced clock to drive deterministic VWAP windows
        and the "first 5 minutes of hour" filter.
    """

    name = "micro_vwap_sniper"

    def __init__(
        self,
        *,
        vwap_window_seconds: float = 60.0,
        deviation_pct: float = 0.0012,
        volume_short_seconds: float = 10.0,
        volume_long_seconds: float = 300.0,
        volume_ratio_max: float = 0.70,
        take_profit_pct: float = 0.0007,
        tp_extra_bps: float = 0.0,
        stop_loss_pct: float = 0.0005,
        take_profit_usd: float = 0.0,
        stop_loss_usd: float = 0.0,
        max_hold_seconds: float = 90.0,
        max_spread_pct: float = 0.0005,
        hour_skip_seconds: float = 300.0,
        leverage_override: Optional[float] = 5.0,
        notional_usd: float = 1000.0,
        rearm_distance_bps: float = 3.0,
        min_edge: float = 0.0005,
        portfolio: Optional["Portfolio"] = None,
        enabled: bool = True,
        clock: Optional[callable] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.vwap_window_seconds = max(1.0, float(vwap_window_seconds))
        self.deviation_pct = max(0.0, float(deviation_pct))
        self.volume_short_seconds = max(0.1, float(volume_short_seconds))
        self.volume_long_seconds = max(
            self.volume_short_seconds * 2.0, float(volume_long_seconds)
        )
        self.volume_ratio_max = max(0.0, float(volume_ratio_max))
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.tp_extra_bps = max(0.0, float(tp_extra_bps))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.take_profit_usd = max(0.0, float(take_profit_usd))
        self.stop_loss_usd = max(0.0, float(stop_loss_usd))
        self.max_hold_seconds = max(0.0, float(max_hold_seconds))
        self.max_spread_pct = max(0.0, float(max_spread_pct))
        self.hour_skip_seconds = max(0.0, float(hour_skip_seconds))
        self.leverage_override = (
            float(leverage_override) if leverage_override is not None else None
        )
        self.notional_usd = max(0.0, float(notional_usd))
        self.rearm_distance_bps = max(0.0, float(rearm_distance_bps))
        self.min_edge = max(0.0, float(min_edge))
        self.portfolio = portfolio
        self._clock = clock or time.time
        self._state: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # scan loop
    # ------------------------------------------------------------------

    def scan(self, markets: Iterable[Market]) -> List[Signal]:
        signals: List[Signal] = []
        now = float(self._clock())
        in_skip_window = (
            self.hour_skip_seconds > 0
            and (now % 3600.0) < self.hour_skip_seconds
        )

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
            # Always feed the stream so VWAP / volume baselines keep
            # tracking even while we're positioned — the next setup
            # post-exit must use a fresh baseline.
            st.stream.update(book, now=now)

            # Exit path always wins on the tick a fresh entry might want.
            close = self._maybe_emit_close(
                st, m, outcome, bid_lvl.price, ask_lvl.price, mid, now,
            )
            if close is not None:
                signals.append(close)
                continue

            if st.position_side is not None:
                continue

            # Spec: skip the first hour_skip_seconds of each wall-clock
            # hour; the rolling-VWAP buffer drains across that boundary
            # and produces transient false deviations.
            if in_skip_window:
                continue

            # Spread guard — skip when the round-trip can't be cleared.
            if self.max_spread_pct > 0:
                spread = ask_lvl.price - bid_lvl.price
                if spread / mid > self.max_spread_pct:
                    continue

            vwap = st.stream.vwap(self.vwap_window_seconds, now=now)
            if vwap is None or vwap <= 0:
                continue

            deviation = (mid - vwap) / vwap
            if abs(deviation) < self.deviation_pct:
                continue

            # Volume drop-off: the recent window must be QUIETER than
            # the long baseline (= exhausted aggressors). The opposite
            # of the spike-veto path the other scalpers use.
            ratio = st.stream.volume_ratio(
                short_seconds=self.volume_short_seconds,
                long_seconds=self.volume_long_seconds,
                now=now,
            )
            if ratio is None:
                continue   # cold start — wait for baseline to populate
            if ratio >= self.volume_ratio_max:
                continue

            # Rearm debounce.
            if st.last_signal_mid > 0:
                move_bps = abs(mid - st.last_signal_mid) / st.last_signal_mid * 1e4
                if move_bps < self.rearm_distance_bps:
                    continue

            # FADE direction: long when mid is BELOW vwap, short when
            # above. Limit at the touch (best ask for BUY, best bid for
            # SELL) — the executor's live path runs with IOC.
            entry_side = "BUY" if deviation < 0 else "SELL"
            limit_price = ask_lvl.price if entry_side == "BUY" else bid_lvl.price

            leverage = self._leg_leverage(m)

            # Edge — proportional to how far we strayed from VWAP,
            # capped at 1% so it sorts above the basic scalpers
            # without dominating the queue.
            strength = min(1.0, abs(deviation) / max(self.deviation_pct * 4.0, 1e-9))
            edge = max(self.min_edge, strength * 0.01)

            reason = (
                f"dev={deviation*100:+.4f}% vwap={vwap:.4f} "
                f"vol_ratio={ratio:.2f} (<{self.volume_ratio_max:g})"
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
                        "mid": float(mid),
                        "vwap": float(vwap),
                        "deviation_pct": float(deviation),
                        "volume_ratio": float(ratio),
                    },
                )
            )

            st.last_signal_mid = mid
            st.position_side = "LONG" if entry_side == "BUY" else "SHORT"
            st.entry_mid = mid
            st.entry_vwap = vwap
            st.entry_size_usd = self.notional_usd
            st.entry_time = now
            log.info(
                "micro-vwap FIRE %s %s mid=%.4f vwap=%.4f dev=%+.4f%% vol_ratio=%.2f",
                m.id, entry_side, mid, vwap, deviation * 100, ratio,
            )

        return signals

    # ------------------------------------------------------------------
    # state plumbing
    # ------------------------------------------------------------------

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState(
                stream=VWAPStream(window_seconds=self.vwap_window_seconds),
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
        """Emit a reduce-only close on hold-timeout / SL / VWAP snap-back.

        Priority (highest first):
            1. Hold-timeout — ``now − entry_time > max_hold_seconds``.
            2. USD or % stop-loss (extension past entry).
            3. VWAP snap-back to ``entry_vwap`` (with optional
               ``tp_extra_bps`` stretch past the VWAP target).
            4. Floor ``take_profit_pct`` (only when VWAP target is
               unavailable / not yet hit).
        """
        if st.position_side is None or st.entry_mid <= 0:
            return None

        side = st.position_side
        close_side = "SELL" if side == "LONG" else "BUY"
        # Close at the opposite touch — that's where a market order
        # would actually fill.
        limit_price = bid if side == "LONG" else ask
        leverage = self._leg_leverage(market)

        # --- 1. hold-timeout ----------------------------------------
        if self.max_hold_seconds > 0 and (now - st.entry_time) > self.max_hold_seconds:
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind="hold-timeout",
                reason=f"held {now - st.entry_time:.2f}s > {self.max_hold_seconds:.2f}s",
                extra_metadata={"hold_seconds": now - st.entry_time},
            )

        # --- 2. USD or % stop-loss ----------------------------------
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
            sl = self.stop_loss_pct
            entry = st.entry_mid
            if side == "LONG":
                if sl > 0 and mid <= entry * (1.0 - sl):
                    hit_sl = True
            else:
                if sl > 0 and mid >= entry * (1.0 + sl):
                    hit_sl = True

        if hit_sl:
            move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
            pnl_label = f" pnl=${pnl_usd:+.2f}" if pnl_usd is not None else ""
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind="stop-loss",
                reason=(
                    f"entry={st.entry_mid:.4f} mid={mid:.4f} "
                    f"({move_bps:+.1f}bps){pnl_label}"
                ),
                extra_metadata=({"pnl_usd": float(pnl_usd)} if pnl_usd is not None else {}),
            )

        # --- 3. VWAP snap-back (primary TP per spec) ----------------
        if st.entry_vwap > 0:
            extra = st.entry_vwap * (self.tp_extra_bps / 1e4)
            if side == "LONG":
                target = st.entry_vwap + extra
                if mid >= target:
                    move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
                    return self._build_close(
                        st, market, outcome, side, close_side, limit_price, leverage,
                        kind="vwap-snapback",
                        reason=(
                            f"entry={st.entry_mid:.4f} vwap={st.entry_vwap:.4f} "
                            f"target={target:.4f} mid={mid:.4f} ({move_bps:+.1f}bps)"
                        ),
                        extra_metadata={
                            "entry_vwap": float(st.entry_vwap),
                            "vwap_target": float(target),
                        },
                    )
            else:
                target = st.entry_vwap - extra
                if mid <= target:
                    move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
                    return self._build_close(
                        st, market, outcome, side, close_side, limit_price, leverage,
                        kind="vwap-snapback",
                        reason=(
                            f"entry={st.entry_mid:.4f} vwap={st.entry_vwap:.4f} "
                            f"target={target:.4f} mid={mid:.4f} ({move_bps:+.1f}bps)"
                        ),
                        extra_metadata={
                            "entry_vwap": float(st.entry_vwap),
                            "vwap_target": float(target),
                        },
                    )

        # --- 4. USD or % take-profit (fallback) ---------------------
        if not usd_active and self.take_profit_pct > 0:
            tp = self.take_profit_pct
            entry = st.entry_mid
            if side == "LONG":
                if mid >= entry * (1.0 + tp):
                    hit_tp = True
            else:
                if mid <= entry * (1.0 - tp):
                    hit_tp = True

        if hit_tp:
            move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
            pnl_label = f" pnl=${pnl_usd:+.2f}" if pnl_usd is not None else ""
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind="take-profit",
                reason=(
                    f"entry={st.entry_mid:.4f} mid={mid:.4f} "
                    f"({move_bps:+.1f}bps){pnl_label}"
                ),
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
            "micro-vwap EXIT %s %s side=%s entry=%.4f close=%.4f (%s)",
            market.id, kind.upper(), position_side, entry_mid, limit_price, reason,
        )
        return sig

    @staticmethod
    def _reset_position(st: _SymbolState) -> None:
        st.position_side = None
        st.entry_mid = 0.0
        st.entry_vwap = 0.0
        st.entry_size_usd = 0.0
        st.entry_time = 0.0
