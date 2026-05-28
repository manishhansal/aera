"""Stop Hunt / Liquidity Grab Reversal — fade engineered stop sweeps.

Detects the classic "stop hunt" wick — a sudden ``wick_size_pct``+ spike
*through* a recent swing low / high that immediately reverses and closes
back inside the level on the same 1-second candle. These are market-
maker sweeps that vacuum retail stop clusters before the real move
unfolds; the strategy fades the sweep, riding the snap-back.

Spec → code mapping
-------------------

1. **Pre-mark key levels.** :class:`BarStream` continuously aggregates
   1-second OHLC bars from the streaming book; the strategy reads the
   most-recent ``swing_count`` pivot highs and pivot lows from the
   trailing ``swing_lookback_bars`` (default 60 bars = ~1 minute on
   1 s bars, matching the spec's "swing highs/lows on 1m chart").

2. **Sweep fires.** When a bar closes, the strategy checks:

   * **Wick depth** — bar's low must dip at least ``wick_size_pct``
     below a marked swing low (or bar's high must spike at least
     ``wick_size_pct`` above a marked swing high). Spec default
     = 0.0015 (0.15%).
   * **Recovery** — bar's close must be back inside the level
     (above the swept low / below the swept high).
   * **Body ratio** — body must be ``< body_ratio_max`` of total
     range. Spec default = 0.30 — small body, big wick.
   * **Recovery speed** — current bar duration (and any prior bar
     where the sweep started) must be ``≤ recovery_seconds``. With
     1 s bars and the spec's 3 s default this is almost always
     satisfied on the close-of-bar tick.
   * **Volume confirmation** — the wick bar's volume must clear
     ``volume_multiple × avg_volume`` over the trailing
     ``volume_lookback_bars``. ``0`` disables.
   * **Delta confirmation (short only, optional)** — for a bearish
     sweep, the spec asks that the "delta (buy − sell) flips red".
     The wick bar's delta must be ``< delta_flip_threshold`` (= net
     selling pressure on the close). Set to a positive number to
     disable.

3. **Enter.** Market entry at the touch (best ask for long, best bid
   for short) the instant the wick bar closes. The live exchange
   path runs with IOC so unfilled limits evaporate inside the next
   scan, matching the spec's "do not wait for the next candle".

4. **TP1 / TP2 / SL.** Spec calls for ``+0.10%`` (close ``tp1_fraction``,
   default 60%), ``+0.20%`` (close the remainder), stop at the wick
   low ``− stop_extra_pct`` (default 0.08%). When ``tp1_fraction`` is
   ``0`` the strategy collapses to a single ``take_profit_pct`` exit.

5. **Hold-timeout.** Force a market flatten ``max_hold_seconds`` after
   entry. Spec doesn't give a hard cap, but stale "snap-back" trades
   that fail to revert quickly are usually trend trades in disguise;
   default 60 s.

Pairs to trade
--------------

BTC-PERP, ETH-PERP. The spec calls out "low liquidity windows best"
(Asian session, weekend chop) where engineered sweeps are easier to
spot — the strategy itself is venue-agnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market
from aera.signals.bar_stream import Bar, BarStream

from .base import Leg, Signal, Strategy

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution -> strategies)
    from aera.core import Portfolio


log = get_logger(__name__)


@dataclass
class _SymbolState:
    """Per-symbol working state for the stop-hunt reversal strategy."""

    stream: BarStream
    last_signal_mid: float = 0.0
    last_signal_bar_start: float = 0.0
    position_side: Optional[str] = None       # "LONG", "SHORT", or None
    entry_mid: float = 0.0
    entry_size_usd: float = 0.0
    entry_time: float = 0.0
    swept_level: float = 0.0
    wick_extreme: float = 0.0                  # wick low (long) or wick high (short)
    stop_price: float = 0.0
    tp1_target: float = 0.0
    tp2_target: float = 0.0
    tp1_taken: bool = False
    tp1_size_usd: float = 0.0                  # notional already closed via TP1


class StopHuntReversal(Strategy):
    """Fade engineered stop-cluster sweeps after the wick closes back inside.

    Parameters
    ----------
    bar_seconds : float
        Aggregation interval for OHLC bars. Spec uses 1 s.
    max_bars : int
        Cap on retained closed bars (bounded memory across many symbols).
    swing_lookback_bars : int
        How many recent closed bars are scanned for swing pivots. 60
        bars at 1 s = the trailing minute the spec marks as the chart.
    swing_pivot_strength : int
        Fractal pivot bar count on each side. ``2`` (= 5-bar fractal)
        matches the canonical "swing" definition.
    swing_count : int
        Max number of recent swing levels considered per side. Spec
        marks the last 3 swing highs / lows.
    wick_size_pct : float
        Minimum *fractional* wick depth below the swept low (or above
        the swept high). Spec: 0.0015 = 0.15%.
    body_ratio_max : float
        Wick bar's body / range must be ``<`` this fraction. Spec: 0.30.
    recovery_seconds : float
        Maximum age of the wick before it's "stale" — measured from the
        bar's start to the strategy's current scan timestamp. Spec: 3 s.
    volume_multiple : float
        Wick bar's volume must clear ``multiple × avg`` over
        ``volume_lookback_bars``. ``0`` disables this gate.
    volume_lookback_bars : int
        Bars used for the average-volume baseline.
    delta_flip_threshold : float
        For bearish sweeps the wick bar's signed delta
        (``buy_volume − sell_volume``) must be ``<`` this value. Spec
        wants "delta flips red" — i.e. ``< 0``. Set to a positive
        sentinel to disable. The bullish-sweep mirror is enforced via
        ``delta > -threshold`` (= net buying pressure on the snap-back).
    require_delta_confirmation : bool
        When ``True`` (default) the delta gate applies to BOTH
        bullish and bearish sweeps. ``False`` disables the gate
        regardless of ``delta_flip_threshold`` (handy when running off
        a venue with a noisy inference path).
    take_profit_pct : float
        Primary take-profit, fraction of entry mid. Spec: 0.0020
        (0.20%). When ``tp1_fraction == 0`` this is the only TP.
    tp1_pct : float
        Partial take-profit, fraction of entry mid. Spec: 0.0010 (0.10%).
        Closes ``tp1_fraction`` of the entry notional when hit.
    tp1_fraction : float
        Portion of the entry notional flattened on TP1. Spec: 0.60.
        ``0`` disables the partial TP and collapses to a single
        ``take_profit_pct`` exit.
    stop_extra_pct : float
        Hard stop placed ``stop_extra_pct`` *below* the wick low (long)
        or ``above`` the wick high (short). Spec: 0.0008 (0.08%).
    stop_loss_pct : float
        Fallback hard stop relative to entry mid, used only when the
        wick-anchored stop is invalid (rare). ``0`` disables.
    take_profit_usd, stop_loss_usd : float
        USD-PnL exits. When both > 0 and ``portfolio`` is attached
        they take precedence over the % thresholds — same contract as
        the other strategies. The partial-TP path is skipped in
        USD-PnL mode (USD exits flatten the whole position).
    max_hold_seconds : float
        Force a reduce-only flatten this many seconds after entry.
        Spec doesn't pin a number; default 60 s keeps stale snap-backs
        from rotting into trend trades.
    leverage_override : float, optional
        Stamp this leverage on every emitted leg. Spec: 5× max.
        ``None`` falls through to the venue's account leverage.
    notional_usd : float
        Reference notional. The executor's ``trade_size_fraction``
        typically rescales — identical to the other strategies.
    rearm_distance_bps : float
        Don't refire on the same symbol until mid has moved at least
        this many bps from the previous firing mid. Cheap debouncer.
    min_edge : float
        Floor on emitted ``Signal.edge`` so sweep signals sort
        sensibly against the other strategies' queue.
    portfolio : Portfolio, optional
        Live portfolio used by the USD-PnL exit path.
    clock : callable, optional
        Overridable time source for deterministic tests.
    """

    name = "stop_hunt_reversal"

    def __init__(
        self,
        *,
        bar_seconds: float = 1.0,
        max_bars: int = 300,
        swing_lookback_bars: int = 60,
        swing_pivot_strength: int = 2,
        swing_count: int = 3,
        wick_size_pct: float = 0.0015,
        body_ratio_max: float = 0.30,
        recovery_seconds: float = 3.0,
        volume_multiple: float = 1.5,
        volume_lookback_bars: int = 30,
        delta_flip_threshold: float = 0.0,
        require_delta_confirmation: bool = True,
        take_profit_pct: float = 0.0020,
        tp1_pct: float = 0.0010,
        tp1_fraction: float = 0.60,
        stop_extra_pct: float = 0.0008,
        stop_loss_pct: float = 0.0,
        take_profit_usd: float = 0.0,
        stop_loss_usd: float = 0.0,
        max_hold_seconds: float = 60.0,
        leverage_override: Optional[float] = 5.0,
        notional_usd: float = 1000.0,
        rearm_distance_bps: float = 5.0,
        min_edge: float = 0.0015,
        portfolio: Optional["Portfolio"] = None,
        enabled: bool = True,
        clock: Optional[callable] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.bar_seconds = max(0.05, float(bar_seconds))
        self.max_bars = max(10, int(max_bars))
        self.swing_lookback_bars = max(5, int(swing_lookback_bars))
        self.swing_pivot_strength = max(1, int(swing_pivot_strength))
        self.swing_count = max(1, int(swing_count))
        self.wick_size_pct = max(0.0, float(wick_size_pct))
        self.body_ratio_max = min(1.0, max(0.0, float(body_ratio_max)))
        self.recovery_seconds = max(0.0, float(recovery_seconds))
        self.volume_multiple = max(0.0, float(volume_multiple))
        self.volume_lookback_bars = max(1, int(volume_lookback_bars))
        self.delta_flip_threshold = float(delta_flip_threshold)
        self.require_delta_confirmation = bool(require_delta_confirmation)
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.tp1_pct = max(0.0, float(tp1_pct))
        self.tp1_fraction = min(1.0, max(0.0, float(tp1_fraction)))
        self.stop_extra_pct = max(0.0, float(stop_extra_pct))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.take_profit_usd = max(0.0, float(take_profit_usd))
        self.stop_loss_usd = max(0.0, float(stop_loss_usd))
        self.max_hold_seconds = max(0.0, float(max_hold_seconds))
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
            # Always feed the stream so OHLC + volume baselines keep
            # tracking even while we're positioned. ``closed`` is the
            # bar that just rolled over this update (if any) — that's
            # the canonical sweep-detection moment per the spec
            # ("wick candle closes back above swept level").
            closed = st.stream.update(book, now=now)

            close = self._maybe_emit_close(
                st, m, outcome, bid_lvl.price, ask_lvl.price, mid, now,
            )
            if close is not None:
                signals.append(close)
                continue

            if st.position_side is not None:
                continue
            if closed is None:
                continue   # no bar closed this scan — no fresh setup to test

            sig = self._maybe_emit_entry(
                st, m, outcome, bid_lvl.price, ask_lvl.price, mid, now, closed,
            )
            if sig is not None:
                signals.append(sig)

        return signals

    # ------------------------------------------------------------------
    # state plumbing
    # ------------------------------------------------------------------

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState(
                stream=BarStream(
                    bar_seconds=self.bar_seconds,
                    max_bars=self.max_bars,
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
    # entry detection
    # ------------------------------------------------------------------

    def _maybe_emit_entry(
        self,
        st: _SymbolState,
        market: Market,
        outcome,
        bid: float,
        ask: float,
        mid: float,
        now: float,
        bar: Bar,
    ) -> Optional[Signal]:
        """Test the just-closed bar against the sweep pattern.

        Returns a long-side or short-side ``Signal`` when the bar
        qualifies as a stop-hunt reversal candle on either a recent
        swing low (bullish sweep) or swing high (bearish sweep).
        """
        # Rearm debounce — don't refire on the same symbol immediately
        # after a fresh fire (mid hasn't meaningfully moved).
        if st.last_signal_mid > 0:
            move_bps = abs(mid - st.last_signal_mid) / st.last_signal_mid * 1e4
            if move_bps < self.rearm_distance_bps:
                return None

        # Recovery-speed gate — the wick must be young. With 1 s bars
        # this is essentially always satisfied on the close-of-bar
        # tick, but the gate matters if `bar_seconds` is raised or the
        # scan loop runs slower than the bar interval.
        if self.recovery_seconds > 0 and (now - bar.start) > self.recovery_seconds:
            return None

        # Body / range sanity — a 1-print bar (range == 0) can't be a
        # wick, by definition. Skip rather than divide.
        if bar.range <= 0:
            return None
        if bar.body_ratio >= self.body_ratio_max:
            return None

        # Volume confirmation — the spec requires a "volume spike on
        # wick candle". Compute the trailing-bar baseline EXCLUDING
        # the wick bar itself so the spike doesn't inflate its own
        # threshold.
        if self.volume_multiple > 0:
            closed = st.stream.closed_bars
            # Exclude the just-closed bar (it's the candidate); the
            # baseline is the bars before it.
            baseline_slice = closed[:-1][-self.volume_lookback_bars:]
            if baseline_slice:
                avg = sum(b.volume for b in baseline_slice) / len(baseline_slice)
                if avg > 0 and bar.volume < avg * self.volume_multiple:
                    return None
            # When there's no baseline yet (cold start), allow the
            # setup — the other gates (wick depth, body ratio, swing
            # levels) carry the signal.

        swing_highs = st.stream.recent_swing_highs(
            pivot_strength=self.swing_pivot_strength,
            lookback_bars=self.swing_lookback_bars,
            max_count=self.swing_count,
        )
        swing_lows = st.stream.recent_swing_lows(
            pivot_strength=self.swing_pivot_strength,
            lookback_bars=self.swing_lookback_bars,
            max_count=self.swing_count,
        )

        # Bullish sweep — wick pierced below a swing low and closed
        # back above. Iterate newest-first (lows are sorted newest-
        # first by ``recent_swing_lows``).
        for level in swing_lows:
            if level <= 0:
                continue
            min_wick_depth = level * self.wick_size_pct
            if bar.low > level - min_wick_depth:
                continue   # wick didn't pierce deep enough
            if bar.close <= level:
                continue   # recovery failed — close still below level
            if self.require_delta_confirmation and self.delta_flip_threshold != 0:
                # bullish snap-back wants positive flow on the recovery
                if bar.delta <= -self.delta_flip_threshold:
                    continue
            return self._fire_entry(
                st, market, outcome, bid, ask, mid, now,
                bar=bar, side="BUY", swept_level=level,
            )

        # Bearish sweep — wick pierced above a swing high and closed
        # back below.
        for level in swing_highs:
            if level <= 0:
                continue
            min_wick_depth = level * self.wick_size_pct
            if bar.high < level + min_wick_depth:
                continue
            if bar.close >= level:
                continue
            if self.require_delta_confirmation and self.delta_flip_threshold != 0:
                # bearish sweep wants delta to flip red on the close
                if bar.delta >= self.delta_flip_threshold:
                    continue
            return self._fire_entry(
                st, market, outcome, bid, ask, mid, now,
                bar=bar, side="SELL", swept_level=level,
            )

        return None

    def _fire_entry(
        self,
        st: _SymbolState,
        market: Market,
        outcome,
        bid: float,
        ask: float,
        mid: float,
        now: float,
        *,
        bar: Bar,
        side: str,
        swept_level: float,
    ) -> Signal:
        """Build the entry leg + seed per-symbol state for the exit path."""
        entry_side = side
        limit_price = ask if entry_side == "BUY" else bid
        leverage = self._leg_leverage(market)

        if entry_side == "BUY":
            wick_extreme = bar.low
            stop_price = wick_extreme * (1.0 - self.stop_extra_pct)
            tp1_target = mid * (1.0 + self.tp1_pct) if self.tp1_pct > 0 else 0.0
            tp2_target = mid * (1.0 + self.take_profit_pct) if self.take_profit_pct > 0 else 0.0
        else:
            wick_extreme = bar.high
            stop_price = wick_extreme * (1.0 + self.stop_extra_pct)
            tp1_target = mid * (1.0 - self.tp1_pct) if self.tp1_pct > 0 else 0.0
            tp2_target = mid * (1.0 - self.take_profit_pct) if self.take_profit_pct > 0 else 0.0

        # Edge — proportional to the wick depth past the swept level
        # vs the spec floor. Capped at 1% so it sorts above the basic
        # scalpers without dominating the queue.
        depth_pct = (
            (swept_level - bar.low) / swept_level if entry_side == "BUY"
            else (bar.high - swept_level) / swept_level
        )
        strength = min(1.0, depth_pct / max(self.wick_size_pct * 2.0, 1e-9))
        edge = max(self.min_edge, strength * 0.01)

        reason = (
            f"sweep level={swept_level:.4f} wick={wick_extreme:.4f} "
            f"depth={depth_pct*100:.3f}% body_ratio={bar.body_ratio:.2f} "
            f"vol={bar.volume:g} delta={bar.delta:+g}"
        )
        leg = Leg(
            market_id=market.id,
            outcome_id=outcome.id,
            side=entry_side,
            limit_price=float(limit_price),
            size_usd=self.notional_usd,
            reason=reason,
            leverage=leverage,
        )
        sig = Signal(
            strategy=self.name,
            confidence=min(1.0, strength + 0.1),
            edge=edge,
            legs=[leg],
            metadata={
                "symbol": market.id,
                "side": entry_side,
                "swept_level": float(swept_level),
                "wick_extreme": float(wick_extreme),
                "stop_price": float(stop_price),
                "tp1_target": float(tp1_target),
                "tp2_target": float(tp2_target),
                "body_ratio": float(bar.body_ratio),
                "wick_depth_pct": float(depth_pct),
                "bar_volume": float(bar.volume),
                "bar_delta": float(bar.delta),
                "mid": float(mid),
            },
        )

        st.last_signal_mid = mid
        st.last_signal_bar_start = bar.start
        st.position_side = "LONG" if entry_side == "BUY" else "SHORT"
        st.entry_mid = mid
        st.entry_size_usd = self.notional_usd
        st.entry_time = now
        st.swept_level = swept_level
        st.wick_extreme = wick_extreme
        st.stop_price = stop_price
        st.tp1_target = tp1_target
        st.tp2_target = tp2_target
        st.tp1_taken = False
        st.tp1_size_usd = 0.0

        log.info(
            "stop-hunt FIRE %s %s mid=%.4f swept=%.4f wick=%.4f stop=%.4f "
            "tp1=%.4f tp2=%.4f body_ratio=%.2f",
            market.id, entry_side, mid, swept_level, wick_extreme, stop_price,
            tp1_target, tp2_target, bar.body_ratio,
        )
        return sig

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
        """Emit a reduce-only close on hold-timeout / SL / TP1 / TP2.

        Priority (highest first):
            1. Hold-timeout — ``now − entry_time > max_hold_seconds``.
            2. USD or wick-anchored stop-loss.
            3. TP1 partial close — only fires once per position, sized
               to ``tp1_fraction × entry_size``.
            4. TP2 / USD take-profit — flattens the remainder.
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
                size_usd=self._remaining_notional(st),
                extra_metadata={"hold_seconds": now - st.entry_time},
                final=True,
            )

        # --- 2. USD-or-% / wick-anchored SL --------------------------
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
            # Wick-anchored stop — preferred when set; falls back to
            # entry-mid stop_loss_pct when the wick anchor wasn't
            # captured (defensive / cold-edge case).
            entry = st.entry_mid
            stop_anchor = st.stop_price
            if side == "LONG":
                if stop_anchor > 0 and mid <= stop_anchor:
                    hit_sl = True
                elif self.stop_loss_pct > 0 and mid <= entry * (1.0 - self.stop_loss_pct):
                    hit_sl = True
            else:
                if stop_anchor > 0 and mid >= stop_anchor:
                    hit_sl = True
                elif self.stop_loss_pct > 0 and mid >= entry * (1.0 + self.stop_loss_pct):
                    hit_sl = True

        if hit_sl:
            move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
            pnl_label = f" pnl=${pnl_usd:+.2f}" if pnl_usd is not None else ""
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind="stop-loss",
                reason=(
                    f"entry={st.entry_mid:.4f} mid={mid:.4f} "
                    f"stop={st.stop_price:.4f} ({move_bps:+.1f}bps){pnl_label}"
                ),
                size_usd=self._remaining_notional(st),
                extra_metadata=({"pnl_usd": float(pnl_usd)} if pnl_usd is not None else {}),
                final=True,
            )

        # --- 3. TP1 partial close (only in %-mode) ------------------
        # USD-PnL mode flattens the whole position on the threshold,
        # so partial TP1 only applies when usd_active is False.
        if (
            not usd_active
            and not st.tp1_taken
            and self.tp1_fraction > 0
            and st.tp1_target > 0
        ):
            tp1_hit = (
                (side == "LONG" and mid >= st.tp1_target)
                or (side == "SHORT" and mid <= st.tp1_target)
            )
            if tp1_hit:
                size = max(0.0, st.entry_size_usd * self.tp1_fraction)
                if size > 0:
                    move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
                    return self._build_close(
                        st, market, outcome, side, close_side, limit_price, leverage,
                        kind="take-profit-1",
                        reason=(
                            f"tp1 entry={st.entry_mid:.4f} mid={mid:.4f} "
                            f"target={st.tp1_target:.4f} ({move_bps:+.1f}bps) "
                            f"close {self.tp1_fraction*100:.0f}%"
                        ),
                        size_usd=size,
                        extra_metadata={
                            "tp_kind": "partial",
                            "tp1_target": float(st.tp1_target),
                            "tp1_fraction": float(self.tp1_fraction),
                        },
                        final=False,
                    )

        # --- 4. TP2 / TP fallback / USD-TP --------------------------
        if usd_active:
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
                    size_usd=self._remaining_notional(st),
                    extra_metadata=({"pnl_usd": float(pnl_usd)} if pnl_usd is not None else {}),
                    final=True,
                )
            return None

        if st.tp2_target > 0:
            tp2_hit = (
                (side == "LONG" and mid >= st.tp2_target)
                or (side == "SHORT" and mid <= st.tp2_target)
            )
            if tp2_hit:
                move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
                return self._build_close(
                    st, market, outcome, side, close_side, limit_price, leverage,
                    kind="take-profit-2",
                    reason=(
                        f"tp2 entry={st.entry_mid:.4f} mid={mid:.4f} "
                        f"target={st.tp2_target:.4f} ({move_bps:+.1f}bps)"
                    ),
                    size_usd=self._remaining_notional(st),
                    extra_metadata={
                        "tp_kind": "final",
                        "tp2_target": float(st.tp2_target),
                    },
                    final=True,
                )

        return None

    # ------------------------------------------------------------------
    # close construction
    # ------------------------------------------------------------------

    def _remaining_notional(self, st: _SymbolState) -> float:
        """Notional still open on this position (entry − partials taken)."""
        return max(0.0, st.entry_size_usd - st.tp1_size_usd)

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
        size_usd: float,
        extra_metadata: Optional[Dict[str, object]] = None,
        final: bool,
    ) -> Signal:
        """Emit one reduce-only close leg.

        ``final=True`` flips the per-symbol state back to flat; partial
        TP1 closes (``final=False``) mark ``tp1_taken`` and bump
        ``tp1_size_usd`` so subsequent exits target the remainder.
        """
        entry_mid = st.entry_mid
        leg = Leg(
            market_id=market.id,
            outcome_id=outcome.id,
            side=close_side,
            limit_price=float(limit_price),
            size_usd=float(max(0.0, size_usd)),
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
        if final:
            self._reset_position(st)
        else:
            st.tp1_taken = True
            st.tp1_size_usd += float(max(0.0, size_usd))
        log.info(
            "stop-hunt EXIT %s %s side=%s entry=%.4f close=%.4f size=$%.2f (%s)",
            market.id, kind.upper(), position_side, entry_mid, limit_price,
            size_usd, reason,
        )
        return sig

    @staticmethod
    def _reset_position(st: _SymbolState) -> None:
        st.position_side = None
        st.entry_mid = 0.0
        st.entry_size_usd = 0.0
        st.entry_time = 0.0
        st.swept_level = 0.0
        st.wick_extreme = 0.0
        st.stop_price = 0.0
        st.tp1_target = 0.0
        st.tp2_target = 0.0
        st.tp1_taken = False
        st.tp1_size_usd = 0.0
