"""Tick Reversal Scalp — fade an N-tick exhaustion run.

When the mid prints N consecutive same-direction ticks and the eaten
per-tick liquidity decays end-to-end (momentum is weakening), the
strategy fires a small fade in the opposite direction. Targets a
micro-profit (~4 bps), stops tightly (~2.5 bps), and force-exits after
``max_hold_seconds`` so a stale reversal doesn't decay into a trend
position.

Spec → code mapping
-------------------

* "5+ consecutive downticks / upticks" → :meth:`TickStream.current_streak`
  with ``min_streak`` (default 5).
* "Trade size shrinking each tick" → :meth:`TickStream.size_decay` against
  ``size_decay_threshold`` (default 0.20). Compares first-tick eaten size
  to last-tick eaten size across the streak.
* "Price at prior S/R level" → ``recent_extreme`` over
  ``sr_lookback_ticks`` must be within ``sr_band_bps`` of the current mid.
  Permissive when the streak made new lows / highs (which it usually has
  done by definition) but tightens up when a longer lookback shows the
  same level was tested earlier — the spec's "prior S/R" semantics.
* "Bid depth increasing" → :meth:`TickStream.depth_trend` on the
  favoured side over the streak window.
* Anti-news filters → spread > N × EMA, single-tick > 50 bps in the last
  60 s, or volume-spike ratio > 5×.
* "Limit order at mid-price, TIF=300ms IOC" → ``limit_price = mid +
  entry_offset_bps`` (paper exchange fills against the live book; the
  live exchange already uses ``time_in_force = "ioc"`` so unfilled limits
  evaporate within a tick).
* "TP 0.04% / SL 0.025% / 30 s hold" → matches the existing TP/SL
  framework plus a ``max_hold_seconds`` time exit (same pattern as the
  Order Book Sniper).

Funding-rate / news filters from the spec are *not* implemented because
the engine doesn't ingest those feeds today. The strategy compensates
with a wider proxy filter set (spread, volume, single-tick move) so a
flash event still gets vetoed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market
from aera.signals.tick_stream import TickStream

from .base import Leg, Signal, Strategy

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution -> strategies)
    from aera.core import Portfolio


log = get_logger(__name__)


@dataclass
class _SymbolState:
    """Per-symbol working state owned by the strategy."""

    stream: TickStream
    last_signal_mid: float = 0.0
    position_side: Optional[str] = None       # "LONG", "SHORT", or None
    entry_mid: float = 0.0
    entry_size_usd: float = 0.0
    entry_time: float = 0.0


@dataclass
class _FilterResult:
    """Why an otherwise-valid setup was vetoed (or accepted)."""

    accept: bool
    reason: str = ""


class TickReversalScalp(Strategy):
    """Fade an N-tick exhaustion run with shrinking per-tick volume.

    Parameters
    ----------
    min_streak : int
        Minimum same-direction ticks required to fire. Spec default = 5.
    size_decay_threshold : float
        Fraction of size decay across the streak (``1 − last/first``).
        ``0.20`` = the last tick eats at most 80% of what the first tick
        ate.
    sr_band_bps : float
        Current mid must be within ``±sr_band_bps`` of the
        ``sr_lookback_ticks`` extreme in the streak direction (the low
        for a long entry, the high for a short). 0 disables.
    sr_lookback_ticks : int
        How far back to look for the S/R reference. Larger = more
        meaningful "previous level" but slower to warm up. Default 50.
    require_depth_trend : bool
        Require that the favoured side's depth grew across the streak —
        "buyers stepping in" for a long, "sellers stepping in" for a
        short. Default True.
    max_spread_multiple, spread_ema_alpha : float
        Spread guard. Skip when current_spread > multiple × EMA spread.
        0 disables. ``spread_ema_alpha`` is the EMA smoothing factor for
        the baseline.
    volume_spike_multiple, volume_short_window_seconds, volume_long_window_seconds : float
        Volume-spike guard. Skip when the short-window per-second eaten
        rate exceeds the long-window rate by ``multiple×``. 0 disables.
    news_lookback_seconds, news_max_tick_bps : float
        News-spike proxy. Skip when *any* tick in the last
        ``news_lookback_seconds`` moved more than ``news_max_tick_bps``.
        0 disables.
    notional_usd : float
        Reference notional. The executor's ``trade_size_fraction`` will
        typically override this, identical to the other strategies.
    entry_offset_bps : float
        Offset from mid for the entry limit price, in bps. Positive = bid
        below mid / ask above mid (more passive). Spec asks for "at mid",
        i.e. ``0.0``.
    take_profit_pct, stop_loss_pct : float
        Percent-of-entry-mid TP / SL. Spec defaults: +0.04% / −0.025%.
    take_profit_usd, stop_loss_usd : float
        USD-PnL exits. Activated when both are > 0 and ``portfolio`` is
        attached. Same precedence as the other strategies.
    max_hold_seconds : float
        Force a market exit this many seconds after entry. 0 disables.
        Spec default = 30 s.
    rearm_distance_bps : float
        Cheap rearm debouncer.
    min_edge : float
        Floor on the emitted ``Signal.edge`` so sniper / scalper /
        reversal signals sort sensibly.
    portfolio : Portfolio, optional
        Live portfolio used by the USD-PnL exit path.
    clock : callable, optional
        Overridable time source so tests can drive time-based exits
        without sleeping. Defaults to ``time.time``.
    """

    name = "tick_reversal_scalp"

    def __init__(
        self,
        *,
        min_streak: int = 5,
        max_buffer_ticks: int = 200,
        size_decay_threshold: float = 0.20,
        sr_band_bps: float = 5.0,
        sr_lookback_ticks: int = 50,
        require_depth_trend: bool = True,
        max_spread_multiple: float = 3.0,
        spread_ema_alpha: float = 0.05,
        volume_spike_multiple: float = 5.0,
        volume_short_window_seconds: float = 5.0,
        volume_long_window_seconds: float = 60.0,
        news_lookback_seconds: float = 60.0,
        news_max_tick_bps: float = 50.0,
        notional_usd: float = 1000.0,
        entry_offset_bps: float = 0.0,
        take_profit_pct: float = 0.0004,
        stop_loss_pct: float = 0.00025,
        take_profit_usd: float = 0.0,
        stop_loss_usd: float = 0.0,
        max_hold_seconds: float = 30.0,
        rearm_distance_bps: float = 3.0,
        min_edge: float = 0.0004,
        portfolio: Optional["Portfolio"] = None,
        enabled: bool = True,
        clock: Optional[callable] = None,
    ) -> None:
        super().__init__(enabled=enabled)
        self.min_streak = max(2, int(min_streak))
        self.max_buffer_ticks = max(self.min_streak * 4, int(max_buffer_ticks))
        self.size_decay_threshold = min(1.0, max(0.0, float(size_decay_threshold)))
        self.sr_band_bps = max(0.0, float(sr_band_bps))
        self.sr_lookback_ticks = max(0, int(sr_lookback_ticks))
        self.require_depth_trend = bool(require_depth_trend)
        self.max_spread_multiple = max(0.0, float(max_spread_multiple))
        self.spread_ema_alpha = min(1.0, max(0.001, float(spread_ema_alpha)))
        self.volume_spike_multiple = max(0.0, float(volume_spike_multiple))
        self.volume_short_window_seconds = max(0.1, float(volume_short_window_seconds))
        self.volume_long_window_seconds = max(
            self.volume_short_window_seconds * 2.0,
            float(volume_long_window_seconds),
        )
        self.news_lookback_seconds = max(0.0, float(news_lookback_seconds))
        self.news_max_tick_bps = max(0.0, float(news_max_tick_bps))
        self.notional_usd = max(0.0, float(notional_usd))
        self.entry_offset_bps = float(entry_offset_bps)
        self.take_profit_pct = max(0.0, float(take_profit_pct))
        self.stop_loss_pct = max(0.0, float(stop_loss_pct))
        self.take_profit_usd = max(0.0, float(take_profit_usd))
        self.stop_loss_usd = max(0.0, float(stop_loss_usd))
        self.max_hold_seconds = max(0.0, float(max_hold_seconds))
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
            # Always update the tick stream — even when we're already in
            # a position, the EMA / volume / extremes need to keep
            # tracking so the next setup post-exit is fresh.
            st.stream.update(book, now=now)

            # Exit path runs first; a tripped exit always wins on the
            # same tick as a fresh entry attempt would.
            close = self._maybe_emit_close(st, m, outcome, bid_lvl.price, ask_lvl.price, mid, now)
            if close is not None:
                signals.append(close)
                continue

            if st.position_side is not None:
                continue

            # Cheap debounce: don't refire near the last firing mid.
            if st.last_signal_mid > 0:
                move_bps = abs(mid - st.last_signal_mid) / st.last_signal_mid * 1e4
                if move_bps < self.rearm_distance_bps:
                    continue

            direction, length, sizes = st.stream.current_streak()
            if direction == 0 or length < self.min_streak:
                continue

            # Size-decay: enough end-to-end shrinkage across the streak.
            if self.size_decay_threshold > 0:
                decay = TickStream.size_decay(sizes)
                if decay < self.size_decay_threshold:
                    continue
            else:
                decay = 0.0

            # FADE direction: BUY after a downtick streak; SELL after up.
            fade_side = "BUY" if direction < 0 else "SELL"

            # Pre-entry filters: any one failure is a skip.
            filt = self._filters(st, fade_side, mid, now)
            if not filt.accept:
                log.debug(
                    "tick-fade %s veto: %s", m.id, filt.reason,
                )
                continue

            # S/R proxy.
            if self.sr_band_bps > 0 and self.sr_lookback_ticks > 0:
                # For a long fade, the local LOW (direction = −1) should
                # be near mid; for a short fade, the local HIGH (+1).
                extreme = st.stream.recent_extreme(direction, self.sr_lookback_ticks)
                if extreme is None:
                    continue
                band = mid * (self.sr_band_bps / 1e4)
                if abs(mid - extreme) > band:
                    continue

            # Depth trend on the favoured side.
            if self.require_depth_trend:
                side_key = "bid" if fade_side == "BUY" else "ask"
                trend = st.stream.depth_trend(side_key, lookback=length)
                if trend <= 0:
                    continue

            # Compute entry limit. Spec: "at mid"; offset_bps lets the
            # operator bias passive or aggressive.
            offset = mid * (self.entry_offset_bps / 1e4)
            limit_price = mid - offset if fade_side == "BUY" else mid + offset
            # Don't let the offset cross the spread the wrong way.
            if fade_side == "BUY" and limit_price > ask_lvl.price:
                limit_price = ask_lvl.price
            if fade_side == "SELL" and limit_price < bid_lvl.price:
                limit_price = bid_lvl.price

            try:
                leverage = float(m.metadata.get("leverage", 1.0) or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0

            # Edge — proportional to how exhausted the run looks. Longer
            # streak + bigger decay = stronger signal. Capped at 1% so it
            # sorts above the sniper but doesn't dominate the queue.
            strength = min(1.0, (length / max(self.min_streak, 1)) * 0.5
                           + decay * 0.5)
            edge = max(self.min_edge, strength * 0.01)

            reason = (
                f"streak={direction:+d}x{length} decay={decay:.2f} "
                f"first={sizes[0]:g} last={sizes[-1]:g} "
                f"spread_x{st.stream.current_spread_multiple() or 1.0:.2f}"
            )
            leg = Leg(
                market_id=m.id,
                outcome_id=outcome.id,
                side=fade_side,
                limit_price=float(limit_price),
                size_usd=self.notional_usd,
                reason=reason,
                leverage=leverage,
            )
            signals.append(
                Signal(
                    strategy=self.name,
                    confidence=strength,
                    edge=edge,
                    legs=[leg],
                    metadata={
                        "symbol": m.id,
                        "streak_direction": direction,
                        "streak_length": length,
                        "size_decay": decay,
                        "first_tick_size": sizes[0],
                        "last_tick_size": sizes[-1],
                        "mid": float(mid),
                    },
                )
            )
            st.last_signal_mid = mid
            st.position_side = "LONG" if fade_side == "BUY" else "SHORT"
            st.entry_mid = mid
            st.entry_size_usd = self.notional_usd
            st.entry_time = now
            log.debug(
                "tick-fade FIRE %s %s mid=%.4f streak=%+dx%d decay=%.2f",
                m.id, fade_side, mid, direction, length, decay,
            )
        return signals

    # ------------------------------------------------------------------
    # state plumbing
    # ------------------------------------------------------------------

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState(
                stream=TickStream(
                    max_ticks=self.max_buffer_ticks,
                    spread_ema_alpha=self.spread_ema_alpha,
                    volume_short_window_seconds=self.volume_short_window_seconds,
                    volume_long_window_seconds=self.volume_long_window_seconds,
                ),
            )
            self._state[symbol] = st
        return st

    # ------------------------------------------------------------------
    # pre-entry filters
    # ------------------------------------------------------------------

    def _filters(
        self,
        st: _SymbolState,
        fade_side: str,
        mid: float,
        now: float,
    ) -> _FilterResult:
        """Return ``accept=False`` for any tripped safety filter."""
        # Spread guard
        if self.max_spread_multiple > 0:
            spread_mult = st.stream.current_spread_multiple()
            if spread_mult is not None and spread_mult > self.max_spread_multiple:
                return _FilterResult(False, f"spread {spread_mult:.2f}× > {self.max_spread_multiple:g}×")

        # News-spike proxy
        if self.news_max_tick_bps > 0 and self.news_lookback_seconds > 0:
            move = st.stream.max_tick_move_bps(self.news_lookback_seconds, now=now)
            if move > self.news_max_tick_bps:
                return _FilterResult(
                    False,
                    f"news-spike {move:.1f}bps > {self.news_max_tick_bps:g}bps "
                    f"in last {self.news_lookback_seconds:g}s",
                )

        # Volume spike
        if self.volume_spike_multiple > 0:
            ratio = st.stream.volume_spike_ratio(now=now)
            if ratio is not None and ratio > self.volume_spike_multiple:
                return _FilterResult(
                    False,
                    f"volume spike {ratio:.2f}× > {self.volume_spike_multiple:g}×",
                )

        return _FilterResult(True)

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
        """Emit a reduce-only close on hold-timeout / SL / TP.

        Priority (highest first):
            1. Hold-timeout — ``now − entry_time > max_hold_seconds``.
               Spec: "close after 30s regardless of P&L".
            2. Stop-loss (USD if active + portfolio present, else %).
            3. Take-profit (USD if active + portfolio present, else %).
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

        # --- 1. hold-timeout ----------------------------------------
        if self.max_hold_seconds > 0 and (now - st.entry_time) > self.max_hold_seconds:
            return self._build_close(
                st, market, outcome, side, close_side, limit_price, leverage,
                kind="hold-timeout",
                reason=f"held {now - st.entry_time:.2f}s > {self.max_hold_seconds:.2f}s",
                extra_metadata={"hold_seconds": now - st.entry_time},
            )

        # --- 2 + 3: SL / TP -----------------------------------------
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

        if not (hit_sl or hit_tp):
            return None

        kind = "stop-loss" if hit_sl else "take-profit"
        move_bps = (mid - st.entry_mid) / st.entry_mid * 1e4
        pnl_label = f" pnl=${pnl_usd:+.2f}" if pnl_usd is not None else ""
        return self._build_close(
            st, market, outcome, side, close_side, limit_price, leverage,
            kind=kind,
            reason=f"entry={st.entry_mid:.4f} mid={mid:.4f} ({move_bps:+.1f}bps){pnl_label}",
            extra_metadata=({"pnl_usd": float(pnl_usd)} if pnl_usd is not None else {}),
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
            "tick-fade EXIT %s %s side=%s entry=%.4f close=%.4f (%s)",
            market.id, kind.upper(), position_side, entry_mid, limit_price, reason,
        )
        return sig

    @staticmethod
    def _reset_position(st: _SymbolState) -> None:
        st.position_side = None
        st.entry_mid = 0.0
        st.entry_size_usd = 0.0
        st.entry_time = 0.0
