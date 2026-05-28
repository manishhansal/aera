"""Adaptive Brain — live edge measurement + regime-aware signal routing.

The :class:`AdaptiveBrain` is an overlay that sits between the strategies
and the executor (after the greedy overlay, before the risk vet). It does
five things the rest of the stack does not:

1. **Measures live per-strategy edge.** Every closed round-trip is fed
   in via :meth:`on_trade_closed`; the brain keeps the last N PnLs per
   strategy and exposes rolling win-rate, expectancy, average win/loss,
   profit factor, and a Sharpe-like score.

2. **Auto-mutes losing strategies.** Once a strategy has at least
   ``min_trades_for_eval`` closed trades, if its rolling win-rate or
   expectancy is below the configured floor the brain puts it in
   *cooldown* — no new entries from that strategy fire for
   ``mute_seconds``. After cooldown the strategy probates back in at a
   reduced size multiplier until it proves itself again.

3. **Regime-routes signals.** A :class:`~aera.signals.regime.RegimeBook`
   classifies each symbol as RANGE / TREND_UP / TREND_DOWN / HIGH_VOL /
   NEWS_SPIKE. Each strategy declares which regimes it likes; the brain
   vetoes signals that arrive in the wrong regime (e.g. mean-reversion
   in a strong trend = death). News spikes veto everything.

4. **Enforces a daily loss circuit-breaker and correlation cap.** If the
   rolling 24h realised PnL drops below ``-daily_loss_pct × bankroll``,
   no new entries fire for the rest of the day. Likewise, the total
   gross long (or short) exposure across all symbols is capped so the
   bot can't accidentally end up 100% long crypto on a single bad day.

5. **Dynamic sizing multiplier per strategy.** Wraps the executor's
   ``trade_size_fraction`` / greedy ``compound_fraction`` so each
   strategy gets sized based on its *measured* edge. Brand-new
   strategies start at ``probation_size_mult`` (typically 0.5) and grow
   to 1.0 once they've shown a positive expectancy on a small sample.

The brain is **non-destructive**: it only ever drops or shrinks signals.
It never opens a trade the strategy didn't suggest, never overrides
TP/SL (greedy still owns those), and never touches reduce-only legs
(closes always go through so the bot can flatten regardless of state).

When ``cfg.enabled == False`` the brain becomes a transparent passthrough
— every signal flows through unchanged and stats are still tracked for
the dashboard.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from aera.logging import get_logger
from aera.markets import Market
from aera.signals.regime import Regime, RegimeBook, RegimeSnapshot
from aera.strategies import Leg, Signal

from .portfolio import Portfolio


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy-to-regime preferences
# ---------------------------------------------------------------------------
#
# Each entry maps a strategy name to the set of regimes its signals
# should be permitted in. The brain vetoes signals whose symbol is in
# any other regime. The defaults below encode the structural thesis of
# each strategy:
#
# * Mean-reversion / VWAP / sweep / tick-reversal strategies fade
#   extension — they only work in RANGE or against a *weak* trend.
# * Flow scalp rides momentum — it only works in trending regimes.
# * Order book sniper and bid-ask spread fade are microstructure-only
#   and work in RANGE; they get steamrolled in trends.
#
# NEWS_SPIKE and HIGH_VOL are NEVER in any preference set — a news
# spike vetoes everything, and high vol shrinks aggressively (see
# the size multiplier below). UNKNOWN is allowed for everyone since
# it's just "not enough samples yet".
STRATEGY_REGIME_PREFS: Dict[str, Set[Regime]] = {
    "delta_perp_scalper": {Regime.RANGE, Regime.UNKNOWN},
    "order_book_sniper": {Regime.RANGE, Regime.UNKNOWN},
    "tick_reversal_scalp": {Regime.RANGE, Regime.UNKNOWN},
    "bid_ask_spread_fade": {Regime.RANGE, Regime.UNKNOWN},
    "micro_vwap_sniper": {Regime.RANGE, Regime.UNKNOWN},
    "stop_hunt_reversal": {Regime.RANGE, Regime.UNKNOWN},
    "flow_scalp": {Regime.TREND_UP, Regime.TREND_DOWN, Regime.UNKNOWN},
}


# Strategies that get a vol-regime shrink even in their preferred regime.
# All of them — scalpers across the board lose money on wide spreads.
VOL_SENSITIVE_STRATEGIES: Set[str] = set(STRATEGY_REGIME_PREFS.keys())


@dataclass
class _StratPerf:
    """Rolling performance record for one strategy."""

    name: str
    pnls: Deque[float] = field(default_factory=deque)
    total_trades: int = 0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    # Soft mute: when ``muted_until > now`` the brain drops every entry
    # signal from this strategy. Refreshes only when the gate trips
    # again (so an already-muted strategy that keeps failing gets a
    # fresh full cooldown).
    muted_until: float = 0.0
    mute_count: int = 0
    # Size multiplier applied to every entry signal. 1.0 = no change,
    # 0.5 = half size, 0.0 = effectively muted (we still set
    # ``muted_until`` so the dashboard shows mute, not 0-sized fire).
    size_mult: float = 1.0
    # Strategy is on probation: just came back from mute, runs at
    # ``probation_size_mult`` until it accumulates ``probation_trades``
    # of new closed trades. Then graduates back to 1.0.
    probation: bool = False
    probation_trades_left: int = 0

    @property
    def n(self) -> int:
        return len(self.pnls)

    @property
    def wins(self) -> int:
        return sum(1 for p in self.pnls if p > 0)

    @property
    def losses(self) -> int:
        return sum(1 for p in self.pnls if p < 0)

    @property
    def win_rate(self) -> float:
        n = self.n
        return (self.wins / n) if n else 0.0

    @property
    def expectancy(self) -> float:
        n = self.n
        return (sum(self.pnls) / n) if n else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(p for p in self.pnls if p > 0)
        gross_loss = abs(sum(p for p in self.pnls if p < 0))
        if gross_loss <= 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    def to_dict(self) -> dict:
        pf = self.profit_factor
        return {
            "name": self.name,
            "sample_size": self.n,
            "total_trades": self.total_trades,
            "total_pnl": float(self.total_pnl),
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": float(self.win_rate),
            "expectancy": float(self.expectancy),
            "profit_factor": (None if pf == float("inf") else float(pf)),
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "muted_until": float(self.muted_until),
            "mute_count": self.mute_count,
            "size_mult": float(self.size_mult),
            "probation": self.probation,
            "probation_trades_left": self.probation_trades_left,
        }


@dataclass
class BrainStats:
    """Aggregated brain counters surfaced to the dashboard."""

    signals_seen: int = 0
    signals_passed: int = 0
    signals_vetoed_regime: int = 0
    signals_vetoed_mute: int = 0
    signals_vetoed_daily_loss: int = 0
    signals_vetoed_correlation: int = 0
    signals_shrunk: int = 0
    daily_pnl: float = 0.0
    daily_pnl_floor: float = 0.0  # the trip level (computed dynamically)
    daily_window_started_at: float = 0.0
    halted_until: float = 0.0     # if > now, everything's vetoed
    last_event_at: float = 0.0


class AdaptiveBrain:
    """Live edge tracking + regime routing + adaptive sizing overlay.

    Wire it into the engine via the ``brain=`` constructor kwarg. The
    engine will:

    1. Call :meth:`observe_markets` once per scan tick to keep the
       per-symbol regime detectors warm.
    2. Pipe every collected ``Signal`` through :meth:`filter_signals`
       *after* greedy's flatten signals and the strategy scans, but
       *before* the executor runs.
    3. Call :meth:`on_execution` on every ``ExecutionResult`` so the
       brain can absorb closed-trade PnLs and update its trackers.

    Construction
    ------------
    The only required arg is the live :class:`Portfolio` (for the daily
    loss math and exposure cap). Everything else is config: see
    :class:`~aera.settings.BrainConfig` for the full set of tunables.

    Defaults are deliberately conservative — the brain shrinks size
    and mutes strategies easily; it's much harder to revive a muted
    strategy. The asymmetry is intentional: most "make my bot
    profitable" outcomes come from refusing to trade when the edge
    isn't there, not from finding new edges.
    """

    def __init__(
        self,
        cfg,
        portfolio: Portfolio,
        *,
        clock=None,
        regime_book: Optional[RegimeBook] = None,
    ) -> None:
        self.cfg = cfg
        self.portfolio = portfolio
        self._clock = clock or time.time
        self.regimes: RegimeBook = regime_book or RegimeBook(
            short_window=int(getattr(cfg, "regime_short_window", 30)),
            long_window=int(getattr(cfg, "regime_long_window", 300)),
            trend_threshold=float(getattr(cfg, "regime_trend_threshold", 0.30)),
            high_vol_ratio=float(getattr(cfg, "regime_high_vol_ratio", 2.0)),
            news_tick_bps=float(getattr(cfg, "regime_news_tick_bps", 25.0)),
        )
        self._perf: Dict[str, _StratPerf] = {}
        # Rolling 24h PnL: (timestamp, pnl) pairs, evicted on observe.
        self._daily_pnls: Deque[Tuple[float, float]] = deque()
        self._daily_window_seconds = float(
            getattr(cfg, "daily_window_seconds", 24 * 3600)
        )
        # Bankroll snapshot used as the denominator for the daily loss
        # check. Updated on every successful execution so growth from
        # compounding raises the floor proportionally.
        self._bankroll_anchor: float = max(0.0, float(portfolio.bankroll))
        self.stats = BrainStats()
        self.stats.daily_window_started_at = self._clock()

    # ------------------------------------------------------------------
    # public state
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", True))

    def perf(self, strategy: str) -> _StratPerf:
        p = self._perf.get(strategy)
        if p is None:
            p = _StratPerf(name=strategy)
            self._perf[strategy] = p
        return p

    def performance(self) -> Dict[str, dict]:
        return {name: p.to_dict() for name, p in self._perf.items()}

    def regimes_snapshot(self) -> Dict[str, dict]:
        return {sym: snap.to_dict() for sym, snap in self.regimes.snapshots().items()}

    # ------------------------------------------------------------------
    # market observation
    # ------------------------------------------------------------------

    def observe_markets(self, markets: Dict[str, Market]) -> None:
        """Refresh the per-symbol regime detectors."""
        if not self.enabled:
            return
        self.regimes.observe_markets(markets)

    # ------------------------------------------------------------------
    # signal filter
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals: List[Signal],
        markets: Dict[str, Market],
        *,
        now: Optional[float] = None,
    ) -> List[Signal]:
        """Apply every brain veto / size adjustment to ``signals``.

        Returns a NEW list with:

        * reduce-only signals always passed through (closes must flow),
        * fresh-entry signals vetoed if the brain disapproves,
        * surviving fresh-entry signals' ``size_usd`` scaled by the
          strategy's current ``size_mult`` (capped against the daily-
          loss / correlation guard).

        ``now`` is injected for tests; production callers should leave
        it as ``None`` to use the configured clock.
        """
        if not self.enabled:
            return signals
        now = float(now if now is not None else self._clock())
        self.stats.signals_seen += len(signals)
        self.stats.last_event_at = now

        self._evict_daily(now)

        # Daily-loss circuit breaker
        if self._daily_loss_tripped():
            # Still allow reduce-only legs through so positions can close.
            out = [s for s in signals if any(getattr(l, "reduce_only", False) for l in s.legs)]
            vetoed = len(signals) - len(out)
            self.stats.signals_vetoed_daily_loss += vetoed
            if vetoed:
                log.warning(
                    "brain: daily loss cap tripped (pnl=$%+.4f floor=$%.4f) "
                    "— vetoed %d entry signals",
                    self.stats.daily_pnl, self.stats.daily_pnl_floor, vetoed,
                )
            return out

        # Per-strategy mute auto-expire
        for p in self._perf.values():
            if p.muted_until and now >= p.muted_until:
                self._end_mute(p)

        kept: List[Signal] = []
        # Track planned new exposure across this batch so the
        # correlation cap can refuse later signals once the limit is
        # reached on a single tick.
        planned_long = 0.0
        planned_short = 0.0
        existing_long, existing_short = self._existing_exposure()

        for sig in signals:
            reduce_only = all(getattr(l, "reduce_only", False) for l in sig.legs)
            if reduce_only:
                kept.append(sig)
                continue

            # Strategy mute
            perf = self.perf(sig.strategy)
            if perf.muted_until and now < perf.muted_until:
                self.stats.signals_vetoed_mute += 1
                self._tag_vetoed(sig, "brain mute")
                continue

            # Regime veto (per leg's market). One bad market = drop the
            # whole signal. NEWS_SPIKE is always a hard veto; wrong-
            # regime "soft" vetoes shrink size instead (see
            # _size_multiplier) when ``regime_soft_veto`` is on.
            allow, regime_reason = self._regime_allows(sig)
            if not allow:
                self.stats.signals_vetoed_regime += 1
                self._tag_vetoed(sig, regime_reason)
                continue

            # Compute the multiplier we'll apply (strategy mult × regime
            # mult). Multipliers <= 0 turn into vetoes; > 0 reshape size.
            mult = self._size_multiplier(sig, markets, perf)
            if mult <= 0:
                self.stats.signals_vetoed_regime += 1
                self._tag_vetoed(sig, "size multiplier collapsed to 0")
                continue

            # Correlation / gross-exposure check on this batch. The cap
            # scales with leverage (= buying power = bankroll × lev) so
            # leveraged perps can carry a sensible number of concurrent
            # scalps rather than being blocked on the first $X trade.
            sig_notional = sum(l.size_usd for l in sig.legs) * mult
            side = "BUY" if any(l.side == "BUY" for l in sig.legs) else "SELL"
            sig_leverage = max(
                (float(getattr(l, "leverage", 1.0) or 1.0) for l in sig.legs),
                default=1.0,
            )
            gross_cap = self._gross_exposure_cap(sig_leverage)
            projected_long = existing_long + planned_long + (
                sig_notional if side == "BUY" else 0.0
            )
            projected_short = existing_short + planned_short + (
                sig_notional if side == "SELL" else 0.0
            )
            if gross_cap > 0 and max(projected_long, projected_short) > gross_cap:
                self.stats.signals_vetoed_correlation += 1
                reason = (
                    f"correlation cap: projected gross {side}=$"
                    f"{max(projected_long, projected_short):.2f} > "
                    f"cap ${gross_cap:.2f}"
                )
                self._tag_vetoed(sig, reason)
                log.info(
                    "brain: correlation cap hit, vetoing %s on %s "
                    "(projected gross %s=$%.2f > cap $%.2f)",
                    sig.strategy, sig.metadata.get("symbol", "?"), side,
                    max(projected_long, projected_short), gross_cap,
                )
                continue

            if side == "BUY":
                planned_long += sig_notional
            else:
                planned_short += sig_notional

            if mult != 1.0:
                sig = self._rescale(sig, mult)
                self.stats.signals_shrunk += 1
            kept.append(sig)
            self.stats.signals_passed += 1

        return kept

    @staticmethod
    def _tag_vetoed(sig: Signal, reason: str) -> None:
        """Stamp the brain's veto reason on a signal so the dashboard
        can surface it. We use the signal's metadata dict — strategies
        and listeners already read from there."""
        try:
            sig.metadata["brain_vetoed"] = True
            sig.metadata["brain_veto_reason"] = reason
        except Exception:
            # Metadata might be frozen on some custom Signal subclass;
            # ignore — the tally counters still tick.
            pass

    # ------------------------------------------------------------------
    # execution-result subscriber
    # ------------------------------------------------------------------

    def on_execution(self, result) -> None:
        """Update tracker state from a fresh ``ExecutionResult``."""
        if not self.enabled:
            return
        if not getattr(result, "success", False):
            return
        sig = getattr(result, "signal", None)
        strategy = getattr(sig, "strategy", "") if sig is not None else ""
        if not strategy:
            return
        # Refresh the bankroll anchor for the daily-loss base.
        self._bankroll_anchor = max(
            self._bankroll_anchor, float(self.portfolio.bankroll)
        )

    def on_trade_closed(
        self, strategy: str, pnl_usd: float, *, now: Optional[float] = None
    ) -> None:
        """Absorb a round-trip's realised PnL into the trackers.

        Called by the engine when a round-trip closes (the dashboard's
        round-trip tracker is the most reliable source; the engine
        forwards ``TradeEvent.pnl`` here).
        """
        if not self.enabled:
            return
        now = float(now if now is not None else self._clock())
        perf = self.perf(strategy)
        max_window = int(getattr(self.cfg, "perf_window", 30))
        perf.pnls.append(float(pnl_usd))
        while len(perf.pnls) > max_window:
            perf.pnls.popleft()
        perf.total_trades += 1
        perf.total_pnl += float(pnl_usd)
        if pnl_usd > 0:
            perf.consecutive_wins += 1
            perf.consecutive_losses = 0
        elif pnl_usd < 0:
            perf.consecutive_losses += 1
            perf.consecutive_wins = 0

        # Daily PnL bookkeeping
        self._daily_pnls.append((now, float(pnl_usd)))
        self._recompute_daily_pnl()

        # Probation progress / graduation
        if perf.probation:
            perf.probation_trades_left = max(0, perf.probation_trades_left - 1)
            if perf.probation_trades_left == 0 and perf.expectancy >= 0:
                perf.probation = False
                perf.size_mult = 1.0
                log.info(
                    "brain: strategy %s graduated from probation "
                    "(expectancy=$%+.4f over %d trades)",
                    strategy, perf.expectancy, perf.n,
                )

        # Performance gate
        self._maybe_mute(perf, now)

    # ------------------------------------------------------------------
    # internals: gating + sizing
    # ------------------------------------------------------------------

    def _regime_allows(self, sig: Signal) -> Tuple[bool, str]:
        """Return ``(allow, reason)``. The reason populates the brain-
        veto tag on the dashboard so the user can see WHY a signal
        was dropped.

        Regime priority:

        * NEWS_SPIKE is ALWAYS a hard veto — we never trade through
          a 25 bp single-tick jump (configurable via
          ``regime_news_tick_bps``).
        * HIGH_VOL is NOT a hard veto; ``_size_multiplier`` shrinks
          the entry by ``high_vol_size_mult`` instead.
        * Wrong-regime (e.g. mean-reversion in TREND_UP) is a SOFT
          veto by default (``regime_soft_veto`` = True): the entry
          fires at ``wrong_regime_size_mult`` of normal size instead
          of being dropped. The brain's perf gate then auto-mutes the
          strategy if it actually loses money — letting empirical
          evidence drive the kill decision rather than a prior.
          Set ``regime_soft_veto = False`` in config to revert to
          hard veto behaviour.
        """
        prefs = STRATEGY_REGIME_PREFS.get(sig.strategy)
        if prefs is None:
            return True, ""
        soft = bool(getattr(self.cfg, "regime_soft_veto", True))
        for leg in sig.legs:
            snap = self.regimes.snapshot(leg.market_id)
            if snap.regime == Regime.NEWS_SPIKE:
                return False, f"news spike on {leg.market_id} (last_tick={snap.last_tick_bps:.1f}bps)"
            if snap.regime == Regime.HIGH_VOL:
                continue  # shrinks via size_multiplier, doesn't veto
            if snap.regime not in prefs and not soft:
                return False, f"wrong regime {snap.regime.value} on {leg.market_id}"
        return True, ""

    def _size_multiplier(
        self, sig: Signal, markets: Dict[str, Market], perf: _StratPerf
    ) -> float:
        """Combined per-strategy + regime multiplier in [0, 1]."""
        mult = float(perf.size_mult)
        prefs = STRATEGY_REGIME_PREFS.get(sig.strategy)
        soft = bool(getattr(self.cfg, "regime_soft_veto", True))
        wrong_mult = float(getattr(self.cfg, "wrong_regime_size_mult", 0.5))
        hv_mult = float(getattr(self.cfg, "high_vol_size_mult", 0.5))
        # Shrink in high-vol regimes for vol-sensitive strategies AND
        # in wrong-regime when soft-veto is active. Both can stack.
        for leg in sig.legs:
            snap = self.regimes.snapshot(leg.market_id)
            if (
                snap.regime == Regime.HIGH_VOL
                and sig.strategy in VOL_SENSITIVE_STRATEGIES
            ):
                mult *= hv_mult
            if (
                prefs is not None
                and snap.regime not in prefs
                and snap.regime not in (Regime.HIGH_VOL, Regime.NEWS_SPIKE)
                and soft
            ):
                mult *= wrong_mult
        # Streak penalty: shrink as consecutive losses build.
        if perf.consecutive_losses >= 2:
            mult *= 1.0 / (1.0 + 0.5 * (perf.consecutive_losses - 1))
        return max(0.0, min(1.0, mult))

    def _rescale(self, sig: Signal, mult: float) -> Signal:
        new_legs: List[Leg] = []
        for l in sig.legs:
            if getattr(l, "reduce_only", False):
                new_legs.append(l)
                continue
            new_legs.append(Leg(
                market_id=l.market_id, outcome_id=l.outcome_id, side=l.side,
                limit_price=l.limit_price, size_usd=l.size_usd * mult,
                reason=l.reason + f" (brain x{mult:.3f})",
                leverage=getattr(l, "leverage", 1.0),
                reduce_only=False,
                time_in_force=getattr(l, "time_in_force", None),
                post_only=getattr(l, "post_only", None),
            ))
        return Signal(
            strategy=sig.strategy,
            confidence=sig.confidence,
            edge=sig.edge,
            legs=new_legs,
            metadata={**sig.metadata, "brain_size_mult": float(mult)},
        )

    def _maybe_mute(self, perf: _StratPerf, now: float) -> None:
        """Trip the cooldown if rolling perf is below the configured floor."""
        min_n = int(getattr(self.cfg, "min_trades_for_eval", 10))
        if perf.n < min_n:
            return

        min_wr = float(getattr(self.cfg, "min_win_rate", 0.40))
        min_exp = float(getattr(self.cfg, "min_expectancy_usd", 0.0))
        bad_wr = perf.win_rate < min_wr
        bad_exp = perf.expectancy < min_exp
        streak_cap = int(getattr(self.cfg, "max_strategy_loss_streak", 4))
        bad_streak = perf.consecutive_losses >= streak_cap

        if not (bad_wr or bad_exp or bad_streak):
            # Strategy is healthy; if it was on probation and earning,
            # graduation is handled in on_trade_closed.
            return

        mute_seconds = float(getattr(self.cfg, "mute_seconds", 600.0))
        prev = perf.muted_until
        perf.muted_until = now + mute_seconds
        perf.mute_count += 1
        perf.probation = True
        perf.probation_trades_left = int(
            getattr(self.cfg, "probation_trades", 5)
        )
        perf.size_mult = float(getattr(self.cfg, "probation_size_mult", 0.5))
        # Reset the rolling window so the strategy gets a clean re-eval
        # after cooldown; the lifetime counters persist for the dashboard.
        perf.pnls.clear()
        perf.consecutive_losses = 0
        perf.consecutive_wins = 0

        log.warning(
            "brain MUTE %s: wr=%.0f%% exp=$%+.4f streak=%d "
            "(min_wr=%.0f%% min_exp=$%.2f streak_cap=%d) → muted %.0fs, "
            "returns on probation at %.2fx",
            perf.name, perf.win_rate * 100, perf.expectancy,
            perf.consecutive_losses, min_wr * 100, min_exp, streak_cap,
            mute_seconds, perf.size_mult,
        )

    def _end_mute(self, perf: _StratPerf) -> None:
        # Mute window expired — strategy can fire again at the probation
        # size mult. Graduation back to 1.0 happens in on_trade_closed
        # once probation_trades have all been positive on average.
        if perf.muted_until == 0:
            return
        log.info(
            "brain: strategy %s mute expired, fires resumed at %.2fx "
            "size on probation",
            perf.name, perf.size_mult,
        )
        perf.muted_until = 0

    # ------------------------------------------------------------------
    # daily loss / correlation helpers
    # ------------------------------------------------------------------

    def _evict_daily(self, now: float) -> None:
        cutoff = now - self._daily_window_seconds
        evicted = False
        while self._daily_pnls and self._daily_pnls[0][0] < cutoff:
            self._daily_pnls.popleft()
            evicted = True
        if evicted:
            # Cached daily PnL is stale; refresh so the circuit breaker
            # and the dashboard reflect the post-evict total.
            self._recompute_daily_pnl()
        if not self._daily_pnls:
            self.stats.daily_window_started_at = now

    def _recompute_daily_pnl(self) -> None:
        self.stats.daily_pnl = float(sum(p for _, p in self._daily_pnls))

    def _daily_loss_tripped(self) -> bool:
        pct = float(getattr(self.cfg, "daily_loss_pct", 0.10))
        if pct <= 0:
            return False
        denom = max(self._bankroll_anchor, float(self.portfolio.starting_bankroll), 1.0)
        floor = -pct * denom
        self.stats.daily_pnl_floor = float(floor)
        return self.stats.daily_pnl <= floor

    def _existing_exposure(self) -> Tuple[float, float]:
        """Sum of |notional| of all open positions, split LONG / SHORT."""
        longs = 0.0
        shorts = 0.0
        for pos in self.portfolio.positions.values():
            if pos.shares == 0:
                continue
            notional = pos.notional_exposure(pos.avg_cost)
            if pos.shares > 0:
                longs += notional
            else:
                shorts += notional
        return longs, shorts

    def _gross_exposure_cap(self, leverage: float = 1.0) -> float:
        """Per-side gross-notional cap = ``mult × bankroll × leverage``.

        Scales with leverage so a $100 bankroll on 25× perps gets a
        cap of ``mult × $2500 = $5000`` at the default 2.0× — i.e. up
        to roughly 13 concurrent $375 scalps before hitting the
        ceiling. Without the leverage factor a $100 bankroll caps at
        $200 of notional which blocks every fire on its first
        attempt (the bug that produced the all-vetoes log).
        """
        cap_mult = float(getattr(self.cfg, "max_gross_exposure_mult", 0.0))
        if cap_mult <= 0:
            return 0.0
        lev = max(1.0, float(leverage))
        # Use settled wealth as the cash base so the cap compounds
        # with realised growth.
        return cap_mult * max(0.0, float(self.portfolio.settled_wealth)) * lev
