"""Greedy autopilot — dynamic TP/SL + leverage selection + fast compounding.

The :class:`GreedyTradeManager` is an *overlay* that sits between the
strategies and the executor. It takes ownership of three decisions on
every trade:

1. **Per-trade take-profit** (USD): computed at fill time as
   ``round_trip_fees × fee_pad_multiple + min_profit_usd``. Defaults
   yield ``$0.50 + $1.00 = $1.50`` of profit on a $500 notional at 5 bps
   taker fee. The TP rolls forward on extension — every locked dollar
   stretches the next target by ``extend_tp_step_usd``.

2. **Trailing stop-loss** (USD): starts at ``-initial_sl_usd``. Once
   unrealised PnL crosses ``lock_in_trigger_ratio × tp_target`` the SL
   ratchets up to ``running_best_pnl - trailing_giveback_usd`` so the
   worst-case give-back from the peak is small. SL never moves down.

3. **Leverage**: picked from the live consecutive-win streak. Starts at
   ``min_leverage`` and steps up by ``leverage_step`` per win, capped at
   ``max_leverage`` (and the venue cap if ``respect_venue_cap``). Loss
   streaks divide the chosen leverage by ``(1 + consecutive_losses)``
   so a bad run de-risks fast.

Compounding is implicit: ``compound_fraction`` replaces the risk
module's ``trade_size_fraction`` when greedy is enabled, so almost the
entire live bankroll is deployed on every fresh entry. Realised wins
flow back into the bankroll the instant they hit, and the next trade
sizes against the new wealth.

The manager subscribes to the engine's ``on_execution`` listener to
track entries / closes, and exposes :meth:`proposed_closes` for the
engine to call at the top of each scan tick.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aera.logging import get_logger
from aera.markets import Market
from aera.settings import GreedyConfig
from aera.strategies import Leg, Signal

from .portfolio import Fill, Portfolio


log = get_logger(__name__)


# Public strategy name the greedy manager stamps on emitted close signals.
GREEDY_STRATEGY_NAME = "greedy"


@dataclass
class _GreedyPosition:
    """Per-(market, outcome) state the manager keeps for live positions."""

    market_id: str
    outcome_id: str
    side: str                       # "LONG" or "SHORT"
    entry_price: float
    entry_shares: float             # absolute, always >= 0
    entry_notional: float
    leverage: float
    entry_time: float
    # Round-trip fee estimate at entry time (USD). Used to compute the
    # initial TP target and refreshed never — the target is fixed at
    # fill time so a moving fee model does not destabilise the exit.
    fees_round_trip_usd: float
    # Current TP target in USD-PnL space. Rolls forward each time it is
    # crossed by ``extend_tp_step_usd`` (greedy continuation).
    tp_target_usd: float
    # Current SL in USD-PnL space. Starts at ``-initial_sl_usd`` and
    # only ever moves UP (toward profit) — never down.
    sl_level_usd: float
    # Running best PnL seen since entry. Drives the trailing ratchet.
    best_pnl_usd: float = 0.0
    # Strategy that emitted the original entry, for logging.
    source_strategy: str = ""
    # When > 0 the manager is in BPS-OF-NOTIONAL mode: thresholds are
    # derived dynamically from ``entry_notional × bps × 1e-4`` so the
    # trade auto-scales with whatever the compounder actually opened.
    # When 0 the position falls back to the legacy absolute-USD
    # thresholds (``min_profit_usd`` / ``initial_sl_usd`` /
    # ``extend_tp_step_usd``).
    extend_step_usd: float = 0.0
    trailing_giveback_usd: float = 0.0


@dataclass
class GreedyStats:
    """Lightweight counters the dashboard can read for greedy state."""

    consecutive_wins: int = 0
    consecutive_losses: int = 0
    chosen_leverage: float = 0.0
    open_positions: int = 0
    tp_hits: int = 0
    sl_hits: int = 0
    timeout_hits: int = 0


class GreedyTradeManager:
    """Dynamic TP/SL + leverage selection + fast compounding overlay.

    Attach to:
      * an :class:`~aera.execution.Executor` (leverage override +
        compound sizing), via the manager being passed in at construction;
      * a :class:`~aera.core.DeltaEngine` (close-signal overlay +
        execution-result subscriber).

    The manager is fully no-op when ``cfg.enabled == False``; callers
    can safely keep a reference and check ``manager.enabled`` per tick.
    """

    def __init__(
        self,
        cfg: GreedyConfig,
        portfolio: Portfolio,
        *,
        taker_fee_bps: float = 0.0,
        clock=None,
    ) -> None:
        self.cfg = cfg
        self.portfolio = portfolio
        # Allow a per-manager fee override (e.g. the live venue reports
        # a different fee than the paper config). When 0 we fall back to
        # the executor's configured taker fee.
        self.taker_fee_bps = float(
            cfg.fee_override_bps if cfg.fee_override_bps > 0 else taker_fee_bps
        )
        self._clock = clock or time.time
        self._positions: Dict[str, _GreedyPosition] = {}
        # Streak tracking. We mirror the portfolio's consecutive_losses
        # (it is the source of truth) and maintain our own win streak.
        self._wins = 0
        self._losses = 0
        self.stats = GreedyStats()

    # ------------------------------------------------------------------
    # public state
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    @property
    def positions(self) -> Dict[str, _GreedyPosition]:
        return self._positions

    # ------------------------------------------------------------------
    # leverage decision
    # ------------------------------------------------------------------

    def decide_leverage(self, market: Optional[Market]) -> float:
        """Pick a leverage for a fresh entry on ``market``.

        Algorithm (greedy):
            target = min_leverage + wins × leverage_step
            target /= (1 + losses)
            target = clamp(target, min_leverage, cap)

        where ``cap = min(max_leverage, venue_cap)`` when
        ``respect_venue_cap`` is on, else ``max_leverage``.

        When greedy is disabled the manager returns the venue leverage
        unchanged so callers can use a single code path.
        """
        venue_lev = self._venue_leverage(market)
        if not self.enabled:
            return venue_lev
        cap = self.cfg.max_leverage
        if self.cfg.respect_venue_cap:
            cap = min(cap, venue_lev) if venue_lev > 0 else cap
        cap = max(self.cfg.min_leverage, cap)

        # Step up with wins, scale down with losses.
        target = self.cfg.min_leverage + self._wins * self.cfg.leverage_step
        if self._losses > 0:
            target /= (1.0 + self._losses)
        target = max(self.cfg.min_leverage, min(target, cap))
        self.stats.chosen_leverage = float(target)
        return float(target)

    # ------------------------------------------------------------------
    # fee math
    # ------------------------------------------------------------------

    def estimate_round_trip_fee_usd(self, notional_usd: float) -> float:
        """Estimate the entry + exit taker fee on ``notional_usd``."""
        if notional_usd <= 0 or self.taker_fee_bps <= 0:
            return 0.0
        per_leg = notional_usd * self.taker_fee_bps / 1e4
        return per_leg * 2.0  # entry + exit

    def tp_target_for(self, notional_usd: float) -> float:
        """Compute the initial TP target in USD-PnL space.

        Two modes, picked by ``cfg.tp_bps``:

        * **bps mode** (``tp_bps > 0``, the post-2026-05 default):
          ``target = notional × tp_bps × 1e-4``. The TP trigger is
          a fixed fraction of the actual entry notional so a $125
          trade and a $2000 trade both require the same price-move %
          to lock in. This is the only mode that produces consistent
          behaviour across compounding sizers — without it, halving
          notional doubles the required price move and rapidly
          becomes unreachable inside the hold window.

        * **legacy USD mode** (``tp_bps == 0``):
          ``target = round_trip_fees × fee_pad_multiple + min_profit_usd``.
          Kept for backward-compat; flip ``tp_bps`` back on once
          you've calibrated bps thresholds.
        """
        tp_bps = float(getattr(self.cfg, "tp_bps", 0.0) or 0.0)
        if tp_bps > 0 and notional_usd > 0:
            return notional_usd * tp_bps * 1e-4
        fees = self.estimate_round_trip_fee_usd(notional_usd)
        return fees * max(0.0, self.cfg.fee_pad_multiple) + max(
            0.0, self.cfg.min_profit_usd
        )

    def sl_level_for(self, notional_usd: float) -> float:
        """Compute the initial SL trigger in USD-PnL space (always ≤ 0).

        Mirrors :meth:`tp_target_for` — bps-of-notional when
        ``cfg.sl_bps > 0``, else the legacy ``-initial_sl_usd``
        constant. Returning a non-positive value keeps the ratchet
        math (SL only moves up) consistent across modes.
        """
        sl_bps = float(getattr(self.cfg, "sl_bps", 0.0) or 0.0)
        if sl_bps > 0 and notional_usd > 0:
            return -abs(notional_usd * sl_bps * 1e-4)
        return -abs(float(self.cfg.initial_sl_usd))

    def extend_step_for(self, notional_usd: float) -> float:
        """Per-position TP extension step (USD).

        bps mode: ``notional × extend_tp_step_bps × 1e-4``.
        USD mode: ``cfg.extend_tp_step_usd`` (unchanged legacy value).
        """
        ext_bps = float(getattr(self.cfg, "extend_tp_step_bps", 0.0) or 0.0)
        tp_bps = float(getattr(self.cfg, "tp_bps", 0.0) or 0.0)
        if tp_bps > 0 and ext_bps > 0 and notional_usd > 0:
            return notional_usd * ext_bps * 1e-4
        return max(0.0, float(self.cfg.extend_tp_step_usd))

    def trailing_giveback_for(self, notional_usd: float) -> float:
        """Per-position trailing give-back (USD)."""
        gb_bps = float(getattr(self.cfg, "trailing_giveback_bps", 0.0) or 0.0)
        sl_bps = float(getattr(self.cfg, "sl_bps", 0.0) or 0.0)
        if sl_bps > 0 and gb_bps > 0 and notional_usd > 0:
            return notional_usd * gb_bps * 1e-4
        return max(0.0, float(self.cfg.trailing_giveback_usd))

    # ------------------------------------------------------------------
    # execution-result subscriber
    # ------------------------------------------------------------------

    def on_execution(self, result) -> None:  # ExecutionResult; avoids cycle
        """Record entries / clean up on closes.

        Called by the engine's ``on_execution`` listener after every
        executor round-trip. We then read the live portfolio to decide
        whether a fill opened, closed, or flipped a position.
        """
        if not self.enabled:
            return
        if not getattr(result, "success", False):
            return
        fills = getattr(result, "fills", []) or []
        if not fills:
            return
        source = getattr(getattr(result, "signal", None), "strategy", "") or ""

        for fill in fills:
            self._absorb_fill(fill, source_strategy=source)

    def _absorb_fill(self, fill: Fill, *, source_strategy: str) -> None:
        key = Portfolio._key(fill.market_id, fill.outcome_id)
        pos = self.portfolio.positions.get(key)

        if pos is None or pos.shares == 0:
            # Flat after the fill: this was a close (full or last leg of a
            # round-trip). Drop tracking and update the win/loss streak.
            self._positions.pop(key, None)
            self._update_streak_on_close(pos)
            self.stats.open_positions = len(self._positions)
            return

        # Net non-flat → entry or refresh. Compute the round-trip fee from
        # the live position (handles partial fills and averaged entries).
        side = "LONG" if pos.shares > 0 else "SHORT"
        notional = abs(pos.shares) * fill.price
        fees_round_trip = self.estimate_round_trip_fee_usd(notional)
        # Use the actually-paid fee from the entry leg when we have it,
        # so the manager honours the executor's real cost rather than an
        # estimate. The estimate still drives the exit-side projection.
        if getattr(fill, "fee", 0.0):
            fees_round_trip = max(fees_round_trip, float(fill.fee) * 2.0)
        tp_target = self._initial_tp_target(notional, fees_round_trip)
        sl_level = self.sl_level_for(notional)
        extend_step = self.extend_step_for(notional)
        trailing_giveback = self.trailing_giveback_for(notional)

        existing = self._positions.get(key)
        if existing is None or existing.side != side:
            self._positions[key] = _GreedyPosition(
                market_id=fill.market_id,
                outcome_id=fill.outcome_id,
                side=side,
                entry_price=float(pos.avg_cost),
                entry_shares=float(abs(pos.shares)),
                entry_notional=float(notional),
                leverage=float(getattr(fill, "leverage", 1.0) or 1.0),
                entry_time=float(fill.timestamp or self._clock()),
                fees_round_trip_usd=float(fees_round_trip),
                tp_target_usd=float(tp_target),
                sl_level_usd=float(sl_level),
                best_pnl_usd=0.0,
                source_strategy=source_strategy,
                extend_step_usd=float(extend_step),
                trailing_giveback_usd=float(trailing_giveback),
            )
            log.debug(
                "greedy: tracking %s %s shares=%g price=%g lev=%g notional=$%.2f "
                "fees=$%.4f tp=$%.4f sl=$%.4f ext=$%.4f gb=$%.4f",
                fill.market_id, side, abs(pos.shares), pos.avg_cost,
                getattr(fill, "leverage", 1.0), notional, fees_round_trip,
                tp_target, sl_level, extend_step, trailing_giveback,
            )
        else:
            # Same-side add: refresh shares / avg cost; keep best PnL
            # but recompute fees + tp_target + sl for the new notional.
            existing.entry_price = float(pos.avg_cost)
            existing.entry_shares = float(abs(pos.shares))
            existing.entry_notional = float(notional)
            existing.fees_round_trip_usd = float(fees_round_trip)
            # Extend TP target proportionally — don't shrink an already-
            # raised tp_target on adds.
            existing.tp_target_usd = max(
                existing.tp_target_usd, float(tp_target)
            )
            # SL recomputes to the new notional but only ever moves UP
            # toward profit (the trailing ratchet's invariant).
            existing.sl_level_usd = max(existing.sl_level_usd, float(sl_level))
            existing.extend_step_usd = float(extend_step)
            existing.trailing_giveback_usd = float(trailing_giveback)
            log.debug(
                "greedy: refreshed %s shares=%g notional=$%.2f tp=$%.4f sl=$%.4f",
                fill.market_id, existing.entry_shares,
                existing.entry_notional, existing.tp_target_usd,
                existing.sl_level_usd,
            )
        self.stats.open_positions = len(self._positions)

    def _initial_tp_target(self, notional: float, fees_round_trip: float) -> float:
        """Compute the initial TP target for a fill of ``notional`` size.

        In bps mode (``cfg.tp_bps > 0``) returns ``notional × tp_bps × 1e-4``
        — the price-move % required to hit TP stays constant regardless of
        how much the compounding sizer actually opened.

        In legacy USD mode returns
        ``fees_round_trip × fee_pad_multiple + min_profit_usd``. Position-PnL
        trigger that, after the exit fee is paid, leaves ``min_profit_usd``
        of net realised profit. The SL counterpart sits at
        ``-initial_sl_usd`` of position PnL, so realised loss after the exit
        fee is ``-(initial_sl_usd + fees)``. Adding fees to the SL trigger
        would *double-count* them; the proper way to bound absolute losses
        as bankroll grows is the ``max_notional_usd`` cap.
        """
        tp_bps = float(getattr(self.cfg, "tp_bps", 0.0) or 0.0)
        if tp_bps > 0 and notional > 0:
            return notional * tp_bps * 1e-4
        return (
            fees_round_trip * max(0.0, self.cfg.fee_pad_multiple)
            + max(0.0, self.cfg.min_profit_usd)
        )

    def _update_streak_on_close(self, pos) -> None:
        """Bump win/loss streak on a flatten event."""
        # The portfolio's consecutive_losses counter is the source of
        # truth for losses. Use it to detect a fresh loss vs win.
        new_losses = int(self.portfolio.consecutive_losses)
        if new_losses > self._losses:
            self._wins = 0
            self._losses = new_losses
        elif new_losses == 0 and self._losses > 0:
            # Streak was reset by a win.
            self._wins = 1
            self._losses = 0
        else:
            self._wins += 1
        self.stats.consecutive_wins = self._wins
        self.stats.consecutive_losses = self._losses

    # ------------------------------------------------------------------
    # close-signal overlay
    # ------------------------------------------------------------------

    def proposed_closes(
        self,
        markets: Dict[str, Market],
        *,
        now: Optional[float] = None,
    ) -> List[Signal]:
        """Return reduce-only flatten signals for any position that has
        breached its dynamic TP, ratcheted SL, or hold timeout.

        The caller (the engine) should execute these *before* running
        the strategy scan, so greedy exits always beat fresh entries on
        a contested tick.
        """
        if not self.enabled or not self._positions:
            return []
        now = float(now if now is not None else self._clock())
        out: List[Signal] = []

        for key in list(self._positions.keys()):
            gp = self._positions.get(key)
            if gp is None:
                continue
            pos = self.portfolio.positions.get(key)
            if pos is None or pos.shares == 0:
                # Live position vanished underneath us (probably closed
                # by a strategy-side exit). Stop tracking.
                self._positions.pop(key, None)
                self.stats.open_positions = len(self._positions)
                continue

            market = markets.get(gp.market_id)
            if market is None:
                continue
            outcome = market.outcomes.get(gp.outcome_id)
            if outcome is None or outcome.book is None:
                continue
            bid_lvl = outcome.book.best_bid()
            ask_lvl = outcome.book.best_ask()
            if bid_lvl is None or ask_lvl is None:
                continue

            close_price = bid_lvl.price if gp.side == "LONG" else ask_lvl.price
            if close_price <= 0:
                continue
            # PnL_usd computed against the live position so partial-fill
            # math just works.
            pnl_usd = (close_price - pos.avg_cost) * pos.shares
            self._ratchet(gp, pnl_usd)

            kind: Optional[str] = None
            if pnl_usd >= gp.tp_target_usd:
                kind = "greedy-tp"
                self.stats.tp_hits += 1
            elif pnl_usd <= gp.sl_level_usd:
                kind = "greedy-sl"
                self.stats.sl_hits += 1
            elif (
                self.cfg.max_hold_seconds > 0
                and (now - gp.entry_time) > self.cfg.max_hold_seconds
            ):
                kind = "greedy-timeout"
                self.stats.timeout_hits += 1

            if kind is None:
                continue

            sig = self._build_close_signal(gp, pos, close_price, kind, pnl_usd)
            if sig is not None:
                out.append(sig)
                self._positions.pop(key, None)
                self.stats.open_positions = len(self._positions)
                log.info(
                    "greedy EXIT %s %s side=%s pnl=$%+.4f tp=$%.4f sl=$%.4f best=$%.4f",
                    gp.market_id, kind.upper(), gp.side,
                    pnl_usd, gp.tp_target_usd, gp.sl_level_usd, gp.best_pnl_usd,
                )

        return out

    def _ratchet(self, gp: _GreedyPosition, pnl_usd: float) -> None:
        """Update the trailing SL and roll the TP forward on extension.

        Behaviour:
          * Update ``best_pnl_usd`` to the running max.
          * Once ``best_pnl_usd >= lock_in_trigger_ratio × tp_target``,
            raise SL to ``best_pnl_usd − trailing_giveback``. SL
            never moves down.
          * If profit exceeds the current TP target, extend the target
            by the per-position extend step so a winner keeps running.

        Uses the per-position ``extend_step_usd`` and ``trailing_giveback_usd``
        when set (bps mode stamps these at entry from the actual notional),
        falling back to the config-level USD values for legacy mode.
        """
        if pnl_usd > gp.best_pnl_usd:
            gp.best_pnl_usd = pnl_usd

        # Trailing ratchet
        lock_in_at = gp.tp_target_usd * max(0.0, self.cfg.lock_in_trigger_ratio)
        if gp.best_pnl_usd >= lock_in_at and lock_in_at > 0:
            gb = float(gp.trailing_giveback_usd) if gp.trailing_giveback_usd > 0 \
                else max(0.0, self.cfg.trailing_giveback_usd)
            new_sl = gp.best_pnl_usd - gb
            if new_sl > gp.sl_level_usd:
                gp.sl_level_usd = new_sl

        # Greedy extension: roll TP forward past the current high.
        ext = float(gp.extend_step_usd) if gp.extend_step_usd > 0 \
            else max(0.0, self.cfg.extend_tp_step_usd)
        if ext > 0 and gp.best_pnl_usd >= gp.tp_target_usd:
            gp.tp_target_usd = gp.best_pnl_usd + ext

    def _build_close_signal(
        self,
        gp: _GreedyPosition,
        pos,
        close_price: float,
        kind: str,
        pnl_usd: float,
    ) -> Optional[Signal]:
        """Build a reduce-only flatten signal for the live position."""
        close_side = "SELL" if gp.side == "LONG" else "BUY"
        # Size the close to the live notional. The executor's
        # ``_clamp_reduce_only_legs`` will additionally clamp this to
        # the actual open notional so we never over-close.
        size_usd = abs(pos.shares) * close_price
        if size_usd <= 0:
            return None
        leg = Leg(
            market_id=gp.market_id,
            outcome_id=gp.outcome_id,
            side=close_side,
            limit_price=float(close_price),
            size_usd=float(size_usd),
            reason=(
                f"{kind}: pnl=${pnl_usd:+.4f} tp=${gp.tp_target_usd:.4f} "
                f"sl=${gp.sl_level_usd:.4f} best=${gp.best_pnl_usd:.4f}"
            ),
            leverage=float(gp.leverage),
            reduce_only=True,
        )
        return Signal(
            strategy=GREEDY_STRATEGY_NAME,
            confidence=1.0,
            edge=0.01,
            legs=[leg],
            metadata={
                "symbol": gp.market_id,
                "exit": kind,
                "pnl_usd": float(pnl_usd),
                "tp_target_usd": float(gp.tp_target_usd),
                "sl_level_usd": float(gp.sl_level_usd),
                "best_pnl_usd": float(gp.best_pnl_usd),
                "fees_round_trip_usd": float(gp.fees_round_trip_usd),
                "leverage": float(gp.leverage),
                "source_strategy": gp.source_strategy,
            },
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _venue_leverage(market: Optional[Market]) -> float:
        if market is None:
            return 1.0
        try:
            return max(1.0, float(market.metadata.get("leverage", 1.0) or 1.0))
        except (TypeError, ValueError):
            return 1.0
