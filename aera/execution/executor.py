"""Order router with the Delta paper / live exchanges behind a common interface.

Concrete ``Exchange`` implementations live in :mod:`aera.execution.delta_exchange`.
This module only defines the abstract contract, the routing engine, and the
multi-leg sizing logic shared by every Delta strategy.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from aera.core import Fill, Portfolio, RiskManager
from aera.logging import get_logger
from aera.markets import Market
from aera.strategies import Signal, Leg

if TYPE_CHECKING:  # avoid runtime import cycle (core imports execution)
    from aera.core import GreedyTradeManager


log = get_logger(__name__)


class OrderRejected(Exception):
    """Raised by an `Exchange.submit` when an order is rejected pre-submission
    for a *structural* reason that will not resolve on retry.

    Examples:
        * intended notional too small for the venue's minimum contract,
        * unknown outcome / mis-routed leg,
        * authentication missing for a live exchange.

    The executor catches this and surfaces ``reason`` into the
    ``ExecutionResult.reason`` field so the dashboard can show what went
    wrong instead of a generic "all legs failed".

    Exchanges should keep using ``return None`` for *transient* rejections
    (limit price not met, partial-fill that may succeed next tick, etc.)
    so the executor can distinguish "broken" from "wait and retry".
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Exchange(abc.ABC):
    @abc.abstractmethod
    async def submit(self, leg: Leg, market: Market) -> Optional[Fill]:
        """Submit one leg. Return a Fill on success, None on rejection."""


@dataclass
class ExecutionResult:
    signal: Signal
    fills: List[Fill]
    success: bool
    reason: str = ""


class Executor:
    """Atomic multi-leg executor.

    For arbitrage signals we MUST get all legs filled or none — partial fills
    leave the bot directionally exposed. The executor enforces this by:

        1. Pre-vetting every leg through the RiskManager.
        2. Submitting all legs concurrently (asyncio.gather).
        3. If any leg fails, attempting to immediately reverse the successful
           legs at market to flatten exposure.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        risk: RiskManager,
        exchange: Exchange,
        greedy: Optional["GreedyTradeManager"] = None,
    ) -> None:
        self.portfolio = portfolio
        self.risk = risk
        self.exchange = exchange
        # Optional greedy overlay. When attached AND enabled, the
        # executor:
        #   * overrides leg.leverage on fresh entries (non-reduce-only)
        #     with ``greedy.decide_leverage(market)``;
        #   * replaces ``risk.trade_size_fraction`` with
        #     ``greedy.cfg.compound_fraction`` so each entry deploys
        #     almost the whole live bankroll × leverage as buying power.
        # When the greedy overlay is None or disabled, the executor
        # behaves exactly as before.
        self.greedy = greedy

    async def execute(self, signal: Signal, market_lookup: Dict[str, Market]) -> ExecutionResult:
        import asyncio

        # Clamp reduce-only (closing) legs to the actual open position size
        # BEFORE trade-size sizing runs, so the close lands exactly on what's
        # open instead of growing to the configured trade-size target. This
        # is what makes TP/SL signals work after the leverage-aware refactor:
        # without this step the close would be sized like a fresh entry and
        # either over-close (flipping into an opposite position) or get
        # rejected by the exposure cap.
        signal = self._clamp_reduce_only_legs(signal)
        if not signal.legs:
            return ExecutionResult(
                signal=signal, fills=[], success=False,
                reason="no open position to close",
            )

        # Greedy leverage override (only for entry legs — closes keep
        # whatever leverage they were opened at so margin math balances).
        if self.greedy is not None and self.greedy.enabled:
            signal = self._apply_greedy_leverage(signal, market_lookup)

        # Resize the signal to the configured trade-size target / cap.
        signal = self._size_signal(signal)
        if not signal.legs or any(l.size_usd <= 0 for l in signal.legs):
            return ExecutionResult(signal=signal, fills=[], success=False, reason="zero size after scaling")

        greedy_on = self.greedy is not None and self.greedy.enabled
        # Pre-vet
        for leg in signal.legs:
            reduce_only = bool(getattr(leg, "reduce_only", False))
            # In greedy mode, fresh entries bypass the per-market exposure
            # ceiling. The user has explicitly chosen "maximum compounding";
            # the bankroll / margin guard still runs. Closes are unaffected.
            bypass_cap = greedy_on and not reduce_only
            decision = self.risk.vet(
                market_id=leg.market_id,
                outcome_id=leg.outcome_id,
                stake_usd=leg.size_usd,
                market_price=leg.limit_price,
                leverage=float(getattr(leg, "leverage", 1.0) or 1.0),
                reduce_only=reduce_only,
                bypass_market_cap=bypass_cap,
            )
            if not decision.allow:
                return ExecutionResult(signal=signal, fills=[], success=False, reason=decision.reason)

        # Submit
        markets = [market_lookup.get(l.market_id) for l in signal.legs]
        if any(m is None for m in markets):
            return ExecutionResult(signal=signal, fills=[], success=False, reason="missing market")

        responses = await asyncio.gather(
            *(self.exchange.submit(leg, market) for leg, market in zip(signal.legs, markets)),
            return_exceptions=True,
        )
        fills: List[Fill] = []
        failed = False
        reject_reasons: List[str] = []
        for resp in responses:
            if isinstance(resp, OrderRejected):
                failed = True
                reject_reasons.append(resp.reason)
                continue
            if isinstance(resp, Exception):
                failed = True
                reject_reasons.append(f"exchange error: {resp}")
                continue
            if resp is None:
                failed = True
                continue
            fills.append(resp)

        if failed and fills:
            # roll back the partial leg(s)
            log.warning("partial fill, unwinding %d legs", len(fills))
            await self._unwind(fills, market_lookup)
            return ExecutionResult(
                signal=signal, fills=[], success=False,
                reason="partial fill unwound",
            )
        if failed:
            reason = "; ".join(reject_reasons) if reject_reasons else "all legs failed"
            return ExecutionResult(
                signal=signal, fills=[], success=False, reason=reason,
            )

        for f in fills:
            self.portfolio.apply_fill(f)

        return ExecutionResult(signal=signal, fills=fills, success=True)

    def _apply_greedy_leverage(
        self, signal: Signal, market_lookup: Dict[str, Market]
    ) -> Signal:
        """Stamp the greedy-chosen leverage on every fresh-entry leg.

        Closes (``reduce_only=True``) are returned unchanged — they keep
        whatever leverage the position was opened at so the portfolio's
        margin accounting balances on the close.
        """
        if self.greedy is None or not self.greedy.enabled:
            return signal
        new_legs: List[Leg] = []
        changed = False
        for l in signal.legs:
            if getattr(l, "reduce_only", False):
                new_legs.append(l)
                continue
            market = market_lookup.get(l.market_id)
            new_lev = self.greedy.decide_leverage(market)
            if new_lev == getattr(l, "leverage", 1.0):
                new_legs.append(l)
                continue
            changed = True
            new_legs.append(Leg(
                market_id=l.market_id, outcome_id=l.outcome_id, side=l.side,
                limit_price=l.limit_price, size_usd=l.size_usd,
                reason=l.reason + f" (greedy lev {new_lev:g}x)",
                leverage=float(new_lev),
                reduce_only=False,
                # Preserve maker-mode overrides (e.g. bid_ask_spread_fade
                # stamps post_only/gtc on its quotes) so the leverage
                # rewrite doesn't silently re-route a maker leg into a
                # taker IOC submission.
                time_in_force=getattr(l, "time_in_force", None),
                post_only=getattr(l, "post_only", None),
            ))
        if not changed:
            return signal
        return Signal(
            strategy=signal.strategy,
            confidence=signal.confidence,
            edge=signal.edge,
            legs=new_legs,
            metadata={**signal.metadata, "greedy_leverage": True},
        )

    def _clamp_reduce_only_legs(self, signal: Signal) -> Signal:
        """Replace ``reduce_only`` legs with versions sized to the open position.

        The strategy emits a close at ``entry_size_usd`` (the *intended* entry
        notional from when the position was opened), but the actual filled
        position can be smaller if the executor scaled the entry down. We must
        close exactly what's open — not what was intended — otherwise:

          * over-closing flips the position into an opposite leg (e.g. selling
            $1,000 to "close" a $10 short opens a $990 long),
          * under-closing leaves a dust position bleeding fees and margin.

        Procedure for each reduce-only leg:
          1. Look up the open position (signed shares × mark price).
          2. Drop the leg if there's no position to close.
          3. Drop the leg if its side is on the *same* side as the position
             (selling a long short, buying a short long — wrong direction).
          4. Clamp the leg's ``size_usd`` to the open notional. Always smaller
             or equal, never larger.

        Non-reduce-only legs are returned unchanged. A signal whose only legs
        are dropped reduce-only legs returns with an empty leg list and the
        caller surfaces the no-position reason to the user.
        """
        if not any(getattr(l, "reduce_only", False) for l in signal.legs):
            return signal

        new_legs: List[Leg] = []
        for l in signal.legs:
            if not getattr(l, "reduce_only", False):
                new_legs.append(l)
                continue
            pos = self.portfolio.positions.get(
                Portfolio._key(l.market_id, l.outcome_id)
            )
            if pos is None or pos.shares == 0:
                log.info(
                    "executor: dropping reduce-only %s leg on %s — no open position",
                    l.side, l.market_id,
                )
                continue
            # direction check: BUY closes a short (shares < 0), SELL closes a long
            is_closing = (l.side == "BUY" and pos.shares < 0) or (
                l.side == "SELL" and pos.shares > 0
            )
            if not is_closing:
                log.warning(
                    "executor: dropping reduce-only %s leg on %s — would not "
                    "reduce existing position (shares=%g)",
                    l.side, l.market_id, pos.shares,
                )
                continue
            open_notional = abs(pos.shares) * max(l.limit_price, 1e-9)
            clamped = min(l.size_usd, open_notional)
            if clamped <= 0:
                continue
            new_legs.append(Leg(
                market_id=l.market_id, outcome_id=l.outcome_id, side=l.side,
                limit_price=l.limit_price, size_usd=clamped,
                reason=l.reason + (
                    f" (clamped {l.size_usd:.4f}→{clamped:.4f} to open pos)"
                    if clamped < l.size_usd else ""
                ),
                leverage=getattr(l, "leverage", 1.0),
                reduce_only=True,
                time_in_force=getattr(l, "time_in_force", None),
                post_only=getattr(l, "post_only", None),
            ))
        return Signal(
            strategy=signal.strategy, confidence=signal.confidence,
            edge=signal.edge, legs=new_legs,
            metadata={**signal.metadata, "reduce_only_clamped": True},
        )

    def _size_signal(self, signal: Signal) -> Signal:
        """Resize every leg proportionally to honour the configured trade-size
        target and risk caps.

        Two modes, picked by ``risk.trade_size_fraction``:

        * ``trade_size_fraction > 0`` (default): the *largest* leg is scaled
          to ``trade_size_fraction × buying_power`` (capped at
          ``max_trade_fraction × buying_power``). All other legs scale by the
          same factor so multi-leg arbitrage ratios are preserved exactly.
          This is the "trade size knob" path — the executor decides quantity,
          not the strategy.

        * ``trade_size_fraction <= 0``: legacy behaviour — respect the
          strategy-emitted size and only scale **down** to fit the caps.
          Useful when you want Kelly or strategy-side sizing to be the source
          of truth.

        Buying power
        ------------
        ``buying_power = bankroll × leverage`` where ``leverage`` is the
        maximum leverage advertised across the signal's legs (``Leg.leverage``,
        defaulting to ``1.0``). When a leg's leverage is 1.0 the formula
        collapses to ``buying_power = bankroll``, preserving cash-based math.
        On Delta perps, the strategy stamps the configured account leverage
        on every leg so the executor budgets against margin, not cash.

        In both modes the total notional is also clamped to ``0.99 × buying_power``
        so we never try to commit more margin than the bot has.
        """
        if not signal.legs:
            return signal
        bankroll = self.portfolio.bankroll
        if bankroll <= 0:
            return signal

        # Reduce-only (closing) legs are already sized to the open position by
        # ``_clamp_reduce_only_legs``; the trade-size knob would re-inflate
        # them and bust the close. Only consider entry legs for the scale.
        entry_legs = [l for l in signal.legs if not getattr(l, "reduce_only", False)]
        if not entry_legs:
            return signal

        largest = max(l.size_usd for l in entry_legs)
        total = sum(l.size_usd for l in entry_legs)
        if largest <= 0 or total <= 0:
            return signal

        # Use the largest leverage across entry legs as the multiplier.
        # Single-leg leveraged signals (Delta) pick up the perp's leverage;
        # leverage=1.0 legs (e.g. hand-built fixtures) behave as cash.
        leverage = max((float(getattr(l, "leverage", 1.0) or 1.0)
                        for l in entry_legs), default=1.0)
        if leverage <= 0:
            leverage = 1.0
        buying_power = bankroll * leverage

        risk_cfg = self.risk.config
        cap = risk_cfg.max_trade_fraction * buying_power
        cash_cap = buying_power * 0.99
        target_fraction = getattr(risk_cfg, "trade_size_fraction", 0.0) or 0.0
        # Greedy compounding: when the overlay is on, deploy almost the
        # entire bankroll × leverage on every fresh entry. Replaces the
        # risk-config target so the user only has to flip greedy on
        # to get the compounding behaviour.
        if self.greedy is not None and self.greedy.enabled:
            comp = float(getattr(self.greedy.cfg, "compound_fraction", 0.0) or 0.0)
            if comp > 0.0:
                # Cap at the same 99% safety as cash_cap so the bot never
                # tries to commit more margin than it has.
                target_fraction = min(0.99, comp)

        if target_fraction > 0.0:
            # Target mode: pick a uniform scale that drives the largest leg
            # to (target_fraction × buying_power), then clamp to caps.
            # When greedy is active, bypass the per-trade max_trade_fraction
            # ceiling (the user has opted in to maximum compounding) — only
            # the cash_cap remains as a hard safety so the bot never tries
            # to commit more margin than it owns.
            greedy_on = self.greedy is not None and self.greedy.enabled
            if greedy_on:
                target = target_fraction * buying_power
                # Hard absolute-USD ceiling on per-trade notional. Lets the
                # absolute-USD greedy thresholds (min_profit_usd /
                # initial_sl_usd) stay sane regardless of bankroll growth:
                # without it, $100k × 25x × 0.20 = $500k notional, fees =
                # $500, and the $1.50 SL becomes a 0.3 bp cushion that
                # gets steamrolled by spread on every fire.
                max_notional = float(
                    getattr(self.greedy.cfg, "max_notional_usd", 0.0) or 0.0
                )
                if max_notional > 0:
                    target = min(target, max_notional)
            else:
                target = min(target_fraction, risk_cfg.max_trade_fraction) * buying_power
            scale = target / largest
            # ensure neither cap is breached after scaling
            if not greedy_on and largest * scale > cap:
                scale = cap / largest
            if total * scale > cash_cap:
                scale = cash_cap / total
        else:
            # Legacy mode: only ever scale down.
            scale_largest = cap / largest if largest > cap else 1.0
            scale_total = cash_cap / total if total > cash_cap else 1.0
            scale = min(scale_largest, scale_total, 1.0)

        if scale == 1.0:
            return signal

        new_legs = []
        for l in signal.legs:
            # Carry reduce-only legs through unchanged — they were sized at
            # the clamp step against actual open notional.
            if getattr(l, "reduce_only", False):
                new_legs.append(l)
                continue
            new_legs.append(Leg(
                market_id=l.market_id, outcome_id=l.outcome_id, side=l.side,
                limit_price=l.limit_price, size_usd=l.size_usd * scale,
                reason=l.reason + f" (sized x{scale:.4f})",
                leverage=getattr(l, "leverage", 1.0),
                reduce_only=getattr(l, "reduce_only", False),
                # Critical for bid_ask_spread_fade: every quote signal
                # passes through the sizer, and dropping these would
                # silently demote the maker quote into an IOC taker
                # order at the same inside-spread price (which Delta
                # would either fill aggressively at the wrong fee tier
                # or reject for not crossing).
                time_in_force=getattr(l, "time_in_force", None),
                post_only=getattr(l, "post_only", None),
            ))
        return Signal(
            strategy=signal.strategy, confidence=signal.confidence,
            edge=signal.edge, legs=new_legs,
            metadata={**signal.metadata, "sized": scale, "leverage": leverage},
        )

    async def _unwind(self, fills: List[Fill], market_lookup: Dict[str, Market]) -> None:
        for f in fills:
            m = market_lookup.get(f.market_id)
            if m is None:
                continue
            try:
                leverage = float(m.metadata.get("leverage", 1.0) or 1.0)
            except (TypeError, ValueError):
                leverage = 1.0
            reverse = Leg(
                market_id=f.market_id,
                outcome_id=f.outcome_id,
                side="SELL" if f.side == "BUY" else "BUY",
                limit_price=f.price,
                size_usd=f.price * f.size,
                reason="unwind",
                leverage=leverage,
                reduce_only=True,
            )
            try:
                await self.exchange.submit(reverse, m)
            except Exception as exc:
                log.error("unwind failed for %s: %s", f.outcome_id, exc)
