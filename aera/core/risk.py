"""Risk management — Kelly sizing + circuit breakers.

The Kelly criterion is the mathematically optimal staking strategy for
maximising expected log-wealth. For a bet that pays `b:1` on win with
win probability `p`:

    f* = (b * p - (1 - p)) / b   =   p - (1 - p) / b

The helpers here are mostly used by ``compounding.simulate_growth`` for
Monte-Carlo bankroll projections. The Delta perp scalper sizes via the
executor's ``trade_size_fraction`` knob, not Kelly directly, but Kelly
math is kept around as a pricing-agnostic growth utility.

We then multiply ``f*`` by a ``kelly_fraction`` (default 0.25) to control
drawdown — quarter-Kelly halves growth but cuts drawdown variance by 4x.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from aera.logging import get_logger
from aera.settings import RiskConfig

from .portfolio import Portfolio


log = get_logger(__name__)


def kelly_fraction(true_prob: float, market_price: float) -> float:
    """Full-Kelly fraction for a binary $1-on-win bet.

    Buying a share at ``market_price`` pays $1 if the event occurs.
    ``true_prob`` is your estimate of the true probability of a win.
    Returns the fraction of bankroll to stake. Negative => bet the
    opposite side.
    """
    if not (0.0 < market_price < 1.0):
        return 0.0
    if not (0.0 <= true_prob <= 1.0):
        return 0.0

    edge = true_prob - market_price
    if edge == 0:
        return 0.0
    # payout if win = (1 - market_price) per dollar staked
    # loss if loss   = market_price per dollar staked
    # f* = edge / (payout * loss / stake)  simplifies to edge / (market_price*(1-market_price))? no.
    # Standard form: f* = (bp - q) / b with b = (1-price)/price, p=true_prob, q=1-p
    b = (1.0 - market_price) / market_price
    p = true_prob
    q = 1.0 - p
    f = (b * p - q) / b
    return f


def fractional_kelly_bet(
    true_prob: float,
    market_price: float,
    bankroll: float,
    risk: RiskConfig,
) -> float:
    """Return the **dollar** stake for a single bet, clamped by all risk limits.

    A negative return means we should be on the NO side, in which case the caller
    should bet `abs(stake)` on the opposite outcome at price `1 - market_price`.
    """
    f_full = kelly_fraction(true_prob, market_price)
    f = f_full * risk.kelly_fraction
    # absolute cap
    cap = risk.max_trade_fraction
    if f > cap:
        f = cap
    elif f < -cap:
        f = -cap
    return f * bankroll


@dataclass
class RiskDecision:
    allow: bool
    reason: str = ""


class RiskManager:
    """Gatekeeper that vets every trade against bankroll/exposure/drawdown.

    Halt semantics (was: a single sticky ``_halted = True`` that froze
    the bot forever once tripped — kept the user stuck with
    "manually halted" rejections after 6 losses, no recovery):

    * **Manual halt** (``manual_halt()`` / ``resume()``) — still
      sticky. Reserved for explicit operator action.
    * **Drawdown halt** — sticky until ``resume()``. Catastrophic;
      requires human review.
    * **Loss-streak halt** — TIME-LIMITED cool-down (default 5 min).
      Six losses in a row on a scalping bot is normal noise, not a
      reason to kill the bot forever. The bot pauses for
      ``loss_streak_cooldown_seconds``, then auto-resumes IFF the
      portfolio's ``consecutive_losses`` has not climbed further in
      the meantime (the next close resets it on any win).

    The reason string surfaced to the dashboard now distinguishes
    "manually halted" / "drawdown halt" / "cool-down (N losses,
    Ns left)" so the user can see WHY they're stuck.
    """

    def __init__(
        self,
        config: RiskConfig,
        portfolio: Portfolio,
        *,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.config = config
        self.portfolio = portfolio
        self._clock = clock or time.time
        # Sticky manual / drawdown halts.
        self._manual_halt = False
        self._drawdown_halt = False
        # Time-limited cool-down (loss-streak).
        self._cooldown_until: float = 0.0
        self._cooldown_reason: str = ""
        # Snapshot of the loss streak that triggered the cool-down.
        # Lets us detect "still bleeding" vs "we cooled off and a new
        # streak began" so we don't keep extending forever.
        self._cooldown_streak_anchor: int = 0

    @property
    def halted(self) -> bool:
        return self._manual_halt or self._drawdown_halt or self._in_cooldown()

    def _in_cooldown(self) -> bool:
        return self._clock() < self._cooldown_until

    def manual_halt(self) -> None:
        """Operator action: stop everything until ``resume`` is called."""
        self._manual_halt = True

    def resume(self) -> None:
        """Clear ALL halt states so the bot can trade again."""
        self._manual_halt = False
        self._drawdown_halt = False
        self._cooldown_until = 0.0
        self._cooldown_reason = ""
        self._cooldown_streak_anchor = 0
        # Best-effort: don't let stale streak state keep re-tripping
        # the cool-down on the very next vet() call.
        self.portfolio.consecutive_losses = 0

    def check_halts(self) -> RiskDecision:
        # Sticky halts win first.
        if self._manual_halt:
            return RiskDecision(False, "manually halted")
        if self._drawdown_halt:
            return RiskDecision(
                False,
                f"drawdown halt ({self.portfolio.drawdown():.1%} ≥ {self.config.max_drawdown:.0%})",
            )

        # Drawdown trip (sticky from here on).
        if self.portfolio.drawdown() >= self.config.max_drawdown:
            self._drawdown_halt = True
            return RiskDecision(
                False,
                f"drawdown limit {self.config.max_drawdown:.0%} hit",
            )

        # Loss-streak → time-limited cool-down (NOT sticky).
        streak = int(self.portfolio.consecutive_losses)
        cap = int(self.config.max_consecutive_losses)
        cooldown_s = float(getattr(self.config, "loss_streak_cooldown_seconds", 300.0))
        now = float(self._clock())

        if cap > 0 and streak >= cap and not self._in_cooldown():
            # Open a fresh cool-down only when the streak has actually
            # advanced past the previous anchor — protects against the
            # case where the bot ended cool-down, took ONE more loss,
            # and instantly retripped.
            if streak > self._cooldown_streak_anchor:
                self._cooldown_until = now + cooldown_s
                self._cooldown_streak_anchor = streak
                self._cooldown_reason = (
                    f"loss streak cool-down ({streak} losses → pause {cooldown_s:.0f}s)"
                )
                log.warning("risk: %s", self._cooldown_reason)

        if self._in_cooldown():
            remaining = max(0.0, self._cooldown_until - now)
            return RiskDecision(
                False,
                f"loss-streak cool-down ({streak} losses, {remaining:.0f}s left)",
            )

        return RiskDecision(True)

    def vet(
        self,
        market_id: str,
        outcome_id: str,
        stake_usd: float,
        market_price: float,
        leverage: float = 1.0,
        reduce_only: bool = False,
        bypass_market_cap: bool = False,
    ) -> RiskDecision:
        """Vet a single leg.

        ``stake_usd`` is the **notional** USD size of the leg. ``leverage`` is
        the multiplier the venue will apply when converting notional to margin
        (1.0 for spot / cash; higher on leveraged perps). The bankroll check
        compares ``stake_usd / leverage`` against cash, and the per-market
        exposure cap is expressed against ``bankroll × leverage`` so a
        leveraged venue can carry larger notional than a non-leveraged one
        without tripping the same fraction.

        ``reduce_only`` legs are *closing* trades. They can only ever shrink
        an open position (the executor enforces that), so:

          * the per-market exposure cap is **skipped** entirely — adding the
            close to the existing position in the cap math would always
            reject any close on a near-cap position, even though the actual
            post-trade exposure is smaller, not larger;
          * the margin check is also skipped, because closing a leveraged
            position releases margin rather than consuming it.

        Halt conditions (drawdown, manual halt, loss streak) still apply so
        the bot can be hard-stopped while flat regardless of pending closes.

        ``bypass_market_cap`` is set by the executor when the greedy
        autopilot is on and a fresh entry would otherwise be rejected by
        the per-market exposure ceiling — the user has explicitly opted
        into "maximum compounding", so concentrating into one market is
        the intended behaviour. The bankroll / margin check still runs.
        """
        halt = self.check_halts()
        if not halt.allow:
            return halt
        if stake_usd <= 0:
            return RiskDecision(False, "non-positive stake")
        if leverage <= 0:
            leverage = 1.0

        if reduce_only:
            # Closing trades free margin and reduce exposure. The executor
            # has already clamped size to the existing position, so there is
            # nothing left for the risk manager to enforce here.
            return RiskDecision(True)

        required_margin = stake_usd / leverage
        if required_margin > self.portfolio.bankroll:
            return RiskDecision(
                False,
                f"required margin ${required_margin:.4f} (notional ${stake_usd:.4f} "
                f"@ {leverage:g}x) exceeds bankroll ${self.portfolio.bankroll:.4f}",
            )

        # per-market exposure: scale the cap by leverage so the bot can carry
        # bigger leveraged notional in one market without tripping the same
        # fraction. Skipped above for reduce-only legs, and skipped when
        # the caller explicitly opts in via ``bypass_market_cap`` (the
        # greedy compounding path).
        if bypass_market_cap:
            return RiskDecision(True)
        existing = sum(
            p.notional_exposure(market_price)
            for k, p in self.portfolio.positions.items()
            if p.market_id == market_id
        )
        exposure_cap = (
            self.config.max_market_exposure * self.portfolio.bankroll * leverage
        )
        if existing + stake_usd > exposure_cap:
            return RiskDecision(
                False,
                # Include the actual stake and would-be-total so the dashboard
                # shows WHY the cap fired, not just the cap value. The most
                # common misconfiguration is `trade_size_fraction >
                # max_market_exposure`, which makes every first trade scale
                # past the per-market ceiling.
                f"stake ${stake_usd:,.2f} + existing ${existing:,.2f} = "
                f"${existing + stake_usd:,.2f} exceeds market exposure cap "
                f"{self.config.max_market_exposure:.0%} of "
                f"${self.portfolio.bankroll * leverage:,.2f} buying power "
                f"(${exposure_cap:,.2f}). "
                f"Hint: raise risk.max_market_exposure to >= "
                f"trade_size_fraction.",
            )

        return RiskDecision(True)

    def size(
        self,
        true_prob: float,
        market_price: float,
    ) -> float:
        """Return signed dollar size (positive = YES, negative = NO)."""
        return fractional_kelly_bet(
            true_prob, market_price, self.portfolio.bankroll, self.config
        )
