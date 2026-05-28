"""Risk manager halt / cool-down semantics.

The original implementation tripped a single sticky ``_halted = True``
flag on the first drawdown OR loss-streak event and kept the bot in
"manually halted" forever. For a scalping bot taking dozens of trades
per hour this killed sessions after six adjacent losses (= normal
noise). The current implementation:

* keeps drawdown / manual halts STICKY (catastrophic; needs human
  review);
* turns the loss-streak halt into a TIME-LIMITED cool-down that
  auto-resumes after ``loss_streak_cooldown_seconds`` so the bot
  can recover within the session;
* exposes ``resume()`` so the operator can clear all halt states.

This module locks the contract in.
"""
from __future__ import annotations

import pytest

from aera.core.portfolio import Portfolio
from aera.core.risk import RiskManager
from aera.settings import RiskConfig


class _FixedClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> None:
        self.t += dt


def _cfg(**overrides) -> RiskConfig:
    base = dict(
        kelly_fraction=0.25,
        max_trade_fraction=0.50,
        trade_size_fraction=0.50,
        max_market_exposure=0.50,
        max_drawdown=0.50,
        max_consecutive_losses=6,
        loss_streak_cooldown_seconds=300.0,
    )
    base.update(overrides)
    return RiskConfig(**base)


# ---------------------------------------------------------------------------
# loss-streak cool-down: not a permanent halt
# ---------------------------------------------------------------------------


def test_loss_streak_triggers_time_limited_cooldown_not_permanent_halt():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock(t=1000.0)
    rm = RiskManager(_cfg(max_consecutive_losses=3, loss_streak_cooldown_seconds=60.0),
                     pf, clock=clock)

    pf.consecutive_losses = 3
    decision = rm.check_halts()
    assert not decision.allow
    assert "cool-down" in decision.reason.lower()
    assert "3 losses" in decision.reason
    assert rm.halted

    clock.tick(30.0)
    decision = rm.check_halts()
    assert not decision.allow, "still inside the cool-down window"
    assert "30s left" in decision.reason or "31s left" in decision.reason

    clock.tick(31.0)
    decision = rm.check_halts()
    assert decision.allow, "cool-down should have expired"
    assert not rm.halted


def test_loss_streak_cooldown_does_not_retrigger_on_the_same_streak():
    """Once a cool-down expires and the streak is still at the trip
    level (no new loss arrived), we don't immediately re-trip — that
    was the original bug. A fresh trip requires a NEW loss past the
    anchor."""
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock(t=1000.0)
    rm = RiskManager(_cfg(max_consecutive_losses=3, loss_streak_cooldown_seconds=60.0),
                     pf, clock=clock)

    pf.consecutive_losses = 3
    rm.check_halts()
    clock.tick(61.0)
    decision = rm.check_halts()
    assert decision.allow

    # Streak hasn't advanced — staying flat with the same streak count
    # should NOT re-trip.
    decision = rm.check_halts()
    assert decision.allow


def test_loss_streak_cooldown_retriggers_on_a_new_loss_past_anchor():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock(t=1000.0)
    rm = RiskManager(_cfg(max_consecutive_losses=3, loss_streak_cooldown_seconds=60.0),
                     pf, clock=clock)

    pf.consecutive_losses = 3
    rm.check_halts()
    clock.tick(61.0)
    rm.check_halts()  # exits cool-down

    pf.consecutive_losses = 4  # another loss
    decision = rm.check_halts()
    assert not decision.allow, "fresh loss past the previous anchor → new cool-down"
    assert "4 losses" in decision.reason


def test_manual_halt_is_sticky_until_resume():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    rm = RiskManager(_cfg(), pf, clock=clock)

    rm.manual_halt()
    decision = rm.check_halts()
    assert not decision.allow
    assert "manually halted" in decision.reason

    clock.tick(100_000.0)  # time passing doesn't unhalt a manual halt
    decision = rm.check_halts()
    assert not decision.allow

    rm.resume()
    decision = rm.check_halts()
    assert decision.allow


def test_drawdown_halt_is_sticky_until_resume():
    pf = Portfolio(bankroll=100.0)
    pf.peak_bankroll = 100.0
    pf.bankroll = 40.0  # 60% drawdown
    clock = _FixedClock()
    rm = RiskManager(_cfg(max_drawdown=0.50), pf, clock=clock)

    decision = rm.check_halts()
    assert not decision.allow
    assert "drawdown" in decision.reason.lower()

    # Even if the bankroll recovers, the halt is sticky.
    pf.bankroll = 90.0
    decision = rm.check_halts()
    assert not decision.allow

    rm.resume()
    decision = rm.check_halts()
    assert decision.allow


def test_resume_clears_consecutive_loss_counter_so_we_dont_re_trip_instantly():
    """``resume()`` must zero out ``portfolio.consecutive_losses`` —
    otherwise the very next ``check_halts`` would see the same streak
    and immediately re-open the cool-down."""
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    rm = RiskManager(_cfg(max_consecutive_losses=3, loss_streak_cooldown_seconds=60.0),
                     pf, clock=clock)

    pf.consecutive_losses = 5
    rm.check_halts()
    rm.resume()
    assert pf.consecutive_losses == 0
    decision = rm.check_halts()
    assert decision.allow


def test_cooldown_disabled_when_max_consecutive_losses_is_zero():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    rm = RiskManager(_cfg(max_consecutive_losses=0), pf, clock=clock)

    pf.consecutive_losses = 1_000  # arbitrarily large
    decision = rm.check_halts()
    assert decision.allow, "max_consecutive_losses=0 should disable the cool-down"
