"""Portfolio bookkeeping under fills."""
from __future__ import annotations

import math

from aera.core import Portfolio, Fill


def test_buy_updates_bankroll_and_position():
    p = Portfolio(bankroll=10.0)
    p.apply_fill(Fill(timestamp=0, market_id="m", outcome_id="Y",
                      side="BUY", price=0.40, size=10))
    pos = p.position("m", "Y")
    assert math.isclose(p.bankroll, 10.0 - 4.0)
    assert pos.shares == 10
    assert math.isclose(pos.avg_cost, 0.40)


def test_close_realises_pnl():
    p = Portfolio(bankroll=10.0)
    p.apply_fill(Fill(0, "m", "Y", "BUY", 0.40, 10))
    p.apply_fill(Fill(1, "m", "Y", "SELL", 0.55, 10))
    # bought for 4, sold for 5.5
    assert math.isclose(p.bankroll, 10.0 - 4.0 + 5.5)
    assert math.isclose(p.position("m", "Y").realised_pnl, 1.5, abs_tol=1e-9)


def test_growth_multiple_tracks():
    p = Portfolio(bankroll=1.0)
    p.bankroll = 100.0
    assert math.isclose(p.growth_multiple(), 100.0)


def test_consecutive_loss_counter():
    p = Portfolio(bankroll=10.0)
    p.apply_fill(Fill(0, "m", "Y", "BUY", 0.50, 1))
    p.apply_fill(Fill(1, "m", "Y", "SELL", 0.40, 1))   # loss
    assert p.consecutive_losses == 1
    p.apply_fill(Fill(2, "m", "Y", "BUY", 0.50, 1))
    p.apply_fill(Fill(3, "m", "Y", "SELL", 0.60, 1))   # win, resets
    assert p.consecutive_losses == 0


# ---------------------------------------------------------------------------
# Leverage-aware accounting (regression for "bankroll = -$24k after one fill")
# ---------------------------------------------------------------------------


def test_leveraged_open_posts_margin_not_notional():
    """The bug that caused the dashboard's bankroll to drop to -$24k on a
    single $25k BTC paper-fill at 50x leverage: the open used to subtract
    the full notional from bankroll. With leverage on the Fill it must only
    subtract the margin (notional / leverage)."""
    p = Portfolio(bankroll=1000.0)
    # 0.33 BTC contracts at $75,600 = ~$25k notional, 50x leverage = $500 margin.
    p.apply_fill(Fill(
        timestamp=0, market_id="BTCUSD", outcome_id="BTCUSD",
        side="BUY", price=75_600.0, size=0.33, leverage=50.0,
    ))
    assert math.isclose(p.bankroll, 1000.0 - (0.33 * 75_600.0 / 50.0))
    assert math.isclose(p.locked_margin, 0.33 * 75_600.0 / 50.0)
    # settled wealth = bankroll + locked margin, so no wealth was destroyed
    # by simply posting margin.
    assert math.isclose(p.settled_wealth, 1000.0)


def test_leveraged_drawdown_does_not_trip_on_margin_post():
    """Pre-fix: opening a leveraged position pushed bankroll deep negative,
    making drawdown calculation report >100% and tripping the halt. After
    the fix, drawdown is computed against settled_wealth so just posting
    margin produces 0% drawdown."""
    p = Portfolio(bankroll=1000.0)
    p.apply_fill(Fill(
        timestamp=0, market_id="BTCUSD", outcome_id="BTCUSD",
        side="BUY", price=75_600.0, size=0.33, leverage=50.0,
    ))
    assert p.drawdown() < 1e-9, (
        f"opening a margin-posted leg should not register as drawdown, "
        f"got {p.drawdown():.4f}"
    )


def test_leveraged_close_returns_margin_plus_pnl():
    """A 1% mid move in favour of a 50x leveraged long should return the
    posted margin AND the realized PnL to bankroll. End-to-end open + close
    on a $25k notional move = +$250 of realized PnL on $500 margin."""
    p = Portfolio(bankroll=1000.0)
    p.apply_fill(Fill(
        timestamp=0, market_id="BTCUSD", outcome_id="BTCUSD",
        side="BUY", price=75_600.0, size=0.33, leverage=50.0,
    ))
    p.apply_fill(Fill(
        timestamp=1, market_id="BTCUSD", outcome_id="BTCUSD",
        side="SELL", price=76_356.0, size=0.33, leverage=50.0,
    ))
    realised_pnl = (76_356.0 - 75_600.0) * 0.33
    assert math.isclose(p.bankroll, 1000.0 + realised_pnl, abs_tol=1e-6)
    assert math.isclose(p.locked_margin, 0.0, abs_tol=1e-9)
    assert math.isclose(
        p.position("BTCUSD", "BTCUSD").realised_pnl, realised_pnl, abs_tol=1e-6
    )


def test_leveraged_close_at_loss_reduces_bankroll_but_does_not_negate_it():
    """A 1% mid move against the leveraged long: PnL is -$250 on $1000
    starting bankroll. Bankroll should drop by exactly $250 (not by the
    full $25k notional like before the fix)."""
    p = Portfolio(bankroll=1000.0)
    p.apply_fill(Fill(
        timestamp=0, market_id="BTCUSD", outcome_id="BTCUSD",
        side="BUY", price=75_600.0, size=0.33, leverage=50.0,
    ))
    p.apply_fill(Fill(
        timestamp=1, market_id="BTCUSD", outcome_id="BTCUSD",
        side="SELL", price=74_844.0, size=0.33, leverage=50.0,
    ))
    realised_pnl = (74_844.0 - 75_600.0) * 0.33  # negative
    assert math.isclose(p.bankroll, 1000.0 + realised_pnl, abs_tol=1e-6)
    assert p.bankroll > 0  # not negative!
    # drawdown is exactly the realised loss as a fraction of peak.
    expected_dd = -realised_pnl / 1000.0
    assert math.isclose(p.drawdown(), expected_dd, abs_tol=1e-6)


def test_unleveraged_path_is_bit_for_bit_identical_to_before():
    """Regression guard: with leverage=1.0 (default), the new code must
    produce the same bankroll, position, and realized PnL as the pre-
    refactor code for the original test scenarios."""
    p = Portfolio(bankroll=10.0)
    p.apply_fill(Fill(0, "m", "Y", "BUY", 0.40, 10))
    assert math.isclose(p.bankroll, 6.0)  # 10 - 4
    assert math.isclose(p.locked_margin, 4.0)
    assert math.isclose(p.settled_wealth, 10.0)
    p.apply_fill(Fill(1, "m", "Y", "SELL", 0.55, 10))
    # Same end-state as test_close_realises_pnl
    assert math.isclose(p.bankroll, 11.5)
    assert math.isclose(p.locked_margin, 0.0)
    assert math.isclose(p.position("m", "Y").realised_pnl, 1.5)


def test_leveraged_short_then_buy_to_close():
    """Close a short with a BUY: must return short margin and realize PnL."""
    p = Portfolio(bankroll=500.0)
    p.apply_fill(Fill(
        timestamp=0, market_id="DOGEUSD", outcome_id="DOGEUSD",
        side="SELL", price=0.10, size=10_000.0, leverage=50.0,
    ))
    # margin posted = 10_000 * 0.10 / 50 = 20
    assert math.isclose(p.locked_margin, 20.0)
    assert math.isclose(p.bankroll, 480.0)
    assert math.isclose(p.settled_wealth, 500.0)
    # Close at -1% mid (good for short): realised pnl = (0.10 - 0.099) * 10000 = +10
    p.apply_fill(Fill(
        timestamp=1, market_id="DOGEUSD", outcome_id="DOGEUSD",
        side="BUY", price=0.099, size=10_000.0, leverage=50.0,
    ))
    assert math.isclose(p.locked_margin, 0.0, abs_tol=1e-9)
    assert math.isclose(p.bankroll, 510.0, abs_tol=1e-6)
    assert math.isclose(p.position("DOGEUSD", "DOGEUSD").shares, 0.0, abs_tol=1e-9)
