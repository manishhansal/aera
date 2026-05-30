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


def test_leverage_mismatch_does_not_corrupt_locked_margin():
    """Regression for the live dashboard reporting ``locked_margin=-$78``
    after a long paper run.

    Root cause: per-fill leverage was authoritative for both the
    margin posted (open) and the margin returned (close). When an open
    used one leverage (e.g. greedy's chosen 25x) and the close used a
    slightly different leverage (e.g. a strategy that fell back to 1x
    on a path with no metadata), the incremental ``locked_margin``
    update drifted — across hundreds of fills it ended up wildly
    negative, which silently inflated ``bankroll`` and caused new
    trades to be sized too large.

    Fix: ``locked_margin`` is recomputed from open positions every
    fill. The lone open position dictates the floor; bankroll absorbs
    the leverage delta as part of settled_wealth.
    """
    p = Portfolio(bankroll=1000.0)
    p.apply_fill(Fill(
        timestamp=0, market_id="BTCUSD", outcome_id="BTCUSD",
        side="BUY", price=100.0, size=10.0, leverage=25.0,
    ))
    open_margin = 10.0 * 100.0 / 25.0
    assert math.isclose(p.locked_margin, open_margin)
    assert math.isclose(p.bankroll, 1000.0 - open_margin)
    p.apply_fill(Fill(
        timestamp=1, market_id="BTCUSD", outcome_id="BTCUSD",
        side="SELL", price=100.0, size=10.0, leverage=1.0, fee=0.0,
    ))
    assert p.locked_margin >= 0.0, (
        f"locked_margin went negative ({p.locked_margin}); pre-fix the "
        f"close at lev=1 would have returned 10x the original margin to "
        f"bankroll and left locked_margin at -$960."
    )
    assert math.isclose(p.locked_margin, 0.0, abs_tol=1e-9)
    assert math.isclose(p.settled_wealth, 1000.0, abs_tol=1e-6)


def test_scale_in_then_full_close_zeroes_locked_margin():
    """Two opens at different prices, single close: locked_margin must
    return cleanly to zero regardless of the average-cost weighting."""
    p = Portfolio(bankroll=1000.0)
    p.apply_fill(Fill(0, "BTCUSD", "BTCUSD", "BUY", 100.0, 3.0, leverage=10.0))
    p.apply_fill(Fill(1, "BTCUSD", "BTCUSD", "BUY", 110.0, 2.0, leverage=10.0))
    assert math.isclose(p.position("BTCUSD", "BTCUSD").shares, 5.0)
    assert math.isclose(p.position("BTCUSD", "BTCUSD").avg_cost, 104.0)
    p.apply_fill(Fill(2, "BTCUSD", "BTCUSD", "SELL", 108.0, 5.0, leverage=10.0))
    assert math.isclose(p.locked_margin, 0.0, abs_tol=1e-9)
    # realized = (108 - 104) * 5 = 20
    assert math.isclose(p.position("BTCUSD", "BTCUSD").realised_pnl, 20.0, abs_tol=1e-6)
    assert math.isclose(p.bankroll, 1020.0, abs_tol=1e-6)


def test_dust_residual_is_swept_on_reducing_fill():
    """Regression for ghost positions accumulating on the dashboard.

    Live evidence: after running for ~30 min the bot showed 5 "open"
    positions with shares=-0.00 (rounded) and notionals of $0.01 yet
    realised PnL of -$0.18 to -$0.74 attached to each. Those were the
    residuals of partial closes that the venue refused to flatten
    further (sub-min-contract size) so they accumulated forever,
    silently locking margin and surfacing as dust ghosts.

    Fix: a partial close that leaves the residual below
    ``dust_threshold_usd`` triggers an automatic sweep — book
    ``(mark - avg_cost) × shares`` as realised PnL and zero out the
    position. The next ``apply_fill`` re-derives ``locked_margin``
    from open positions so the swept margin returns to bankroll.
    """
    p = Portfolio(bankroll=1000.0, dust_threshold_usd=1.0)
    # Open a $200 BTC short.
    p.apply_fill(Fill(0, "BTCUSD", "BTCUSD", "SELL", 100.0, 2.0, leverage=25.0))
    assert math.isclose(p.position("BTCUSD", "BTCUSD").shares, -2.0)
    # Close 1.995 (= $199.5) — residual $0.5 notional, below the $1 dust floor.
    p.apply_fill(Fill(1, "BTCUSD", "BTCUSD", "BUY", 100.5, 1.995, leverage=25.0))
    pos = p.position("BTCUSD", "BTCUSD")
    assert pos.shares == 0.0, (
        f"dust residual should be swept to zero, got {pos.shares}"
    )
    assert math.isclose(p.locked_margin, 0.0, abs_tol=1e-9), (
        "locked margin must return to zero after dust sweep"
    )
    # The sweep booked (mark - avg_cost) × shares = (100.5 - 100) × -0.005
    # = -$0.0025 of mark-to-market loss on the residual, on top of the
    # main close's -$1.0 of loss (100 - 100.5) × 1.995 + tiny fee = -$0.998.
    # Total realised ≈ -$1.0 (the main close dominates; dust is a rounding
    # adjustment).
    assert pos.realised_pnl < 0, "loss-side close should book negative PnL"


def test_dust_sweep_disabled_when_threshold_zero():
    """``dust_threshold_usd=0`` preserves the legacy behaviour where dust
    residuals sit on the book forever. Useful when running live on a
    venue that supports arbitrary-size closes (no min contract size)."""
    p = Portfolio(bankroll=1000.0, dust_threshold_usd=0.0)
    p.apply_fill(Fill(0, "BTCUSD", "BTCUSD", "SELL", 100.0, 2.0, leverage=25.0))
    p.apply_fill(Fill(1, "BTCUSD", "BTCUSD", "BUY", 100.5, 1.995, leverage=25.0))
    pos = p.position("BTCUSD", "BTCUSD")
    # Residual 0.005 contracts at $100.5 = $0.5025 notional — kept on book.
    assert math.isclose(abs(pos.shares), 0.005, abs_tol=1e-9)
    assert math.isclose(abs(pos.shares) * pos.avg_cost, 0.5, abs_tol=1e-3)


def test_sweep_dust_bulk_clears_pre_existing_dust():
    """``Portfolio.sweep_dust`` cleans up dust that pre-existed the
    per-fill sweep (e.g. bot restart with stale residuals)."""
    p = Portfolio(bankroll=1000.0, dust_threshold_usd=0.0)
    # Open then partial-close to leave $0.5 of dust (dust sweep disabled).
    p.apply_fill(Fill(0, "BTCUSD", "BTCUSD", "SELL", 100.0, 2.0, leverage=25.0))
    p.apply_fill(Fill(1, "BTCUSD", "BTCUSD", "BUY", 100.0, 1.995, leverage=25.0))
    pos = p.position("BTCUSD", "BTCUSD")
    assert abs(pos.shares) > 0  # dust is on the book

    # Now flip on the dust threshold and sweep with a stale mid.
    p.dust_threshold_usd = 1.0
    swept = p.sweep_dust({"BTCUSD": 102.0})  # +2 % adverse-for-short mid
    assert swept == ["BTCUSD"]
    pos = p.position("BTCUSD", "BTCUSD")
    assert pos.shares == 0.0
    # Sweep booked (102 - 100) × -0.005 = -$0.01 of mark-to-market loss on
    # the residual. Total realised includes the main close's PnL + dust.
    assert pos.realised_pnl <= 0.0
    assert math.isclose(p.locked_margin, 0.0, abs_tol=1e-9)


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
