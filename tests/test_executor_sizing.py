"""Executor trade-size targeting + paper-exchange fee accounting.

Three areas covered here:

* ``Executor._size_signal`` — verifies that ``risk.trade_size_fraction``
  is honoured (largest leg = target × buying_power), capped by
  ``max_trade_fraction``, and that the legacy "scale down only" mode
  still works when the target is disabled.
* ``DeltaPaperExchange`` — verifies ``taker_fee_bps`` flows through to
  ``Fill.fee`` and is then deducted from bankroll by ``Portfolio.apply_fill``.
* ``RiskManager.vet`` — leverage-aware bankroll + per-market exposure
  checks, and the ``reduce_only`` bypass for closing legs.
"""
from __future__ import annotations

import math

import pytest

from aera.core import Portfolio, RiskManager
from aera.execution import (
    DeltaPaperExchange,
    Executor,
    LinearSlippageModel,
)
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.settings import RiskConfig
from aera.strategies import Leg, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_delta_market(
    symbol: str = "ETHUSD",
    price: float = 100.0,
    *,
    contract_value: float = 1.0,
    spread_bps: float = 50.0,
) -> Market:
    """Synthetic Delta market with a proportional spread.

    ``spread_bps`` is the full spread expressed in basis points of mid,
    split symmetrically around ``price``. Default 50 bps = 0.25% on each
    side, tight enough that a BUY at the mid limit can fill at the ask.
    """
    half = price * spread_bps * 5e-5  # = price * (spread_bps/10_000) / 2
    book = OrderBook()
    book.replace(bids=[(price - half, 100.0)], asks=[(price + half, 100.0)])
    return Market(
        id=symbol, slug=symbol.lower(), question=f"{symbol} perp",
        category="perpetual_futures",
        outcomes={symbol: Outcome(id=symbol, label=DELTA_OUTCOME_LABEL, book=book)},
        venue="delta",
        metadata={"contract_value": contract_value, "leverage": 50.0},
    )


def _make_executor(
    *,
    trade_size_fraction: float,
    max_trade_fraction: float = 0.5,
    max_market_exposure: float = 0.5,
    bankroll: float = 100.0,
) -> tuple[Executor, Portfolio]:
    cfg = RiskConfig(
        kelly_fraction=0.25,
        max_trade_fraction=max_trade_fraction,
        trade_size_fraction=trade_size_fraction,
        max_market_exposure=max_market_exposure,
    )
    portfolio = Portfolio(bankroll=bankroll)
    risk = RiskManager(cfg, portfolio)
    exchange = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=10.0,
    )
    return Executor(portfolio, risk, exchange), portfolio


# ---------------------------------------------------------------------------
# Executor._size_signal — target mode (trade_size_fraction > 0)
# ---------------------------------------------------------------------------


def test_target_mode_scales_largest_leg_up_to_target():
    """Strategy emits a tiny $1 leg; executor configured for 50% target on
    a $100 bankroll must scale it up to ~$50 (leverage=1 cash leg)."""
    executor, _ = _make_executor(trade_size_fraction=0.5, bankroll=100.0)
    sig = Signal(
        strategy="t", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="m", outcome_id="Y", side="BUY",
                  limit_price=0.50, size_usd=1.0)],
    )
    sized = executor._size_signal(sig)
    assert sized.legs[0].size_usd == pytest.approx(50.0)
    assert sized.metadata["sized"] == pytest.approx(50.0)  # scale factor x50


def test_target_mode_scales_largest_leg_down_when_strategy_oversizes():
    executor, _ = _make_executor(trade_size_fraction=0.5, bankroll=100.0)
    sig = Signal(
        strategy="t", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="m", outcome_id="Y", side="BUY",
                  limit_price=0.50, size_usd=200.0)],
    )
    sized = executor._size_signal(sig)
    assert sized.legs[0].size_usd == pytest.approx(50.0)


def test_target_mode_caps_at_max_trade_fraction():
    """trade_size_fraction=0.9 with a stricter max_trade_fraction=0.3 must
    clamp at 30% of bankroll, not 90%."""
    executor, _ = _make_executor(
        trade_size_fraction=0.9, max_trade_fraction=0.3, bankroll=100.0,
    )
    sig = Signal(
        strategy="t", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="m", outcome_id="Y", side="BUY",
                  limit_price=0.50, size_usd=1.0)],
    )
    sized = executor._size_signal(sig)
    assert sized.legs[0].size_usd == pytest.approx(30.0)


def test_target_mode_preserves_leg_ratios_for_multi_leg_signals():
    """Multi-leg signal: legs must keep their original size ratio after
    sizing so the strategy's intended balance is preserved."""
    executor, _ = _make_executor(trade_size_fraction=0.4, bankroll=100.0)
    sig = Signal(
        strategy="multi", confidence=1.0, edge=0.05,
        legs=[
            Leg(market_id="m", outcome_id="A", side="BUY",
                limit_price=0.40, size_usd=2.0),
            Leg(market_id="m", outcome_id="B", side="BUY",
                limit_price=0.55, size_usd=1.0),  # half the size of leg 0
        ],
    )
    sized = executor._size_signal(sig)
    # Largest leg should hit target (40% of $100 = $40) and the smaller leg
    # should be exactly half of it.
    assert sized.legs[0].size_usd == pytest.approx(40.0)
    assert sized.legs[1].size_usd == pytest.approx(20.0)


def test_target_mode_clamps_when_total_would_exceed_cash():
    """Many-leg signal whose target total would exceed 99% bankroll —
    the cash cap must shrink the scale further."""
    executor, _ = _make_executor(trade_size_fraction=0.5, bankroll=100.0)
    legs = [
        Leg(market_id="m", outcome_id=f"O{i}", side="BUY",
            limit_price=0.10, size_usd=1.0)
        for i in range(5)  # 5 equal legs
    ]
    sig = Signal(strategy="multi", confidence=1.0, edge=0.05, legs=legs)
    sized = executor._size_signal(sig)
    total = sum(l.size_usd for l in sized.legs)
    # 5 equal legs at 50% each would be 250% of bankroll → cash cap clamps to 99%.
    assert total <= 99.0 + 1e-6
    assert math.isclose(total, 99.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Executor._size_signal — legacy mode (trade_size_fraction = 0)
# ---------------------------------------------------------------------------


def test_legacy_mode_only_scales_down_never_up():
    """With trade_size_fraction=0, a small strategy-emitted leg must NOT
    be scaled up — Kelly / strategy-side sizing is preserved."""
    executor, _ = _make_executor(
        trade_size_fraction=0.0, max_trade_fraction=0.5, bankroll=100.0,
    )
    sig = Signal(
        strategy="kelly", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="m", outcome_id="Y", side="BUY",
                  limit_price=0.50, size_usd=1.0)],
    )
    sized = executor._size_signal(sig)
    assert sized.legs[0].size_usd == pytest.approx(1.0)
    assert "sized" not in sized.metadata  # untouched signal


def test_legacy_mode_still_caps_oversized_leg():
    executor, _ = _make_executor(
        trade_size_fraction=0.0, max_trade_fraction=0.1, bankroll=100.0,
    )
    sig = Signal(
        strategy="kelly", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="m", outcome_id="Y", side="BUY",
                  limit_price=0.50, size_usd=80.0)],  # 80% > 10% cap
    )
    sized = executor._size_signal(sig)
    assert sized.legs[0].size_usd == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# DeltaPaperExchange fee accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delta_paper_exchange_charges_taker_fee():
    market = _make_delta_market(price=100.0)
    px = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=10.0,
        taker_fee_bps=5.0,
    )
    leg = Leg(market_id="ETHUSD", outcome_id="ETHUSD",
              side="BUY", limit_price=100.5, size_usd=200.0)
    fill = await px.submit(leg, market)
    assert fill is not None
    notional = fill.size * fill.price
    assert fill.fee == pytest.approx(notional * 5e-4, rel=1e-3)
    assert fill.fee > 0.0


@pytest.mark.asyncio
async def test_delta_paper_exchange_zero_fee_by_default():
    market = _make_delta_market(price=100.0)
    px = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=10.0,
    )
    leg = Leg(market_id="ETHUSD", outcome_id="ETHUSD",
              side="BUY", limit_price=100.5, size_usd=200.0)
    fill = await px.submit(leg, market)
    assert fill is not None
    assert fill.fee == 0.0


@pytest.mark.asyncio
async def test_delta_paper_fee_deducts_from_portfolio_bankroll():
    """End-to-end: fee on a paper fill must reduce bankroll by exactly that
    amount on top of the margin commitment."""
    market = _make_delta_market(price=100.0)
    px = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=10.0,
        taker_fee_bps=10.0,
    )
    leg = Leg(market_id="ETHUSD", outcome_id="ETHUSD",
              side="BUY", limit_price=100.5, size_usd=200.0, leverage=50.0)
    fill = await px.submit(leg, market)
    assert fill is not None

    portfolio = Portfolio(bankroll=100.0)
    pre = portfolio.bankroll
    portfolio.apply_fill(fill)
    # Margin = notional / leverage moves from free to locked; fee always
    # comes out of free cash.
    notional = fill.size * fill.price
    margin = notional / max(fill.leverage, 1.0)
    assert portfolio.bankroll == pytest.approx(pre - margin - fill.fee)


# ---------------------------------------------------------------------------
# Settings round-trip (defaults are documented to be 0.5 / 0.5)
# ---------------------------------------------------------------------------


def test_risk_config_defaults():
    cfg = RiskConfig()
    assert cfg.trade_size_fraction == 0.5
    assert cfg.max_trade_fraction == 0.5
    # The cap must always be at least the target, otherwise the target is
    # silently clipped which is a confusing default. Treat this as a guard.
    assert cfg.max_trade_fraction >= cfg.trade_size_fraction


# ---------------------------------------------------------------------------
# Leverage-aware sizing (Delta perps)
# ---------------------------------------------------------------------------


def test_leverage_aware_target_scales_against_buying_power():
    """A leg with leverage=100 should be sized against (bankroll x 100), not
    bare bankroll. With $20 bankroll, 100x leverage, and a 50% target, the
    largest leg should land at $1000 of notional."""
    executor, _ = _make_executor(
        trade_size_fraction=0.5, max_trade_fraction=0.5, bankroll=20.0,
    )
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=1.0, leverage=100.0)],
    )
    sized = executor._size_signal(sig)
    # target = 0.5 * 20 * 100 = 1000
    assert sized.legs[0].size_usd == pytest.approx(1000.0)
    assert sized.metadata["leverage"] == pytest.approx(100.0)
    # Leverage must be carried through onto the resized leg so downstream
    # vet() still treats it as leveraged.
    assert sized.legs[0].leverage == pytest.approx(100.0)


def test_leverage_aware_cap_clamps_target():
    """A 90% target with 50% max_trade_fraction must clamp the *scaled*
    notional to 50% of buying power, not 90%."""
    executor, _ = _make_executor(
        trade_size_fraction=0.9, max_trade_fraction=0.5, bankroll=20.0,
    )
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=1.0, leverage=100.0)],
    )
    sized = executor._size_signal(sig)
    # cap = 0.5 * 20 * 100 = 1000
    assert sized.legs[0].size_usd == pytest.approx(1000.0)


def test_leverage_aware_legacy_mode_caps_against_buying_power():
    """In legacy mode (trade_size_fraction=0), an oversized leg should still
    be allowed up to the leverage-aware cap (not the cash bankroll cap)."""
    executor, _ = _make_executor(
        trade_size_fraction=0.0, max_trade_fraction=0.5, bankroll=20.0,
    )
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=2000.0, leverage=100.0)],
    )
    sized = executor._size_signal(sig)
    # cap = 0.5 * 20 * 100 = 1000; original $2000 must shrink to $1000.
    assert sized.legs[0].size_usd == pytest.approx(1000.0)


def test_cash_legs_unaffected_by_leverage_path():
    """Sanity: a non-leveraged leg (leverage defaults to 1.0) sizes exactly
    against bankroll alone, not buying power."""
    executor, _ = _make_executor(trade_size_fraction=0.5, bankroll=100.0)
    sig = Signal(
        strategy="t", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="m", outcome_id="Y", side="BUY",
                  limit_price=0.50, size_usd=1.0)],
    )
    sized = executor._size_signal(sig)
    assert sized.legs[0].size_usd == pytest.approx(50.0)
    assert sized.legs[0].leverage == 1.0  # default preserved


# ---------------------------------------------------------------------------
# RiskManager.vet — leverage-aware bankroll check
# ---------------------------------------------------------------------------


def test_risk_vet_rejects_unleveraged_stake_over_bankroll():
    cfg = RiskConfig(max_trade_fraction=1.0)
    portfolio = Portfolio(bankroll=20.0)
    risk = RiskManager(cfg, portfolio)
    decision = risk.vet(market_id="m", outcome_id="Y",
                        stake_usd=100.0, market_price=0.5, leverage=1.0)
    assert not decision.allow
    assert "bankroll" in decision.reason.lower()


def test_risk_vet_allows_leveraged_stake_over_bankroll():
    """100x leverage on a $1000 notional => $10 margin, which fits in $20
    bankroll. The pre-leverage path used to reject this."""
    cfg = RiskConfig(max_trade_fraction=1.0, max_market_exposure=1.0)
    portfolio = Portfolio(bankroll=20.0)
    risk = RiskManager(cfg, portfolio)
    decision = risk.vet(market_id="DOGEUSD", outcome_id="DOGEUSD",
                        stake_usd=1000.0, market_price=0.10, leverage=100.0)
    assert decision.allow, decision.reason


def test_risk_vet_market_exposure_scales_with_leverage():
    """A 30% market exposure cap on a $20 bankroll with 100x leverage allows
    up to $600 of notional in one market — not $6 (which is what the
    pre-leverage path enforced)."""
    cfg = RiskConfig(max_market_exposure=0.30, max_trade_fraction=1.0)
    portfolio = Portfolio(bankroll=20.0)
    risk = RiskManager(cfg, portfolio)
    ok = risk.vet(market_id="DOGEUSD", outcome_id="DOGEUSD",
                  stake_usd=500.0, market_price=0.10, leverage=100.0)
    too_big = risk.vet(market_id="DOGEUSD", outcome_id="DOGEUSD",
                       stake_usd=700.0, market_price=0.10, leverage=100.0)
    assert ok.allow, ok.reason
    assert not too_big.allow
    assert "market exposure" in too_big.reason


# ---------------------------------------------------------------------------
# reduce_only / closing-leg path (the TP/SL fix)
# ---------------------------------------------------------------------------


def _open_short_position(portfolio: Portfolio, *, market_id: str, outcome_id: str,
                          notional: float, price: float) -> None:
    """Helper: directly install a SHORT Position into the Portfolio without
    routing through ``apply_fill``.

    We bypass ``apply_fill`` because Portfolio's cash accounting treats a SELL
    as "received cash for shares" (unhelpful when we only need the Position
    to look open). For these tests we only need ``shares < 0`` so the
    executor's reduce-only path can see an open short — bankroll mutations
    are not relevant.
    """
    from aera.core.portfolio import Position

    key = Portfolio._key(market_id, outcome_id)
    portfolio.positions[key] = Position(
        market_id=market_id,
        outcome_id=outcome_id,
        shares=-(notional / price),
        avg_cost=price,
    )


def test_risk_vet_skips_exposure_cap_for_reduce_only_legs():
    """Pre-fix bug: closing a near-cap position would always fail because
    `existing + new` exceeded the cap even though the close shrinks
    exposure. reduce_only must bypass that check."""
    cfg = RiskConfig(max_market_exposure=0.30, max_trade_fraction=1.0)
    portfolio = Portfolio(bankroll=20.0)
    _open_short_position(portfolio, market_id="DOGEUSD", outcome_id="DOGEUSD",
                          notional=1000.0, price=0.10)
    risk = RiskManager(cfg, portfolio)
    # Without reduce_only, the close gets blocked (existing $1000 + new $1000
    # blows past the $600 cap).
    rejected = risk.vet(market_id="DOGEUSD", outcome_id="DOGEUSD",
                        stake_usd=1000.0, market_price=0.10, leverage=100.0)
    assert not rejected.allow
    # With reduce_only, the same close is allowed through.
    allowed = risk.vet(market_id="DOGEUSD", outcome_id="DOGEUSD",
                       stake_usd=1000.0, market_price=0.10, leverage=100.0,
                       reduce_only=True)
    assert allowed.allow, allowed.reason


def test_executor_clamps_reduce_only_to_open_position_notional():
    """Strategy emits a $1,000 close, but only $10 is actually open. The
    executor must shrink the close to $10 so it doesn't flip the position."""
    executor, portfolio = _make_executor(
        trade_size_fraction=0.5, max_trade_fraction=0.5, bankroll=20.0,
    )
    _open_short_position(portfolio, market_id="DOGEUSD", outcome_id="DOGEUSD",
                          notional=10.0, price=0.10)
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=1000.0,
                  leverage=100.0, reduce_only=True)],
    )
    clamped = executor._clamp_reduce_only_legs(sig)
    assert clamped.legs[0].size_usd == pytest.approx(10.0)
    assert clamped.legs[0].reduce_only is True
    # And the size pass must leave it alone (it's already correctly sized).
    sized = executor._size_signal(clamped)
    assert sized.legs[0].size_usd == pytest.approx(10.0)


def test_executor_drops_reduce_only_when_no_open_position():
    """If the strategy emits a close on a market we're not in, the executor
    must drop the leg, not blindly open a fresh opposite position."""
    executor, portfolio = _make_executor(
        trade_size_fraction=0.5, max_trade_fraction=0.5, bankroll=20.0,
    )
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=1000.0,
                  leverage=100.0, reduce_only=True)],
    )
    clamped = executor._clamp_reduce_only_legs(sig)
    assert clamped.legs == []


def test_executor_drops_reduce_only_when_wrong_direction():
    """SELL on an already-short position is not a close — it would double
    the short. The executor must drop it."""
    executor, portfolio = _make_executor(
        trade_size_fraction=0.5, max_trade_fraction=0.5, bankroll=20.0,
    )
    _open_short_position(portfolio, market_id="DOGEUSD", outcome_id="DOGEUSD",
                          notional=10.0, price=0.10)
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="SELL",
                  limit_price=0.10, size_usd=1000.0,
                  leverage=100.0, reduce_only=True)],
    )
    clamped = executor._clamp_reduce_only_legs(sig)
    assert clamped.legs == []


@pytest.mark.asyncio
async def test_executor_execute_closes_short_position_end_to_end():
    """End-to-end: install a short position, then run a reduce_only BUY
    close through `execute`. The close must succeed (not get rejected by the
    risk vet's exposure cap)."""
    executor, portfolio = _make_executor(
        trade_size_fraction=0.5, max_trade_fraction=0.5,
        max_market_exposure=1.0, bankroll=20.0,
    )
    market = _make_delta_market(symbol="ETHUSD", price=100.0)
    _open_short_position(portfolio, market_id="ETHUSD", outcome_id="ETHUSD",
                          notional=800.0, price=100.0)
    pre_close_shares = portfolio.positions["ETHUSD:ETHUSD"].shares
    assert pre_close_shares < 0  # confirm short

    close = Signal(
        strategy="t", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="BUY",
                  limit_price=100.5, size_usd=1000.0,
                  leverage=50.0, reduce_only=True)],
    )
    result = await executor.execute(close, {"ETHUSD": market})
    assert result.success, result.reason
    assert len(result.fills) == 1
    # Position should be flat (or near-flat) after the close.
    post_shares = portfolio.positions["ETHUSD:ETHUSD"].shares
    assert abs(post_shares) < abs(pre_close_shares)


@pytest.mark.asyncio
async def test_executor_execute_rejects_close_with_no_position():
    executor, _ = _make_executor(
        trade_size_fraction=0.5, max_trade_fraction=0.5, bankroll=20.0,
    )
    market = _make_delta_market(symbol="ETHUSD", price=100.0)
    close = Signal(
        strategy="t", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="BUY",
                  limit_price=100.5, size_usd=10.0, reduce_only=True)],
    )
    result = await executor.execute(close, {"ETHUSD": market})
    assert not result.success
    assert "no open position" in result.reason


# ---------------------------------------------------------------------------
# Misconfiguration: trade_size_fraction > max_market_exposure
# (regression for "every Delta signal gets REJECTED on a fresh bankroll")
# ---------------------------------------------------------------------------


def test_risk_vet_reject_message_includes_stake_and_cap():
    """The reject message must surface the *actual* stake and cap so a user
    can diagnose the common ``trade_size_fraction > max_market_exposure``
    misconfiguration from the dashboard alone."""
    cfg = RiskConfig(max_market_exposure=0.30, max_trade_fraction=1.0)
    portfolio = Portfolio(bankroll=1000.0)
    risk = RiskManager(cfg, portfolio)
    decision = risk.vet(
        market_id="DOGEUSD", outcome_id="DOGEUSD",
        stake_usd=25_000.0, market_price=0.10, leverage=50.0,
    )
    assert not decision.allow
    msg = decision.reason
    # Stake, would-be total, cap, and remediation hint all surfaced.
    assert "25,000.00" in msg
    assert "15,000.00" in msg  # cap = 0.30 * 1000 * 50
    assert "50,000.00" in msg  # buying power
    assert "max_market_exposure" in msg


def test_sized_signal_busts_market_exposure_cap_when_misconfigured():
    """Regression for the live-dashboard bug: with trade_size_fraction=0.50
    and max_market_exposure=0.30 (per-market cap tighter than per-trade cap),
    a fresh single-leg Delta signal scales past the cap and immediately gets
    vet-rejected. This pins the broken behaviour we expect when the invariant
    is violated, so we notice if it ever changes silently."""
    cfg = RiskConfig(
        kelly_fraction=0.25,
        max_trade_fraction=0.5,
        trade_size_fraction=0.5,
        max_market_exposure=0.30,  # tighter than trade_size_fraction → trap
    )
    portfolio = Portfolio(bankroll=1000.0)
    risk = RiskManager(cfg, portfolio)
    exchange = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=10.0,
    )
    executor = Executor(portfolio, risk, exchange)

    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=1000.0, leverage=50.0)],
    )
    sized = executor._size_signal(sig)
    leg = sized.legs[0]
    # buying_power = 1000 * 50 = 50,000; target = 0.5 * 50,000 = 25,000.
    assert leg.size_usd == pytest.approx(25_000.0)
    decision = risk.vet(
        market_id=leg.market_id, outcome_id=leg.outcome_id,
        stake_usd=leg.size_usd, market_price=leg.limit_price,
        leverage=leg.leverage,
    )
    assert not decision.allow
    # Hint message helps users self-diagnose from the dashboard.
    assert "max_market_exposure" in decision.reason


def test_risk_config_default_satisfies_market_exposure_invariant():
    """Defaults must satisfy max_market_exposure >= max_trade_fraction so
    out-of-the-box trades aren't rejected."""
    cfg = RiskConfig()
    assert cfg.max_market_exposure >= cfg.max_trade_fraction
    assert cfg.max_market_exposure >= cfg.trade_size_fraction
