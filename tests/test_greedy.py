"""Greedy autopilot — dynamic TP/SL + leverage selection + fast compounding.

Covers:

* :class:`GreedyTradeManager.decide_leverage` — base, win-streak ramp,
  loss-streak shrink, venue cap.
* :class:`GreedyTradeManager.tp_target_for` — TP = fees + min_profit_usd.
* Position tracking via ``on_execution``: opens, closes, and the
  win/loss streak update on a realised PnL fill.
* :meth:`proposed_closes` — TP / SL / hold-timeout emit reduce_only
  flatten signals; trailing ratchet locks in profit; SL never moves
  down; TP rolls forward when extension is enabled.
* End-to-end :class:`Executor` integration — greedy stamps its chosen
  leverage on entry legs, the compound fraction replaces
  ``trade_size_fraction``, and the per-market exposure cap is bypassed
  for fresh entries when greedy is active.
"""
from __future__ import annotations

import pytest

from aera.core import (
    GREEDY_STRATEGY_NAME,
    Fill,
    GreedyTradeManager,
    Portfolio,
    RiskManager,
)
from aera.core.greedy import _GreedyPosition
from aera.execution import (
    DeltaPaperExchange,
    Executor,
    LinearSlippageModel,
)
from aera.execution.executor import ExecutionResult
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.settings import GreedyConfig, RiskConfig
from aera.strategies import Leg, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_delta_market(
    symbol: str = "ETHUSD",
    price: float = 100.0,
    *,
    venue_leverage: float = 50.0,
    spread_bps: float = 50.0,
) -> Market:
    half = price * spread_bps * 5e-5
    book = OrderBook()
    book.replace(bids=[(price - half, 100.0)], asks=[(price + half, 100.0)])
    return Market(
        id=symbol, slug=symbol.lower(), question=f"{symbol} perp",
        category="perpetual_futures",
        outcomes={symbol: Outcome(id=symbol, label=DELTA_OUTCOME_LABEL, book=book)},
        venue="delta",
        metadata={"contract_value": 1.0, "leverage": venue_leverage},
    )


def _move_market(market: Market, new_price: float, spread_bps: float = 50.0) -> None:
    """Reprice the order book of a market in-place to ``new_price``."""
    outcome = next(iter(market.outcomes.values()))
    half = new_price * spread_bps * 5e-5
    outcome.book.replace(
        bids=[(new_price - half, 100.0)],
        asks=[(new_price + half, 100.0)],
    )


def _greedy_cfg(**overrides) -> GreedyConfig:
    # Default to LEGACY USD mode for these unit tests so the math under
    # test stays the fixed-USD math the assertions encode. New bps-mode
    # tests opt in explicitly with ``tp_bps=...`` / ``sl_bps=...``.
    base = dict(
        enabled=True,
        min_profit_usd=1.0,
        fee_pad_multiple=1.0,
        extend_tp_step_usd=1.0,
        initial_sl_usd=1.5,
        lock_in_trigger_ratio=0.5,
        trailing_giveback_usd=0.5,
        max_hold_seconds=120.0,
        min_leverage=5.0,
        max_leverage=100.0,
        leverage_step=5.0,
        respect_venue_cap=True,
        compound_fraction=0.90,
        fee_override_bps=0.0,
        # Legacy USD mode (post-2026-05 the YAML default is bps mode).
        tp_bps=0.0,
        sl_bps=0.0,
        extend_tp_step_bps=0.0,
        trailing_giveback_bps=0.0,
    )
    base.update(overrides)
    return GreedyConfig(**base)


def _install_long_position(
    portfolio: Portfolio, *, market_id: str, outcome_id: str, shares: float,
    avg_cost: float,
) -> None:
    """Helper: directly install a LONG position into the portfolio so we
    can drive greedy logic without routing through ``apply_fill``."""
    from aera.core.portfolio import Position

    key = Portfolio._key(market_id, outcome_id)
    portfolio.positions[key] = Position(
        market_id=market_id, outcome_id=outcome_id,
        shares=shares, avg_cost=avg_cost,
    )


def _install_short_position(
    portfolio: Portfolio, *, market_id: str, outcome_id: str, shares: float,
    avg_cost: float,
) -> None:
    from aera.core.portfolio import Position

    key = Portfolio._key(market_id, outcome_id)
    portfolio.positions[key] = Position(
        market_id=market_id, outcome_id=outcome_id,
        shares=-abs(shares), avg_cost=avg_cost,
    )


# ---------------------------------------------------------------------------
# Disabled fallthrough
# ---------------------------------------------------------------------------


def test_disabled_manager_returns_venue_leverage_and_no_closes():
    cfg = _greedy_cfg(enabled=False)
    portfolio = Portfolio(bankroll=27.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    market = _make_delta_market(venue_leverage=50.0)
    assert not mgr.enabled
    assert mgr.decide_leverage(market) == 50.0
    assert mgr.proposed_closes({"ETHUSD": market}) == []


# ---------------------------------------------------------------------------
# Leverage selection
# ---------------------------------------------------------------------------


def test_decide_leverage_starts_at_min_with_no_streak():
    cfg = _greedy_cfg(min_leverage=5.0, leverage_step=5.0, max_leverage=100.0)
    portfolio = Portfolio(bankroll=27.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    market = _make_delta_market(venue_leverage=50.0)
    # No wins, no losses → just min_leverage.
    assert mgr.decide_leverage(market) == 5.0


def test_decide_leverage_ramps_up_on_wins_capped_by_venue():
    cfg = _greedy_cfg(min_leverage=5.0, leverage_step=10.0, max_leverage=100.0)
    portfolio = Portfolio(bankroll=27.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    # Venue cap = 25 so even with many wins we cap there.
    market = _make_delta_market(venue_leverage=25.0)
    mgr._wins = 3                                    # min + 3*10 = 35
    assert mgr.decide_leverage(market) == 25.0       # but venue caps at 25


def test_decide_leverage_ignores_venue_when_respect_cap_false():
    cfg = _greedy_cfg(min_leverage=5.0, leverage_step=10.0, max_leverage=100.0,
                      respect_venue_cap=False)
    portfolio = Portfolio(bankroll=27.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    market = _make_delta_market(venue_leverage=10.0)
    mgr._wins = 10
    # min + 10*10 = 105, capped by max_leverage=100.
    assert mgr.decide_leverage(market) == 100.0


def test_decide_leverage_shrinks_on_loss_streak():
    cfg = _greedy_cfg(min_leverage=5.0, leverage_step=5.0, max_leverage=100.0)
    portfolio = Portfolio(bankroll=27.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    market = _make_delta_market(venue_leverage=100.0)
    mgr._wins = 4         # → 5 + 4*5 = 25 nominal
    mgr._losses = 4       # → 25 / 5 = 5 (back to min_leverage)
    assert mgr.decide_leverage(market) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TP target math
# ---------------------------------------------------------------------------


def test_tp_target_is_fees_plus_min_profit():
    """5 bps taker fee on $500 notional: $0.25 each leg → $0.50 round trip,
    plus min_profit_usd=$1 → TP target = $1.50."""
    cfg = _greedy_cfg(min_profit_usd=1.0, fee_pad_multiple=1.0)
    portfolio = Portfolio(bankroll=27.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    assert mgr.estimate_round_trip_fee_usd(500.0) == pytest.approx(0.50)
    assert mgr.tp_target_for(500.0) == pytest.approx(1.50)


def test_tp_target_with_zero_fees_floors_at_min_profit():
    cfg = _greedy_cfg(min_profit_usd=1.0)
    portfolio = Portfolio(bankroll=27.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=0.0)
    assert mgr.tp_target_for(1000.0) == pytest.approx(1.0)


def test_fee_override_bps_takes_precedence():
    cfg = _greedy_cfg(fee_override_bps=10.0, min_profit_usd=0.5)
    portfolio = Portfolio(bankroll=27.0)
    # taker_fee_bps argument is ignored when fee_override_bps > 0.
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=2.0)
    assert mgr.taker_fee_bps == 10.0
    # $200 notional × 10 bps × 2 legs = $0.40 round trip; + $0.50 = $0.90.
    assert mgr.tp_target_for(200.0) == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# bps-of-notional TP / SL (the post-2026-05 default mode)
# ---------------------------------------------------------------------------


def test_tp_target_for_bps_mode_scales_with_notional():
    """40 bps TP target on $2000 notional = $8.00; on $125 = $0.50.

    The whole point of bps mode is that halving notional halves the
    USD trigger so the required PRICE-MOVE % stays constant. This
    fixes the failure mode where a $5 USD TP became a 4 % required
    price move at $125 notional (= unreachable, every trade
    timed out with random PnL drift).
    """
    cfg = _greedy_cfg(tp_bps=40.0, sl_bps=20.0, min_profit_usd=999.0)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    # bps overrides min_profit_usd entirely — it is NOT additive.
    assert mgr.tp_target_for(2000.0) == pytest.approx(8.0)
    assert mgr.tp_target_for(125.0) == pytest.approx(0.5)
    # SL mirrors with the opposite sign.
    assert mgr.sl_level_for(2000.0) == pytest.approx(-4.0)
    assert mgr.sl_level_for(125.0) == pytest.approx(-0.25)


def test_tp_target_falls_back_to_usd_when_bps_disabled():
    """When ``tp_bps == 0`` the manager reverts to the legacy
    ``fees × pad + min_profit_usd`` formula — unchanged from earlier
    versions so the fee_override / pad knobs still behave the same.
    """
    cfg = _greedy_cfg(tp_bps=0.0, sl_bps=0.0,
                       min_profit_usd=1.0, fee_pad_multiple=1.0,
                       initial_sl_usd=2.5)
    portfolio = Portfolio(bankroll=27.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    # Fees = 500 × 5 bps × 2 = $0.50, plus $1 floor → $1.50.
    assert mgr.tp_target_for(500.0) == pytest.approx(1.50)
    # SL is the negation of initial_sl_usd, independent of notional.
    assert mgr.sl_level_for(500.0) == pytest.approx(-2.5)
    assert mgr.sl_level_for(125.0) == pytest.approx(-2.5)


def test_bps_mode_stamps_per_position_extend_and_giveback():
    """bps mode pre-computes the trailing give-back and TP extension
    in USD terms at fill time, so the ratchet uses size-appropriate
    values for that specific trade rather than the config defaults
    (which would be wrong for the actual notional)."""
    cfg = _greedy_cfg(
        tp_bps=40.0, sl_bps=20.0,
        extend_tp_step_bps=10.0, trailing_giveback_bps=8.0,
        extend_tp_step_usd=0.0, trailing_giveback_usd=0.0,
    )
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    fill = _entry_fill(price=100.0, size=12.5, leverage=10.0)  # $1250 notional
    portfolio.apply_fill(fill)
    mgr.on_execution(_exec_result("delta_perp_scalper", [fill]))
    gp = mgr.positions[Portfolio._key("ETHUSD", "ETHUSD")]
    assert gp.tp_target_usd == pytest.approx(5.0)         # 40 bps × $1250
    assert gp.sl_level_usd == pytest.approx(-2.5)         # 20 bps × $1250
    assert gp.extend_step_usd == pytest.approx(1.25)      # 10 bps × $1250
    assert gp.trailing_giveback_usd == pytest.approx(1.0)  # 8 bps × $1250


def test_bps_mode_tp_target_scales_when_compounder_shrinks_notional():
    """Regression for the original "6 % win rate" failure: when the
    actual fill notional collapses well below the ``min_profit_usd``
    calibration, USD mode requires an unreachable price move; bps
    mode keeps the price-move % constant.
    """
    cfg = _greedy_cfg(tp_bps=40.0, sl_bps=20.0)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    # Tiny fill: 1.25 shares of a $100 contract = $125 notional.
    fill = _entry_fill(price=100.0, size=1.25, leverage=25.0)
    portfolio.apply_fill(fill)
    mgr.on_execution(_exec_result("delta_perp_scalper", [fill]))
    gp = mgr.positions[Portfolio._key("ETHUSD", "ETHUSD")]
    # 40 bps of $125 = $0.50 (vs the $5 USD-mode value that needed a
    # 4 % adverse price move to ever hit at this size).
    assert gp.tp_target_usd == pytest.approx(0.5)
    # Price needs to move +50 bps in our favour, i.e. $100 → $100.40,
    # to hit the TP — well inside the hold window's noise band.
    required_price_move_pct = gp.tp_target_usd / (gp.entry_shares * gp.entry_price)
    assert required_price_move_pct == pytest.approx(0.004)  # 40 bps


# ---------------------------------------------------------------------------
# Position tracking via on_execution
# ---------------------------------------------------------------------------


def _entry_fill(market_id="ETHUSD", outcome_id="ETHUSD", *, side="BUY",
                price=100.0, size=5.0, leverage=10.0, fee=0.0) -> Fill:
    return Fill(
        timestamp=1000.0, market_id=market_id, outcome_id=outcome_id,
        side=side, price=price, size=size, fee=fee, leverage=leverage,
    )


def _exec_result(signal_strategy: str, fills: list[Fill]) -> ExecutionResult:
    sig = Signal(
        strategy=signal_strategy, confidence=1.0, edge=0.01,
        legs=[Leg(market_id=f.market_id, outcome_id=f.outcome_id,
                  side=f.side, limit_price=f.price, size_usd=f.size * f.price,
                  leverage=f.leverage) for f in fills],
    )
    return ExecutionResult(signal=sig, fills=fills, success=True)


def test_on_execution_tracks_fresh_long_entry():
    cfg = _greedy_cfg(min_profit_usd=1.0)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    fill = _entry_fill(price=100.0, size=5.0, leverage=10.0)
    portfolio.apply_fill(fill)
    mgr.on_execution(_exec_result("delta_perp_scalper", [fill]))
    key = Portfolio._key("ETHUSD", "ETHUSD")
    assert key in mgr.positions
    gp = mgr.positions[key]
    assert gp.side == "LONG"
    assert gp.leverage == 10.0
    # Notional = 5 × 100 = 500; fees @ 5 bps = $0.50 round trip; TP = $1.50.
    assert gp.tp_target_usd == pytest.approx(1.50)
    assert gp.sl_level_usd == pytest.approx(-1.5)


def test_on_execution_uses_real_fee_when_provided():
    """A fill with a non-zero `fee` field should at least cover the
    actually-paid amount in the fee_round_trip estimate."""
    cfg = _greedy_cfg(min_profit_usd=1.0)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=2.0)
    # Real fee paid = $1.0 on the entry; estimate (200 × 2 bps × 2 legs = $0.08)
    # is well below — manager should pick max(real, estimate).
    fill = _entry_fill(price=100.0, size=2.0, leverage=10.0, fee=1.0)
    portfolio.apply_fill(fill)
    mgr.on_execution(_exec_result("delta_perp_scalper", [fill]))
    gp = mgr.positions[Portfolio._key("ETHUSD", "ETHUSD")]
    # Fees round trip = max(0.08, 2.0) = 2.0; TP = 2.0 + 1.0 = 3.0.
    assert gp.fees_round_trip_usd == pytest.approx(2.0)
    assert gp.tp_target_usd == pytest.approx(3.0)


def test_on_execution_drops_tracking_when_flat_after_close():
    cfg = _greedy_cfg()
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    entry = _entry_fill(side="BUY", price=100.0, size=2.0, leverage=10.0)
    portfolio.apply_fill(entry)
    mgr.on_execution(_exec_result("delta_perp_scalper", [entry]))
    assert Portfolio._key("ETHUSD", "ETHUSD") in mgr.positions

    close = _entry_fill(side="SELL", price=101.0, size=2.0, leverage=10.0)
    portfolio.apply_fill(close)
    mgr.on_execution(_exec_result("delta_perp_scalper", [close]))
    assert Portfolio._key("ETHUSD", "ETHUSD") not in mgr.positions


# ---------------------------------------------------------------------------
# Closes: TP, SL, ratchet, timeout
# ---------------------------------------------------------------------------


def _seed_tracked_long(mgr: GreedyTradeManager, portfolio: Portfolio, *,
                       market_id="ETHUSD", outcome_id="ETHUSD",
                       shares=5.0, avg_cost=100.0, leverage=10.0,
                       tp=1.5, sl=-1.5, entry_time=0.0) -> _GreedyPosition:
    _install_long_position(portfolio, market_id=market_id,
                            outcome_id=outcome_id, shares=shares,
                            avg_cost=avg_cost)
    gp = _GreedyPosition(
        market_id=market_id, outcome_id=outcome_id, side="LONG",
        entry_price=avg_cost, entry_shares=shares,
        entry_notional=shares * avg_cost, leverage=leverage,
        entry_time=entry_time, fees_round_trip_usd=0.5,
        tp_target_usd=tp, sl_level_usd=sl,
    )
    mgr._positions[Portfolio._key(market_id, outcome_id)] = gp
    return gp


def test_proposed_closes_emits_greedy_tp_when_pnl_clears_target():
    cfg = _greedy_cfg(min_profit_usd=1.0, extend_tp_step_usd=0.0)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0,
                              clock=lambda: 1.0)
    market = _make_delta_market(price=100.0)
    _seed_tracked_long(mgr, portfolio, shares=5.0, avg_cost=100.0,
                        tp=1.5, sl=-1.5)

    # Move market up so unrealised PnL on the long crosses the TP target.
    # 5 shares × ($102 − $100) ≈ $10 of PnL (well past $1.5 target).
    _move_market(market, 102.0)
    sigs = mgr.proposed_closes({"ETHUSD": market})
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.strategy == GREEDY_STRATEGY_NAME
    assert sig.metadata["exit"] == "greedy-tp"
    assert sig.legs[0].reduce_only is True
    assert sig.legs[0].side == "SELL"
    # Tracking gets dropped on emit so the next tick doesn't double-close.
    assert Portfolio._key("ETHUSD", "ETHUSD") not in mgr.positions
    assert mgr.stats.tp_hits == 1


def test_proposed_closes_emits_greedy_sl_on_loss():
    cfg = _greedy_cfg(min_profit_usd=1.0, initial_sl_usd=1.5)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0,
                              clock=lambda: 1.0)
    market = _make_delta_market(price=100.0)
    _seed_tracked_long(mgr, portfolio, shares=5.0, avg_cost=100.0,
                        tp=1.5, sl=-1.5)

    # Drop the price so PnL ≈ -$5, below the -$1.5 SL.
    _move_market(market, 99.0)
    sigs = mgr.proposed_closes({"ETHUSD": market})
    assert len(sigs) == 1
    assert sigs[0].metadata["exit"] == "greedy-sl"
    assert mgr.stats.sl_hits == 1


def test_proposed_closes_emits_timeout_when_max_hold_exceeded():
    cfg = _greedy_cfg(max_hold_seconds=10.0, min_profit_usd=10.0,
                      initial_sl_usd=100.0)  # TP/SL impossibly far away
    portfolio = Portfolio(bankroll=100.0)
    clock = {"t": 100.0}
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0,
                              clock=lambda: clock["t"])
    market = _make_delta_market(price=100.0)
    _seed_tracked_long(mgr, portfolio, shares=5.0, avg_cost=100.0,
                        entry_time=100.0)
    # Inside the window — no exit.
    clock["t"] = 105.0
    assert mgr.proposed_closes({"ETHUSD": market}) == []
    # Past the window — timeout fires.
    clock["t"] = 115.0
    sigs = mgr.proposed_closes({"ETHUSD": market})
    assert len(sigs) == 1
    assert sigs[0].metadata["exit"] == "greedy-timeout"
    assert mgr.stats.timeout_hits == 1


def test_ratchet_raises_sl_after_lock_in_trigger():
    cfg = _greedy_cfg(min_profit_usd=1.0, lock_in_trigger_ratio=0.5,
                      trailing_giveback_usd=0.3, extend_tp_step_usd=0.0,
                      initial_sl_usd=1.5)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    gp = _seed_tracked_long(mgr, portfolio, shares=5.0, avg_cost=100.0,
                             tp=2.0, sl=-1.5)
    # Profit = $1.5 — crosses the 50% × $2.0 = $1.0 trigger.
    mgr._ratchet(gp, pnl_usd=1.5)
    assert gp.best_pnl_usd == pytest.approx(1.5)
    # New SL = best - giveback = 1.5 - 0.3 = 1.2 (locked-in).
    assert gp.sl_level_usd == pytest.approx(1.2)
    # A subsequent drop in PnL must NOT lower the SL.
    mgr._ratchet(gp, pnl_usd=0.5)
    assert gp.sl_level_usd == pytest.approx(1.2)
    # best_pnl_usd also never moves down.
    assert gp.best_pnl_usd == pytest.approx(1.5)


def test_ratchet_rolls_tp_target_forward_on_extension():
    cfg = _greedy_cfg(min_profit_usd=1.0, lock_in_trigger_ratio=0.5,
                      trailing_giveback_usd=0.3, extend_tp_step_usd=1.0,
                      initial_sl_usd=1.5)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    gp = _seed_tracked_long(mgr, portfolio, shares=5.0, avg_cost=100.0,
                             tp=2.0, sl=-1.5)
    # PnL crosses TP: target should bump up by extend_tp_step_usd above the new high.
    mgr._ratchet(gp, pnl_usd=2.5)
    assert gp.best_pnl_usd == pytest.approx(2.5)
    # extend = 1.0 → new target = 2.5 + 1.0 = 3.5.
    assert gp.tp_target_usd == pytest.approx(3.5)


def test_short_position_tp_close_emits_buy_side():
    cfg = _greedy_cfg(min_profit_usd=1.0, extend_tp_step_usd=0.0)
    portfolio = Portfolio(bankroll=100.0)
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0,
                              clock=lambda: 1.0)
    market = _make_delta_market(price=100.0)
    _install_short_position(portfolio, market_id="ETHUSD", outcome_id="ETHUSD",
                             shares=5.0, avg_cost=100.0)
    gp = _GreedyPosition(
        market_id="ETHUSD", outcome_id="ETHUSD", side="SHORT",
        entry_price=100.0, entry_shares=5.0, entry_notional=500.0,
        leverage=10.0, entry_time=0.0, fees_round_trip_usd=0.5,
        tp_target_usd=1.5, sl_level_usd=-1.5,
    )
    mgr._positions[Portfolio._key("ETHUSD", "ETHUSD")] = gp

    # Drop price → short PnL goes positive.
    _move_market(market, 98.0)
    sigs = mgr.proposed_closes({"ETHUSD": market})
    assert len(sigs) == 1
    leg = sigs[0].legs[0]
    assert leg.side == "BUY"
    assert leg.reduce_only is True
    assert sigs[0].metadata["exit"] == "greedy-tp"


# ---------------------------------------------------------------------------
# Executor integration: leverage override + compound fraction + cap bypass
# ---------------------------------------------------------------------------


def _make_executor_with_greedy(
    *, bankroll: float = 27.0, max_market_exposure: float = 0.30,
    greedy_kwargs=None,
) -> tuple[Executor, Portfolio, GreedyTradeManager]:
    risk_cfg = RiskConfig(
        kelly_fraction=0.25,
        trade_size_fraction=0.50,
        max_trade_fraction=0.50,
        max_market_exposure=max_market_exposure,
    )
    portfolio = Portfolio(bankroll=bankroll)
    risk = RiskManager(risk_cfg, portfolio)
    exchange = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=100.0,
    )
    cfg = _greedy_cfg(**(greedy_kwargs or {}))
    mgr = GreedyTradeManager(cfg, portfolio, taker_fee_bps=5.0)
    return Executor(portfolio, risk, exchange, greedy=mgr), portfolio, mgr


def test_executor_stamps_greedy_leverage_on_fresh_entry():
    executor, _, mgr = _make_executor_with_greedy(
        greedy_kwargs=dict(min_leverage=20.0, leverage_step=0.0,
                           max_leverage=20.0),
    )
    market = _make_delta_market(venue_leverage=50.0)
    # Original leg with leverage=1 — greedy should override to 20.
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="BUY",
                  limit_price=100.5, size_usd=10.0, leverage=1.0)],
    )
    updated = executor._apply_greedy_leverage(sig, {"ETHUSD": market})
    assert updated.legs[0].leverage == pytest.approx(20.0)
    assert updated.metadata.get("greedy_leverage") is True


def test_executor_skips_greedy_leverage_for_reduce_only_legs():
    executor, _, _ = _make_executor_with_greedy(
        greedy_kwargs=dict(min_leverage=20.0),
    )
    market = _make_delta_market(venue_leverage=50.0)
    sig = Signal(
        strategy="t", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="SELL",
                  limit_price=100.0, size_usd=10.0, leverage=5.0,
                  reduce_only=True)],
    )
    out = executor._apply_greedy_leverage(sig, {"ETHUSD": market})
    # reduce_only legs keep their original leverage so margin math balances.
    assert out.legs[0].leverage == pytest.approx(5.0)


def test_executor_compound_fraction_overrides_trade_size_fraction():
    """With greedy on, compound_fraction (0.90) should drive the largest
    leg to ~90% of bankroll × leverage rather than risk's 50% target."""
    executor, _, _ = _make_executor_with_greedy(
        bankroll=20.0,
        max_market_exposure=10.0,           # huge so cap is irrelevant
        greedy_kwargs=dict(compound_fraction=0.90),
    )
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=1.0, leverage=100.0)],
    )
    sized = executor._size_signal(sig)
    # buying_power = 20 × 100 = 2000; target = 0.9 × 2000 = 1800.
    assert sized.legs[0].size_usd == pytest.approx(1800.0)


def test_executor_compound_fraction_capped_at_99_pct():
    """Even a compound_fraction > 1.0 must be capped so the bot never
    tries to commit more than its bankroll's worth of margin."""
    executor, _, _ = _make_executor_with_greedy(
        bankroll=20.0,
        max_market_exposure=10.0,
        greedy_kwargs=dict(compound_fraction=1.5),
    )
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=1.0, leverage=100.0)],
    )
    sized = executor._size_signal(sig)
    assert sized.legs[0].size_usd == pytest.approx(1980.0)  # 99% × 2000


@pytest.mark.asyncio
async def test_executor_execute_bypasses_market_cap_in_greedy_mode():
    """Without greedy: per-market exposure cap kills a fresh entry whose
    notional exceeds the cap. With greedy: the bypass lets it through.
    """
    executor, _, _ = _make_executor_with_greedy(
        bankroll=20.0,
        max_market_exposure=0.30,                 # tight per-market cap
        greedy_kwargs=dict(compound_fraction=0.90, min_leverage=100.0,
                           max_leverage=100.0, respect_venue_cap=False),
    )
    market = _make_delta_market(symbol="DOGEUSD", price=0.10,
                                 venue_leverage=100.0)
    sig = Signal(
        strategy="delta", confidence=1.0, edge=0.01,
        legs=[Leg(market_id="DOGEUSD", outcome_id="DOGEUSD", side="BUY",
                  limit_price=0.10, size_usd=1.0, leverage=100.0)],
    )
    result = await executor.execute(sig, {"DOGEUSD": market})
    assert result.success, result.reason
    # Sanity: size landed past the would-be cap (cap = 0.3 × 20 × 100 = 600).
    assert result.fills[0].notional > 600.0
