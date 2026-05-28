"""OrderBookSniper + L2 microstructure helpers.

These tests drive the strategy and the underlying depth/tape helpers
deterministically — we feed a controlled clock and synthetic order books
so every entry/exit branch can be exercised without sleeping.
"""
from __future__ import annotations

import math

import pytest

from aera.core import Portfolio
from aera.core.portfolio import Position
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.signals.order_book import (
    TapeInferrer,
    WallSnapshot,
    measure_depth_imbalance,
)
from aera.strategies import OrderBookSniper


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_market(
    symbol: str,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    *,
    tick: float = 0.5,
    leverage: float = 10.0,
) -> Market:
    """Construct a Delta-shaped market with arbitrary L2 ladders."""
    book = OrderBook()
    book.replace(bids=bids, asks=asks)
    outcome = Outcome(id=symbol, label=DELTA_OUTCOME_LABEL, book=book)
    return Market(
        id=symbol,
        slug=symbol.lower(),
        question=f"{symbol} perp",
        category="perpetual_futures",
        outcomes={symbol: outcome},
        venue="delta",
        minimum_tick=tick,
        metadata={"leverage": leverage, "contract_value": 1.0},
    )


def _flat_book(symbol: str, mid: float, *, depth: float = 5.0) -> Market:
    """Symmetric book centred on ``mid`` — neither side wins the imbalance."""
    half = 0.5
    return _make_market(
        symbol,
        bids=[(mid - half, depth), (mid - half - 1, depth)],
        asks=[(mid + half, depth), (mid + half + 1, depth)],
    )


def _stacked_bid(symbol: str, *, wall: float = 100.0, mid: float = 100.0) -> Market:
    """Heavy bid wall vs thin asks — depth ratio >= 3 in the band."""
    return _make_market(
        symbol,
        bids=[(mid - 0.5, wall), (mid - 1.5, wall * 0.5)],
        asks=[(mid + 0.5, 5.0), (mid + 1.5, 5.0)],
    )


def _stacked_ask(symbol: str, *, wall: float = 100.0, mid: float = 100.0) -> Market:
    return _make_market(
        symbol,
        bids=[(mid - 0.5, 5.0), (mid - 1.5, 5.0)],
        asks=[(mid + 0.5, wall), (mid + 1.5, wall * 0.5)],
    )


class _Clock:
    """Manually-advanced clock so the sniper's time-based exits are testable."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 0.25) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
# measure_depth_imbalance
# ---------------------------------------------------------------------------


def test_depth_imbalance_returns_none_on_empty_book():
    assert measure_depth_imbalance(OrderBook()) is None


def test_depth_imbalance_sums_only_levels_in_band():
    book = OrderBook()
    # mid ≈ 100. Band of 10 bps = ±0.1 dollars. Levels at 99.5 / 100.5 are
    # outside the band and must NOT contribute.
    book.replace(
        bids=[(100.0, 50), (99.5, 999)],
        asks=[(100.0, 5), (100.5, 999)],
    )
    snap = measure_depth_imbalance(book, band_bps=10.0)
    assert snap is not None
    assert snap.bid_size == 50
    assert snap.ask_size == 5
    assert snap.ratio == pytest.approx(10.0)
    assert snap.bull is True


def test_depth_imbalance_bear_when_asks_dominate():
    book = OrderBook()
    # Band = mid * 20bps = 100 * 0.002 = ±0.2 — both bid and ask levels fit.
    book.replace(
        bids=[(99.99, 1), (99.98, 1)],
        asks=[(100.01, 30), (100.02, 30)],
    )
    snap = measure_depth_imbalance(book, band_bps=20.0)
    assert snap is not None
    assert snap.bear is True
    # bid_size = 2, ask_size = 60 → inverse_ratio = 60/2 = 30.
    assert snap.inverse_ratio == pytest.approx(30.0)


def test_depth_imbalance_handles_zero_one_side():
    book = OrderBook()
    book.replace(bids=[(99.99, 10)], asks=[(100.01, 0.0001)])
    snap = measure_depth_imbalance(book)
    assert snap is not None
    assert snap.ratio > 0
    assert math.isfinite(snap.ratio) or snap.ratio == float("inf")


# ---------------------------------------------------------------------------
# TapeInferrer
# ---------------------------------------------------------------------------


def test_tape_inferrer_counts_taker_buy_on_shrinking_ask():
    tape = TapeInferrer(window_seconds=2.0, max_step_fraction=0.95)
    book = OrderBook()
    # initial frame: priming, no events
    book.replace(bids=[(100, 5)], asks=[(101, 10)])
    buys, sells = tape.update(book, now=0.0)
    assert (buys, sells) == (0, 0)

    # ask size shrinks 10 -> 7 at the same price -> 1 inferred buy
    book.replace(bids=[(100, 5)], asks=[(101, 7)])
    buys, sells = tape.update(book, now=0.1)
    assert buys == 1
    assert sells == 0


def test_tape_inferrer_counts_taker_sell_on_shrinking_bid():
    tape = TapeInferrer(window_seconds=2.0, max_step_fraction=0.95)
    book = OrderBook()
    book.replace(bids=[(100, 10)], asks=[(101, 5)])
    tape.update(book, now=0.0)
    book.replace(bids=[(100, 4)], asks=[(101, 5)])
    buys, sells = tape.update(book, now=0.1)
    assert sells == 1
    assert buys == 0


def test_tape_inferrer_ignores_full_level_vanish_as_spoof_noise():
    """A 100% drop in a single tick is treated as cancellation noise."""
    tape = TapeInferrer(window_seconds=2.0, max_step_fraction=0.95)
    book = OrderBook()
    book.replace(bids=[(100, 5)], asks=[(101, 100)])
    tape.update(book, now=0.0)
    # ask collapses 100 -> 0 in one tick — exceeds the 95% guard.
    book.replace(bids=[(100, 5)], asks=[(101, 0.0001)])
    buys, sells = tape.update(book, now=0.1)
    assert buys == 0


def test_tape_inferrer_window_evicts_old_events():
    tape = TapeInferrer(window_seconds=1.0)
    book = OrderBook()
    book.replace(bids=[(100, 5)], asks=[(101, 10)])
    tape.update(book, now=0.0)
    book.replace(bids=[(100, 5)], asks=[(101, 7)])
    buys, _ = tape.update(book, now=0.1)
    assert buys == 1

    # 3 seconds later (>> 1s window) — the event must have been evicted.
    book.replace(bids=[(100, 5)], asks=[(101, 7)])
    buys, _ = tape.update(book, now=3.5)
    assert buys == 0


def test_tape_inferrer_ignores_price_improving_ask():
    """An ask whose price moves DOWN means new offers, not aggressive buys."""
    tape = TapeInferrer(window_seconds=2.0, max_step_fraction=0.95)
    book = OrderBook()
    book.replace(bids=[(100, 5)], asks=[(102, 10)])
    tape.update(book, now=0.0)
    book.replace(bids=[(100, 5)], asks=[(101, 5)])  # better ask price
    buys, _ = tape.update(book, now=0.1)
    assert buys == 0


# ---------------------------------------------------------------------------
# WallSnapshot
# ---------------------------------------------------------------------------


def test_wall_vanishes_inside_persist_window():
    snap = WallSnapshot(side="BID", price=100.0, size=50.0, observed_at=0.0)
    book = OrderBook()
    book.replace(bids=[(100.0, 20.0)], asks=[(101.0, 5.0)])  # 60% pull
    assert snap.vanished(book, ratio_threshold=0.5, now=0.5, persist_seconds=1.0)


def test_wall_does_not_vanish_after_persist_window():
    snap = WallSnapshot(side="BID", price=100.0, size=50.0, observed_at=0.0)
    book = OrderBook()
    book.replace(bids=[(100.0, 20.0)], asks=[(101.0, 5.0)])
    # 2s > 1s persist window — should NOT count as spoofing anymore.
    assert not snap.vanished(book, ratio_threshold=0.5, now=2.0, persist_seconds=1.0)


def test_wall_holds_when_size_unchanged():
    snap = WallSnapshot(side="BID", price=100.0, size=50.0, observed_at=0.0)
    book = OrderBook()
    book.replace(bids=[(100.0, 50.0)], asks=[(101.0, 5.0)])
    assert not snap.vanished(book, ratio_threshold=0.5, now=0.5, persist_seconds=1.0)


# ---------------------------------------------------------------------------
# OrderBookSniper — direction / firing
# ---------------------------------------------------------------------------


def test_sniper_skips_non_delta_markets():
    clock = _Clock()
    sniper = OrderBookSniper(clock=clock)
    market = _stacked_bid("BTCUSD")
    market.venue = "other"
    for _ in range(10):
        out = sniper.scan([market])
        clock.tick()
    assert out == []


def test_sniper_requires_tape_confirmation_before_firing():
    """A heavy bid wall alone is not enough — must also see taker buys."""
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        tape_min_count=3,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    # Priming tick + 5 more frames of the SAME book — no ask shrinkage,
    # so tape stays at 0. Should NOT fire.
    for _ in range(6):
        out = sniper.scan([_stacked_bid("BTCUSD")])
        clock.tick()
    assert not any(s.legs[0].side == "BUY" for s in out)


def _drive_buy_setup(
    sniper: OrderBookSniper,
    clock: _Clock,
    symbol: str = "BTCUSD",
) -> Market:
    """Drive a bid-wall + taker-buy tape until the sniper emits a BUY.

    Returns the last market snapshot fed in (which the test can mutate to
    drive exit scenarios).
    """
    # Prime: a wide-flat book so the tape inferrer has a baseline.
    market = _make_market(
        symbol,
        bids=[(99.5, 100.0), (98.5, 50.0)],
        asks=[(100.5, 30.0), (101.5, 30.0)],
    )
    sniper.scan([market])
    clock.tick()

    # Now simulate four ticks where the ask is being eaten (taker buys)
    # while the bid wall persists. Each tick shrinks the ask size by 3
    # contracts — within the 95% step guard, so each counts as a buy.
    for ask_size in (27.0, 24.0, 21.0, 18.0):
        market = _make_market(
            symbol,
            bids=[(99.5, 100.0), (98.5, 50.0)],
            asks=[(100.5, ask_size), (101.5, 30.0)],
        )
        out = sniper.scan([market])
        clock.tick()
        if out and out[0].legs[0].side == "BUY":
            return market
    raise AssertionError("expected sniper to fire BUY after tape + wall confirmation")


def test_sniper_fires_buy_when_wall_and_tape_align():
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,         # wide enough to include both wall levels
        tape_min_count=3,
        tape_window_seconds=10.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)

    # The next scan should have already produced the BUY in _drive_buy_setup.
    # Re-emit by checking internal state.
    state = sniper._state[market.id]
    assert state.position_side == "LONG"
    assert state.wall is not None
    assert state.wall.side == "BID"
    assert state.wall.size >= 99.0  # the 100-contract bid wall (within band)


def test_sniper_fires_sell_when_wall_and_tape_align():
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    symbol = "ETHUSD"
    market = _make_market(
        symbol,
        bids=[(99.5, 30.0), (98.5, 30.0)],
        asks=[(100.5, 100.0), (101.5, 50.0)],
    )
    sniper.scan([market])
    clock.tick()

    out: list = []
    # Shrink the bid wall (taker sells) over a few ticks.
    for bid_size in (27.0, 24.0, 21.0, 18.0):
        market = _make_market(
            symbol,
            bids=[(99.5, bid_size), (98.5, 30.0)],
            asks=[(100.5, 100.0), (101.5, 50.0)],
        )
        out = sniper.scan([market])
        clock.tick()
        if out and out[0].legs[0].side == "SELL":
            break
    assert out, "expected a SELL after taker-sell tape + heavy ask wall"
    assert out[0].legs[0].side == "SELL"
    assert sniper._state[symbol].position_side == "SHORT"


def test_sniper_entry_limit_is_best_bid_plus_one_tick():
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        entry_tick_offset=1,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _make_market(
        "BTCUSD",
        bids=[(99.5, 100.0)],
        asks=[(100.5, 30.0)],
        tick=0.5,
    )
    sniper.scan([market])
    clock.tick()
    out: list = []
    for ask_sz in (27.0, 24.0, 21.0, 18.0):
        market = _make_market(
            "BTCUSD",
            bids=[(99.5, 100.0)],
            asks=[(100.5, ask_sz)],
            tick=0.5,
        )
        out = sniper.scan([market])
        clock.tick()
        if out:
            break
    assert out, "expected a BUY"
    leg = out[0].legs[0]
    # best_bid + 1 tick = 99.5 + 0.5 = 100.0 (one tick below the ask).
    assert leg.limit_price == pytest.approx(100.0)


def test_sniper_does_not_stack_a_second_entry_on_same_symbol():
    """Once long, the sniper must not emit another BUY on the same symbol."""
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    # Keep feeding the favourable setup — must NOT fire again.
    for _ in range(5):
        out = sniper.scan([market])
        clock.tick()
        assert not any(s.legs[0].side == "BUY" and not s.legs[0].reduce_only for s in out)


# ---------------------------------------------------------------------------
# OrderBookSniper — exits
# ---------------------------------------------------------------------------


def test_sniper_take_profit_pct_closes_long():
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        take_profit_pct=0.0005,    # +5 bps
        stop_loss_pct=0.0003,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        spoof_persist_seconds=0.0,  # disable spoof exit for this test
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    entry_mid = sniper._state[market.id].entry_mid

    # Push mid clearly above entry * (1 + 5bps) — 50 bps move.
    target = entry_mid * 1.005
    push = _make_market(
        market.id,
        bids=[(target - 0.5, 100.0)],
        asks=[(target + 0.5, 100.0)],
        tick=0.5,
    )
    out = sniper.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "take-profit"]
    assert exits, "expected a take-profit close"
    assert exits[0].legs[0].side == "SELL"
    assert exits[0].legs[0].reduce_only is True


def test_sniper_stop_loss_pct_closes_long():
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        take_profit_pct=0.0005,
        stop_loss_pct=0.0003,    # -3 bps
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        spoof_persist_seconds=0.0,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    entry_mid = sniper._state[market.id].entry_mid

    target = entry_mid * 0.997  # 30 bps below entry
    push = _make_market(
        market.id,
        bids=[(target - 0.5, 100.0)],
        asks=[(target + 0.5, 100.0)],
        tick=0.5,
    )
    out = sniper.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "stop-loss"]
    assert exits, "expected a stop-loss close"


def test_sniper_hold_timeout_forces_market_exit():
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        max_hold_seconds=5.0,
        rearm_distance_bps=0.0,
        spoof_persist_seconds=0.0,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    # Advance well past the hold limit while the price barely moves.
    clock.tick(dt=6.0)
    out = sniper.scan([market])
    exits = [s for s in out if s.metadata.get("exit") == "hold-timeout"]
    assert exits, "expected a hold-timeout close after 5s hold limit"
    assert exits[0].legs[0].reduce_only is True


def test_sniper_spoof_exit_when_wall_pulled_inside_window():
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        max_hold_seconds=10.0,
        spoof_persist_seconds=1.0,
        spoof_vanish_ratio=0.5,
        spoof_min_wall_contracts=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    # Within 0.5s of entry the entry-side bid wall vanishes (100 -> 10).
    clock.tick(dt=0.4)
    spoofed = _make_market(
        market.id,
        bids=[(99.5, 10.0), (98.5, 5.0)],
        asks=[(100.5, 30.0), (101.5, 30.0)],
        tick=0.5,
    )
    out = sniper.scan([spoofed])
    exits = [s for s in out if s.metadata.get("exit") == "spoof-exit"]
    assert exits, "expected a spoof-exit when the entry-side wall is pulled"
    sig = exits[0]
    assert sig.metadata["wall_initial"] >= 99.0
    assert sig.metadata["wall_remaining"] <= 10.0


def test_sniper_no_spoof_exit_after_persist_window():
    """Wall vanishing AFTER the persist window must not trigger the spoof exit."""
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        max_hold_seconds=10.0,
        spoof_persist_seconds=1.0,
        spoof_vanish_ratio=0.5,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    clock.tick(dt=3.0)  # well past the 1s persist window
    spoofed = _make_market(
        market.id,
        bids=[(99.5, 10.0)],
        asks=[(100.5, 30.0)],
        tick=0.5,
    )
    out = sniper.scan([spoofed])
    spoof_exits = [s for s in out if s.metadata.get("exit") == "spoof-exit"]
    assert not spoof_exits


def test_sniper_spoof_min_wall_contracts_filter():
    """spoof_min_wall_contracts > observed wall must skip the spoof check."""
    clock = _Clock()
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        max_hold_seconds=10.0,
        spoof_persist_seconds=1.0,
        spoof_vanish_ratio=0.5,
        spoof_min_wall_contracts=10_000.0,   # absurdly high — never trips
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    clock.tick(dt=0.3)
    spoofed = _make_market(
        market.id,
        bids=[(99.5, 1.0)],
        asks=[(100.5, 30.0)],
        tick=0.5,
    )
    out = sniper.scan([spoofed])
    assert not any(s.metadata.get("exit") == "spoof-exit" for s in out)


# ---------------------------------------------------------------------------
# USD-P&L exit path (mirrors the mean-reversion scalper contract)
# ---------------------------------------------------------------------------


def _seed_position(portfolio: Portfolio, symbol: str, *, shares: float, avg_cost: float) -> None:
    key = Portfolio._key(symbol, symbol)
    pos = Position(market_id=symbol, outcome_id=symbol)
    pos.shares = shares
    pos.avg_cost = avg_cost
    portfolio.positions[key] = pos


def test_sniper_usd_take_profit_closes_long_at_target_profit():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        take_profit_usd=5.0,
        stop_loss_usd=3.0,
        max_hold_seconds=0.0,
        spoof_persist_seconds=0.0,
        rearm_distance_bps=0.0,
        portfolio=portfolio,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    _seed_position(portfolio, market.id, shares=10.0, avg_cost=100.0)

    # Bid at 100.6 → pnl = +$6 → must close.
    push = _make_market(
        market.id,
        bids=[(100.6, 100.0)],
        asks=[(101.2, 100.0)],
        tick=0.5,
    )
    out = sniper.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "take-profit"]
    assert exits
    assert exits[0].metadata["pnl_usd"] == pytest.approx(6.0, abs=1e-6)


def test_sniper_usd_no_exit_when_position_already_flat():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    sniper = OrderBookSniper(
        imbalance_ratio=3.0,
        imbalance_band_bps=200.0,
        tape_min_count=3,
        tape_window_seconds=10.0,
        take_profit_usd=5.0,
        stop_loss_usd=3.0,
        max_hold_seconds=0.0,
        spoof_persist_seconds=0.0,
        rearm_distance_bps=0.0,
        portfolio=portfolio,
        clock=clock,
    )
    market = _drive_buy_setup(sniper, clock)
    # Don't seed a portfolio position → strategy must reset state silently.
    push = _make_market(
        market.id,
        bids=[(200.0, 100.0)],
        asks=[(201.0, 100.0)],
        tick=0.5,
    )
    out = sniper.scan([push])
    assert not any(s.metadata.get("exit") for s in out)
    assert sniper._state[market.id].position_side is None
