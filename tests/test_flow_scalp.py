"""FlowScalp (Tape Reading Momentum) + TradeTape helpers.

Drive both layers deterministically: fake clock, synthetic books, direct
trade injection. Covers tape avg-size math, whale detection, every entry
gate, and every exit path (hard SL, trailing stop, hard TP, hold-timeout,
USD-P&L).
"""
from __future__ import annotations

import pytest

from aera.core import Portfolio
from aera.core.portfolio import Position
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.signals.trade_tape import TradeTape
from aera.strategies import FlowScalp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_market(
    symbol: str,
    bid_p: float,
    bid_sz: float,
    ask_p: float,
    ask_sz: float,
    *,
    tick: float = 0.5,
    leverage: float = 10.0,
) -> Market:
    book = OrderBook()
    book.replace(bids=[(bid_p, bid_sz)], asks=[(ask_p, ask_sz)])
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


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 0.1) -> None:
        self.t += dt


def _seed_avg(tape: TradeTape, *, n: int = 100, size: float = 10.0, t0: float = 100.0):
    """Prime the tape's avg-size baseline with N small trades on both sides."""
    for i in range(n):
        side = "BUY" if i % 2 == 0 else "SELL"
        tape.record(price=100.0, size=size, side=side, now=t0 + i * 0.01)


def _make_strategy(*, clock: _Clock, **overrides) -> FlowScalp:
    """Build a FlowScalp wired with sane defaults for tests.

    Defaults disable the rearm debouncer (so tests can fire repeatedly
    without juggling mid offsets) and use a generous tape buffer.
    Anything in ``overrides`` wins.
    """
    base = dict(
        whale_multiple=5.0,
        confirm_multiple=2.0,
        confirm_count=1,
        confirm_window_seconds=3.0,
        avg_window=50,
        notional_usd=1000.0,
        take_profit_pct=0.0008,
        stop_loss_pct=0.0004,
        trailing_stop_pct=0.0002,
        max_hold_seconds=60.0,
        rearm_distance_bps=0.0,
        leverage_override=5.0,
        auto_infer_from_book=False,   # tests inject trades directly
        clock=clock,
    )
    base.update(overrides)
    return FlowScalp(**base)


# ---------------------------------------------------------------------------
# TradeTape
# ---------------------------------------------------------------------------


def test_tape_record_appends_and_caps_buffer():
    tape = TradeTape(max_trades=5)
    for i in range(8):
        tape.record(price=100.0 + i, size=1.0, side="BUY", now=float(i))
    assert tape.total_count == 5
    assert tape.trades[0].price == 103.0   # first three evicted


def test_tape_rejects_zero_and_invalid_side():
    tape = TradeTape()
    assert tape.record(price=100, size=0, side="BUY") is None
    assert tape.record(price=100, size=1, side="HOLD") is None
    assert tape.record(price=0, size=1, side="BUY") is None


def test_avg_size_over_rolling_window():
    tape = TradeTape(avg_window=10)
    for i in range(20):
        tape.record(price=100.0, size=float(i + 1), side="BUY", now=float(i))
    # avg of the last 10 sizes (11..20)
    assert tape.avg_size() == pytest.approx(15.5)
    assert tape.avg_size(n=5) == pytest.approx(18.0)   # last 5: 16..20


def test_avg_size_empty_tape_returns_none():
    assert TradeTape().avg_size() is None


def test_latest_whale_detects_oversize_print_in_window():
    tape = TradeTape(avg_window=20)
    _seed_avg(tape, n=20, size=10.0)
    # whale = 60 contracts at t=200 (6× avg)
    tape.record(price=101.0, size=60.0, side="BUY", now=200.0)
    whale = tape.latest_whale(multiple=5.0, lookback_seconds=10.0, now=205.0)
    assert whale is not None
    assert whale.size == 60.0
    assert whale.side == "BUY"


def test_latest_whale_returns_none_outside_lookback():
    tape = TradeTape(avg_window=20)
    _seed_avg(tape, n=20, size=10.0)
    tape.record(price=101.0, size=60.0, side="BUY", now=200.0)
    # query far in the future — whale is outside the 1s window
    assert tape.latest_whale(multiple=5.0, lookback_seconds=1.0, now=300.0) is None


def test_latest_whale_filters_by_side():
    tape = TradeTape(avg_window=20)
    _seed_avg(tape, n=20, size=10.0)
    tape.record(price=99.0, size=70.0, side="SELL", now=200.0)
    assert tape.latest_whale(
        multiple=5.0, lookback_seconds=10.0, side="BUY", now=205.0
    ) is None
    assert tape.latest_whale(
        multiple=5.0, lookback_seconds=10.0, side="SELL", now=205.0
    ) is not None


def test_count_aggressive_since_filters_by_side_and_threshold():
    tape = TradeTape(avg_window=20)
    _seed_avg(tape, n=20, size=10.0)
    tape.record(price=101.0, size=60.0, side="BUY", now=200.0)   # the whale (6×)
    tape.record(price=101.0, size=25.0, side="BUY", now=200.5)   # confirm (2.5×)
    tape.record(price=101.0, size=12.0, side="BUY", now=201.0)   # too small
    tape.record(price=101.0, size=30.0, side="SELL", now=200.7)  # wrong side

    confirms = tape.count_aggressive_since(
        side="BUY", multiple=2.0, since_ts=200.0, now=205.0,
    )
    assert confirms == 1


def test_infer_from_book_uptick_creates_taker_buy():
    tape = TradeTape()
    book = OrderBook()
    book.replace(bids=[(100.0, 10)], asks=[(101.0, 30)])
    tape.infer_from_book(book, now=0.0)
    # ask price holds but size shrinks → BUY of 23
    book.replace(bids=[(100.0, 12)], asks=[(101.0, 7)])
    trades = tape.infer_from_book(book, now=1.0)
    assert len(trades) == 1
    assert trades[0].side == "BUY"
    assert trades[0].size == 23.0


def test_infer_from_book_downtick_creates_taker_sell():
    tape = TradeTape()
    book = OrderBook()
    book.replace(bids=[(100.0, 20)], asks=[(101.0, 10)])
    tape.infer_from_book(book, now=0.0)
    # bid price drops → prior level fully cleared → SELL of 20
    book.replace(bids=[(99.5, 5)], asks=[(100.5, 10)])
    trades = tape.infer_from_book(book, now=1.0)
    assert len(trades) == 1
    assert trades[0].side == "SELL"
    assert trades[0].size == 20.0


def test_infer_from_book_records_trade_on_size_change_at_same_price():
    """Ask size shrinking at the same price is an aggressive buy event,
    even though mid hasn't moved (someone took 5 from the 10-contract
    offer; 5 are still resting at the same level)."""
    tape = TradeTape()
    book = OrderBook()
    book.replace(bids=[(100, 10)], asks=[(101, 10)])
    tape.infer_from_book(book, now=0.0)
    book.replace(bids=[(100, 20)], asks=[(101, 5)])
    trades = tape.infer_from_book(book, now=1.0)
    # Ask shrank 10 → 5 at the same price: 5-contract aggressive BUY.
    # Bid grew → not an aggressive sell.
    assert len(trades) == 1
    assert trades[0].side == "BUY"
    assert trades[0].size == 5.0


def test_infer_from_book_drops_cancellation_noise():
    """A single-tick collapse > 95% of prior size looks like a pull, not a trade."""
    tape = TradeTape(inference_max_step_fraction=0.95)
    book = OrderBook()
    book.replace(bids=[(100, 10)], asks=[(101, 100)])
    tape.infer_from_book(book, now=0.0)
    # mid up + 99% of ask size vanishes — treated as cancel-noise
    book.replace(bids=[(100.5, 10)], asks=[(101, 1)])
    trades = tape.infer_from_book(book, now=1.0)
    assert trades == []


# ---------------------------------------------------------------------------
# FlowScalp — entry gating
# ---------------------------------------------------------------------------


def test_strategy_skips_non_delta_markets():
    clock = _Clock()
    strat = _make_strategy(clock=clock)
    m = _make_market("BTCUSD", 99.5, 10, 100.5, 10)
    m.venue = "other"
    for _ in range(5):
        out = strat.scan([m])
    assert out == []
    assert "BTCUSD" not in strat._state


def test_strategy_does_not_fire_without_whale():
    clock = _Clock()
    strat = _make_strategy(clock=clock, avg_window=20)
    m = _make_market("BTCUSD", 99.5, 10, 100.5, 10)
    # Prime the strategy state, then seed tape with vanilla trades only.
    strat.scan([m])
    st = strat._state["BTCUSD"]
    _seed_avg(st.tape, n=20, size=10.0, t0=clock.t)
    clock.tick(1.0)
    out = strat.scan([m])
    assert out == []
    assert st.position_side is None


def test_strategy_fires_long_on_whale_buy_plus_confirmation():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, avg_window=20)
    m = _make_market("BTCUSD", 99.5, 10, 100.5, 10)
    strat.scan([m])
    st = strat._state["BTCUSD"]
    _seed_avg(st.tape, n=20, size=10.0, t0=clock.t)
    clock.tick(1.0)

    # Whale + confirmation — both BUY-side, both within window.
    st.tape.record(price=100.5, size=60.0, side="BUY", now=clock.t)        # 6× avg
    clock.tick(0.5)
    st.tape.record(price=100.5, size=25.0, side="BUY", now=clock.t)        # 2.5× avg
    clock.tick(0.1)

    out = strat.scan([m])
    assert any(s.legs[0].side == "BUY" for s in out)
    assert st.position_side == "LONG"
    assert st.entry_mid > 0
    fired = [s for s in out if not s.legs[0].reduce_only]
    assert fired and fired[0].legs[0].leverage == 5.0


def test_strategy_fires_short_on_whale_sell_plus_confirmation():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, avg_window=20)
    m = _make_market("ETHUSD", 99.5, 10, 100.5, 10)
    strat.scan([m])
    st = strat._state["ETHUSD"]
    _seed_avg(st.tape, n=20, size=10.0, t0=clock.t)
    clock.tick(1.0)

    st.tape.record(price=99.5, size=70.0, side="SELL", now=clock.t)
    clock.tick(0.5)
    st.tape.record(price=99.5, size=30.0, side="SELL", now=clock.t)
    clock.tick(0.1)

    out = strat.scan([m])
    assert any(s.legs[0].side == "SELL" for s in out)
    assert st.position_side == "SHORT"


def test_strategy_does_not_fire_without_confirmation():
    """Whale prints alone aren't enough — the spec requires a follow-up."""
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, avg_window=20, confirm_count=1)
    m = _make_market("BTCUSD", 99.5, 10, 100.5, 10)
    strat.scan([m])
    st = strat._state["BTCUSD"]
    _seed_avg(st.tape, n=20, size=10.0, t0=clock.t)
    clock.tick(1.0)

    # Single whale, no follow-up.
    st.tape.record(price=100.5, size=60.0, side="BUY", now=clock.t)
    clock.tick(0.5)
    out = strat.scan([m])
    assert st.position_side is None
    assert out == []


def test_strategy_expires_pending_whale_after_window():
    """A whale must be confirmed within `confirm_window_seconds`."""
    clock = _Clock(t=1000.0)
    strat = _make_strategy(
        clock=clock, avg_window=20, confirm_window_seconds=3.0,
    )
    m = _make_market("BTCUSD", 99.5, 10, 100.5, 10)
    strat.scan([m])
    st = strat._state["BTCUSD"]
    _seed_avg(st.tape, n=20, size=10.0, t0=clock.t)
    clock.tick(1.0)

    st.tape.record(price=100.5, size=60.0, side="BUY", now=clock.t)
    strat.scan([m])
    assert st.pending is not None

    # Jump past the confirmation window. Even a confirmation now
    # arrives too late — the whale has expired.
    clock.tick(10.0)
    st.tape.record(price=100.5, size=25.0, side="BUY", now=clock.t)
    out = strat.scan([m])
    assert st.position_side is None
    assert st.pending is None
    assert out == []


def test_strategy_ignores_opposite_direction_confirmation():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, avg_window=20)
    m = _make_market("BTCUSD", 99.5, 10, 100.5, 10)
    strat.scan([m])
    st = strat._state["BTCUSD"]
    _seed_avg(st.tape, n=20, size=10.0, t0=clock.t)
    clock.tick(1.0)

    st.tape.record(price=100.5, size=60.0, side="BUY", now=clock.t)
    clock.tick(0.5)
    # SELL-side confirmation doesn't count for a BUY whale
    st.tape.record(price=100.5, size=30.0, side="SELL", now=clock.t)
    out = strat.scan([m])
    assert st.position_side is None
    assert out == []


def test_strategy_does_not_stack_entries():
    """Once positioned, fresh whale prints don't open a second leg."""
    clock = _Clock(t=1000.0)
    strat = _make_strategy(
        clock=clock, avg_window=20, max_hold_seconds=0.0,    # no time exit
        take_profit_pct=0.0, stop_loss_pct=0.0,              # no price exits
        trailing_stop_pct=0.0,
    )
    m = _make_market("BTCUSD", 99.5, 10, 100.5, 10)
    strat.scan([m])
    st = strat._state["BTCUSD"]
    _seed_avg(st.tape, n=20, size=10.0, t0=clock.t)
    clock.tick(1.0)

    st.tape.record(price=100.5, size=60.0, side="BUY", now=clock.t)
    clock.tick(0.5)
    st.tape.record(price=100.5, size=25.0, side="BUY", now=clock.t)
    strat.scan([m])
    assert st.position_side == "LONG"

    # Slam more whales — no second entry, no close (all exits disabled).
    for _ in range(3):
        clock.tick(0.3)
        st.tape.record(price=100.5, size=80.0, side="BUY", now=clock.t)
        st.tape.record(price=100.5, size=40.0, side="BUY", now=clock.t)
        out = strat.scan([m])
        non_reduce = [s for s in out if not s.legs[0].reduce_only]
        assert non_reduce == []


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------


def _open_long(strat: FlowScalp, clock: _Clock, m: Market) -> None:
    """Helper: warm tape, fire a long entry."""
    strat.scan([m])
    st = strat._state[m.id]
    _seed_avg(st.tape, n=20, size=10.0, t0=clock.t)
    clock.tick(1.0)
    st.tape.record(price=100.5, size=60.0, side="BUY", now=clock.t)
    clock.tick(0.2)
    st.tape.record(price=100.5, size=25.0, side="BUY", now=clock.t)
    clock.tick(0.1)
    strat.scan([m])
    assert st.position_side == "LONG"


def test_take_profit_closes_long_at_pct_target():
    clock = _Clock()
    strat = _make_strategy(
        clock=clock,
        avg_window=20,
        trailing_stop_pct=0.0,    # disable trail for a clean TP test
        max_hold_seconds=0.0,
    )
    m = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    _open_long(strat, clock, m)
    entry = strat._state["BTCUSD"].entry_mid

    target = entry * (1.0 + 0.005)   # 50 bps above, well past 8 bps TP
    push = _make_market("BTCUSD", target - 0.5, 100, target + 0.5, 100)
    out = strat.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "take-profit"]
    assert exits, "expected a take-profit close"
    assert exits[0].legs[0].side == "SELL"
    assert exits[0].legs[0].reduce_only is True


def test_stop_loss_closes_long_at_pct_target():
    clock = _Clock()
    strat = _make_strategy(
        clock=clock,
        avg_window=20,
        trailing_stop_pct=0.0,
        max_hold_seconds=0.0,
    )
    m = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    _open_long(strat, clock, m)
    entry = strat._state["BTCUSD"].entry_mid

    target = entry * (1.0 - 0.005)
    push = _make_market("BTCUSD", target - 0.5, 100, target + 0.5, 100)
    out = strat.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "stop-loss"]
    assert exits, "expected a stop-loss close"


def test_trailing_stop_arms_only_when_in_profit():
    """A small move in profit then giveback past trail level → close."""
    clock = _Clock()
    strat = _make_strategy(
        clock=clock,
        avg_window=20,
        take_profit_pct=0.005,   # raise TP so the trail trips first
        stop_loss_pct=0.005,     # raise SL so the trail trips first
        trailing_stop_pct=0.0002,    # 2 bps trail
        max_hold_seconds=0.0,
    )
    m = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    _open_long(strat, clock, m)
    entry = strat._state["BTCUSD"].entry_mid

    # Push 30 bps above entry — trail arms; best_mid stamped.
    peak = entry * 1.003
    push_up = _make_market("BTCUSD", peak - 0.5, 100, peak + 0.5, 100)
    out = strat.scan([push_up])
    assert all(s.metadata.get("exit") is None for s in out)
    assert strat._state["BTCUSD"].best_mid == pytest.approx(peak)

    # Pull back to 1 bp above entry — that's > 20 bps off the peak, well
    # past a 2 bps trail. Close.
    pull = entry * 1.0001
    push_down = _make_market("BTCUSD", pull - 0.5, 100, pull + 0.5, 100)
    out = strat.scan([push_down])
    exits = [s for s in out if s.metadata.get("exit") == "trailing-stop"]
    assert exits, "expected a trailing-stop close"
    assert exits[0].legs[0].side == "SELL"
    assert exits[0].legs[0].reduce_only is True


def test_trailing_stop_does_not_arm_when_underwater():
    """Underwater (best_mid <= entry_mid) the trail is silent — hard SL governs."""
    clock = _Clock()
    strat = _make_strategy(
        clock=clock,
        avg_window=20,
        take_profit_pct=0.005,
        stop_loss_pct=0.005,
        trailing_stop_pct=0.0002,
        max_hold_seconds=0.0,
    )
    m = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    _open_long(strat, clock, m)
    entry = strat._state["BTCUSD"].entry_mid

    # Drift just below entry — no profit yet, trail dormant.
    pull = entry * 0.9999
    push = _make_market("BTCUSD", pull - 0.5, 100, pull + 0.5, 100)
    out = strat.scan([push])
    assert all(s.metadata.get("exit") is None for s in out)
    assert strat._state["BTCUSD"].position_side == "LONG"


def test_hold_timeout_forces_exit():
    clock = _Clock()
    strat = _make_strategy(
        clock=clock,
        avg_window=20,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        trailing_stop_pct=0.0,
        max_hold_seconds=60.0,
    )
    m = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    _open_long(strat, clock, m)
    clock.tick(dt=65.0)
    out = strat.scan([m])
    exits = [s for s in out if s.metadata.get("exit") == "hold-timeout"]
    assert exits, "expected a hold-timeout close"
    assert exits[0].metadata["hold_seconds"] > 60.0


# ---------------------------------------------------------------------------
# USD-PnL exit (mirrors the other strategies' contract)
# ---------------------------------------------------------------------------


def _seed_position(portfolio: Portfolio, symbol: str, *, shares: float, avg_cost: float):
    key = Portfolio._key(symbol, symbol)
    pos = Position(market_id=symbol, outcome_id=symbol)
    pos.shares = shares
    pos.avg_cost = avg_cost
    portfolio.positions[key] = pos


def test_usd_take_profit_closes_long():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    strat = _make_strategy(
        clock=clock,
        avg_window=20,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        trailing_stop_pct=0.0,
        take_profit_usd=5.0,
        stop_loss_usd=3.0,
        max_hold_seconds=0.0,
        portfolio=portfolio,
    )
    m = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    _open_long(strat, clock, m)
    _seed_position(portfolio, "BTCUSD", shares=10.0, avg_cost=100.0)

    push = _make_market("BTCUSD", 100.6, 100, 101.2, 100)
    out = strat.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "take-profit"]
    assert exits, "expected a USD take-profit close"
    assert exits[0].metadata["pnl_usd"] == pytest.approx(6.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Inference path (auto_infer_from_book=True)
# ---------------------------------------------------------------------------


def test_strategy_can_infer_taker_trades_from_book():
    """Without explicit record_trade, book deltas should still populate the tape."""
    clock = _Clock()
    strat = _make_strategy(
        clock=clock,
        avg_window=20,
        auto_infer_from_book=True,
    )
    # Prime with a baseline book.
    m0 = _make_market("BTCUSD", 99.5, 10, 100.5, 30)
    strat.scan([m0])
    clock.tick()

    # Walk the ask size DOWN at the same price (each step a taker buy
    # of the diff). Bid stays flat, so the tape only records BUYs.
    sizes = [25, 20, 15, 10, 5]
    for sz in sizes:
        m = _make_market("BTCUSD", 99.5, 10, 100.5, sz)
        strat.scan([m])
        clock.tick()
    st = strat._state["BTCUSD"]
    # 5 inferred BUY trades (one per ask-shrink step).
    assert st.tape.total_count == 5
    assert all(t.side == "BUY" for t in st.tape.trades)
    # Sizes are the per-step diffs: 30-25, 25-20, 20-15, 15-10, 10-5 = 5 each.
    assert [t.size for t in st.tape.trades] == [5.0] * 5
