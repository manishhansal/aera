"""Dashboard state container + FastAPI route sanity tests.

These run without spinning up a real engine — we drive the state container
directly and then poke the FastAPI app via the `TestClient`. No network and
no event loop required.
"""
from __future__ import annotations

import math

import pytest

from aera.core import Fill, Portfolio
from aera.dashboard import DashboardState, create_app
from aera.execution.executor import ExecutionResult
from aera.strategies import Leg, Signal


def make_signal(strategy: str = "delta_perp_scalper", edge: float = 0.012) -> Signal:
    return Signal(
        strategy=strategy,
        confidence=1.0,
        edge=edge,
        legs=[
            Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="BUY",
                limit_price=100.0, size_usd=200.0, leverage=50.0),
        ],
        metadata={"note": "test"},
    )


def test_state_seeds_equity_curve_with_starting_bankroll():
    state = DashboardState(Portfolio(bankroll=1.0))
    assert len(state.equity_curve) == 1
    assert math.isclose(state.equity_curve[0].bankroll, 1.0)


def test_record_signals_and_execution_round_trip():
    pf = Portfolio(bankroll=1.0)
    state = DashboardState(pf)

    sig = make_signal()
    state.record_signals([sig])
    assert len(state.signals) == 1
    assert state.signals[0].status == "pending"
    assert state.strategy_stats["delta_perp_scalper"].signals_emitted == 1

    # Apply a fill onto the portfolio, then push an ExecutionResult through.
    fill = Fill(timestamp=10.0, market_id="ETHUSD", outcome_id="ETHUSD",
                side="BUY", price=100.0, size=2.0, fee=0.0, leverage=50.0)
    pf.apply_fill(fill)
    state.record_execution(ExecutionResult(signal=sig, fills=[fill], success=True))

    assert state.strategy_stats["delta_perp_scalper"].trades_executed == 1
    # the pending row was promoted, not duplicated
    assert len(state.signals) == 1
    assert state.signals[0].status == "executed"
    # one FillEvent on the trades feed
    assert len(state.fills) == 1
    assert state.fills[0].strategy == "delta_perp_scalper"
    assert state.fills[0].side == "BUY"


def test_record_execution_rejection_marks_signal():
    state = DashboardState(Portfolio(bankroll=1.0))
    sig = make_signal()
    state.record_signals([sig])
    state.record_execution(
        ExecutionResult(signal=sig, fills=[], success=False, reason="too small")
    )
    assert state.signals[0].status == "rejected"
    assert state.signals[0].reason == "too small"
    assert state.strategy_stats["delta_perp_scalper"].trades_rejected == 1


def test_snapshot_is_json_friendly():
    state = DashboardState(Portfolio(bankroll=1.0))
    snap = state.snapshot()
    # All top-level keys are JSON-serialisable
    import json
    json.dumps(snap)
    assert snap["portfolio"]["bankroll"] == 1.0
    assert snap["engine"]["running"] is False  # no engine bound


def test_fastapi_routes_smoke():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    state = DashboardState(Portfolio(bankroll=2.0))
    app = create_app(state, push_interval_ms=10_000)

    # We don't want the equity-sampler startup event running in tests,
    # but TestClient.__enter__ triggers startup — so use it briefly.
    with TestClient(app) as client:
        r = client.get("/api/state")
        assert r.status_code == 200
        body = r.json()
        assert body["portfolio"]["bankroll"] == 2.0
        assert "strategies" in body
        assert "uptime_seconds" in body

        for path in ("/api/fills", "/api/trades", "/api/signals",
                     "/api/positions", "/api/equity", "/api/markets"):
            assert client.get(path).status_code == 200

        # control endpoints should refuse if no engine is bound
        r = client.post("/api/control/pause")
        assert r.status_code == 409


def test_equity_history_grows():
    state = DashboardState(Portfolio(bankroll=1.0))
    pf = state.portfolio
    pf.bankroll = 1.5
    state.record_equity_sample()
    pf.bankroll = 1.7
    state.record_equity_sample()
    pts = state.equity_history()
    assert len(pts) >= 3
    assert pts[-1]["bankroll"] == 1.7


# ---------------------------------------------------------------------------
# round-trip trade pairing
# ---------------------------------------------------------------------------


def _make_open_close_signals():
    open_sig = Signal(
        strategy="delta_perp_scalper", confidence=1.0, edge=0.012,
        legs=[Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="BUY",
                  limit_price=100.0, size_usd=200.0, leverage=50.0)],
        metadata={"note": "open"},
    )
    close_sig = Signal(
        strategy="delta_perp_scalper", confidence=1.0, edge=0.0,
        legs=[Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="SELL",
                  limit_price=105.0, size_usd=210.0, leverage=50.0,
                  reduce_only=True)],
        metadata={"note": "close"},
    )
    return open_sig, close_sig


def _push_fill(state, pf, sig, *, side, price, size, ts, fee=0.0):
    fill = Fill(timestamp=ts, market_id="ETHUSD", outcome_id="ETHUSD",
                side=side, price=price, size=size, fee=fee, leverage=50.0)
    pf.apply_fill(fill)
    state.record_execution(ExecutionResult(signal=sig, fills=[fill], success=True))
    return fill


def test_long_round_trip_emits_trade_event_with_open_close_pnl():
    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    open_sig, close_sig = _make_open_close_signals()
    state.record_signals([open_sig])

    _push_fill(state, pf, open_sig, side="BUY", price=100.0, size=2.0, ts=10.0)
    # mid-stream: still open, no completed trades yet
    assert len(state.trades) == 0
    assert "ETHUSD:ETHUSD" in state._open_trades

    state.record_signals([close_sig])
    _push_fill(state, pf, close_sig, side="SELL", price=105.0, size=2.0, ts=70.0)

    assert len(state.trades) == 1
    t = state.trades[0]
    assert t.side == "LONG"
    assert t.open_price == pytest.approx(100.0)
    assert t.close_price == pytest.approx(105.0)
    assert t.size == pytest.approx(2.0)
    # (105 - 100) * 2 = $10 P&L on a long, fees were zero
    assert t.pnl == pytest.approx(10.0)
    assert t.duration_seconds == pytest.approx(60.0)
    assert t.strategy == "delta_perp_scalper"
    # round-trip cleared the in-flight tracker
    assert "ETHUSD:ETHUSD" not in state._open_trades


def test_short_round_trip_pnl_is_signed_correctly():
    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    open_sig = Signal(
        strategy="delta_perp_scalper", confidence=1.0, edge=0.012,
        legs=[Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="SELL",
                  limit_price=100.0, size_usd=200.0, leverage=50.0)],
    )
    close_sig = Signal(
        strategy="delta_perp_scalper", confidence=1.0, edge=0.0,
        legs=[Leg(market_id="ETHUSD", outcome_id="ETHUSD", side="BUY",
                  limit_price=98.0, size_usd=196.0, leverage=50.0,
                  reduce_only=True)],
    )
    _push_fill(state, pf, open_sig, side="SELL", price=100.0, size=2.0, ts=1.0)
    _push_fill(state, pf, close_sig, side="BUY", price=98.0, size=2.0, ts=2.0)

    assert len(state.trades) == 1
    t = state.trades[0]
    assert t.side == "SHORT"
    assert t.open_price == pytest.approx(100.0)
    assert t.close_price == pytest.approx(98.0)
    # short: bought back $2 cheaper × 2 shares = +$4
    assert t.pnl == pytest.approx(4.0)


def test_partial_close_keeps_remainder_open():
    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    open_sig, close_sig = _make_open_close_signals()

    _push_fill(state, pf, open_sig, side="BUY", price=100.0, size=4.0, ts=1.0)
    # close only half the position
    _push_fill(state, pf, close_sig, side="SELL", price=110.0, size=1.5, ts=2.0)

    assert len(state.trades) == 1
    t = state.trades[0]
    assert t.size == pytest.approx(1.5)
    assert t.pnl == pytest.approx((110.0 - 100.0) * 1.5)  # +$15
    # remainder still open with the original entry price
    open_trade = state._open_trades["ETHUSD:ETHUSD"]
    assert open_trade.open_size == pytest.approx(2.5)
    assert open_trade.open_price == pytest.approx(100.0)


def test_scale_in_averages_entry_price():
    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    open_sig, close_sig = _make_open_close_signals()

    _push_fill(state, pf, open_sig, side="BUY", price=100.0, size=1.0, ts=1.0)
    _push_fill(state, pf, open_sig, side="BUY", price=104.0, size=1.0, ts=2.0)
    # weighted avg entry should now be $102, no trade emitted yet
    assert len(state.trades) == 0
    assert state._open_trades["ETHUSD:ETHUSD"].open_price == pytest.approx(102.0)
    assert state._open_trades["ETHUSD:ETHUSD"].open_size == pytest.approx(2.0)

    _push_fill(state, pf, close_sig, side="SELL", price=105.0, size=2.0, ts=3.0)
    t = state.trades[0]
    assert t.open_price == pytest.approx(102.0)
    assert t.close_price == pytest.approx(105.0)
    assert t.pnl == pytest.approx(6.0)  # ($105 - $102) * 2


def test_flip_emits_close_and_starts_opposite_trade():
    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    open_sig, close_sig = _make_open_close_signals()

    _push_fill(state, pf, open_sig, side="BUY", price=100.0, size=2.0, ts=1.0)
    # SELL more than we own → fully closes the long and opens a fresh short
    _push_fill(state, pf, close_sig, side="SELL", price=110.0, size=3.0, ts=2.0)

    assert len(state.trades) == 1
    t = state.trades[0]
    assert t.side == "LONG"
    assert t.size == pytest.approx(2.0)
    assert t.pnl == pytest.approx(20.0)
    # new short of size 1.0 carried over at the fill price
    open_trade = state._open_trades["ETHUSD:ETHUSD"]
    assert open_trade.side == "SHORT"
    assert open_trade.open_size == pytest.approx(1.0)
    assert open_trade.open_price == pytest.approx(110.0)


def test_fees_subtract_from_round_trip_pnl():
    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    open_sig, close_sig = _make_open_close_signals()

    _push_fill(state, pf, open_sig, side="BUY", price=100.0, size=2.0,
               ts=1.0, fee=0.30)
    _push_fill(state, pf, close_sig, side="SELL", price=105.0, size=2.0,
               ts=2.0, fee=0.32)

    t = state.trades[0]
    # gross +$10, fees $0.62 → net +$9.38
    assert t.fees == pytest.approx(0.62)
    assert t.pnl == pytest.approx(10.0 - 0.62)


def test_open_positions_surface_mark_and_unrealised_pnl():
    """Regression for the dashboard showing "REALISED $0.0000" on every
    open position (always true and therefore useless) and no unrealised
    PnL view at all.

    ``open_positions`` now exposes the latest cached mark per market
    plus an ``unrealised_pnl`` field. Mark is sourced from the live
    Market objects on every ``record_markets`` tick; until the first
    tick lands we fall back to ``avg_cost`` so uPnL reads zero rather
    than NaN.
    """
    from aera.markets import Market, Outcome
    from aera.markets.orderbook import OrderBook

    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    pf.apply_fill(Fill(timestamp=0, market_id="ETHUSD", outcome_id="ETHUSD",
                       side="BUY", price=2000.0, size=0.1, leverage=25.0))

    # No mark cached yet → uPnL must be zero, mark equals avg_cost.
    rows = state.open_positions()
    assert len(rows) == 1
    row = rows[0]
    assert row["mark"] == pytest.approx(2000.0)
    assert row["unrealised_pnl"] == pytest.approx(0.0)
    assert row["notional"] == pytest.approx(200.0)

    # Tick in a fresh mid — 2% favourable move = +$4 on 0.1 ETH.
    book = OrderBook()
    book.update_level("bid", 2039.0, 5.0)
    book.update_level("ask", 2041.0, 5.0)
    eth = Market(
        id="ETHUSD", slug="eth", question="ETH perp", category="perp",
        outcomes={"ETHUSD": Outcome(
            id="ETHUSD", label="ETHUSD",
            book=book, last_price=2040.0, volume_24h=0.0,
        )},
        metadata={},
    )
    state.record_markets({eth.id: eth})

    rows = state.open_positions()
    row = rows[0]
    assert row["mark"] == pytest.approx(2040.0)
    assert row["unrealised_pnl"] == pytest.approx(4.0, abs=1e-9)
    # Notional now reflects the new mark, not the entry price.
    assert row["notional"] == pytest.approx(204.0, abs=1e-9)


def test_open_positions_handles_high_priced_sub_share_positions():
    """A 0.003 BTC position used to display "0.00 SHARES" because the
    serialiser dropped precision. Now we surface the full float and let
    the front-end format it — verify the value isn't being lost on the
    Python side."""
    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    pf.apply_fill(Fill(timestamp=0, market_id="BTCUSD", outcome_id="BTCUSD",
                       side="BUY", price=72_963.0, size=0.003, leverage=25.0))
    rows = state.open_positions()
    assert len(rows) == 1
    assert rows[0]["shares"] == pytest.approx(0.003)
    assert rows[0]["notional"] == pytest.approx(0.003 * 72_963.0, rel=1e-6)


def test_recent_trades_and_snapshot_expose_round_trips():
    pf = Portfolio(bankroll=1000.0)
    state = DashboardState(pf)
    open_sig, close_sig = _make_open_close_signals()

    _push_fill(state, pf, open_sig, side="BUY", price=100.0, size=1.0, ts=1.0)
    _push_fill(state, pf, close_sig, side="SELL", price=110.0, size=1.0, ts=2.0)
    _push_fill(state, pf, open_sig, side="BUY", price=200.0, size=1.0, ts=3.0)
    _push_fill(state, pf, close_sig, side="SELL", price=190.0, size=1.0, ts=4.0)

    rows = state.recent_trades()
    # most recent first
    assert rows[0]["close_price"] == pytest.approx(190.0)
    assert rows[0]["pnl"] == pytest.approx(-10.0)
    assert rows[1]["pnl"] == pytest.approx(10.0)

    snap = state.snapshot()
    assert snap["trades_closed"] == 2
    assert snap["closed_trade_pnl"] == pytest.approx(0.0)
    assert snap["closed_trade_wins"] == 1
    assert snap["closed_trade_losses"] == 1
