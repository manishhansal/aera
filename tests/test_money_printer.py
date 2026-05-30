"""MoneyPrinter strategy — gates, ATR exits, ML hooks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aera.core import Portfolio
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.strategies import MoneyPrinter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_market(symbol: str, price: float, ts: int = 1_700_000_000) -> Market:
    half = price * 5e-5
    book = OrderBook()
    book.replace(bids=[(price - half, 100.0)], asks=[(price + half, 100.0)])
    return Market(
        id=symbol, slug=symbol.lower(), question="t", category="perpetual_futures",
        outcomes={symbol: Outcome(id=symbol, label=DELTA_OUTCOME_LABEL, book=book)},
        venue="delta", last_update=float(ts),
        metadata={"leverage": 25.0, "contract_value": 1.0},
    )


def _warm_up_with_random_walk(strat, symbol, n_bars=120, start_ts=1_700_000_000, seed=1):
    """Stream enough synthetic ticks for the strategy's feature window
    to fill. ``bar_seconds`` defaults to 60s so we walk forward by 60s
    per bar (across multiple ticks)."""
    rng = np.random.default_rng(seed)
    price = 1000.0
    ts = start_ts
    for i in range(n_bars):
        for _ in range(strat.bar_seconds // 10 + 1):
            price *= 1 + rng.normal(0.0, 0.001)
            market = _make_market(symbol, price, ts=ts)
            strat.scan([market])
            ts += 10


# ---------------------------------------------------------------------------
# basic plumbing
# ---------------------------------------------------------------------------


def test_money_printer_loads_without_artefacts(tmp_path):
    strat = MoneyPrinter(
        hour_map_path=str(tmp_path / "missing_hour_map.json"),
        model_path=str(tmp_path / "missing_model.joblib"),
    )
    # No artefacts → empty hour maps, no classifier (lazy-loaded).
    assert strat._hour_maps == {}
    assert strat._classifier_or_none() is None


def test_money_printer_returns_no_signals_before_warmup(tmp_path):
    strat = MoneyPrinter(
        hour_map_path=str(tmp_path / "x.json"),
        model_path=str(tmp_path / "y.joblib"),
    )
    market = _make_market("BTCUSD", 1000.0)
    assert strat.scan([market]) == []


# ---------------------------------------------------------------------------
# ATR-band gate
# ---------------------------------------------------------------------------


def test_money_printer_skips_when_atr_below_min(tmp_path):
    """Constant-price warmup → ATR ≈ 0 → entry skipped."""
    strat = MoneyPrinter(
        hour_map_path=str(tmp_path / "x.json"),
        model_path=str(tmp_path / "y.joblib"),
        min_atr_pct=0.0001,    # 1 bp — easily missed by zero-vol path
        bar_seconds=10,
    )
    ts = 1_700_000_000
    for i in range(80):
        market = _make_market("BTCUSD", 1000.0, ts=ts)
        strat.scan([market])
        ts += 60
    # All bars at exactly 1000 → ATR ≈ 0 → no fire
    market = _make_market("BTCUSD", 1000.0, ts=ts)
    assert strat.scan([market]) == []


def test_money_printer_skips_when_atr_above_max(tmp_path):
    """Massive volatility — ATR breaches the upper cap → entry skipped."""
    strat = MoneyPrinter(
        hour_map_path=str(tmp_path / "x.json"),
        model_path=str(tmp_path / "y.joblib"),
        max_atr_pct=0.001,     # 10 bps — easily breached by the wild walk below
        bar_seconds=10,
    )
    rng = np.random.default_rng(42)
    price = 1000.0
    ts = 1_700_000_000
    for i in range(80):
        # 5% per-bar noise — huge ATR
        price = price * (1 + rng.normal(0.0, 0.05))
        market = _make_market("BTCUSD", abs(price), ts=ts)
        strat.scan([market])
        ts += 60
    market = _make_market("BTCUSD", abs(price), ts=ts)
    assert strat.scan([market]) == []


# ---------------------------------------------------------------------------
# hour-of-day gate
# ---------------------------------------------------------------------------


def test_money_printer_blocks_excluded_hour(tmp_path):
    """Write an hour map that says only hour 0 is profitable. Feed
    the strategy a tick at hour 5 → no fire."""
    from aera.backtest.analysis import HourMap, write_hour_maps

    m = HourMap(strategy="money_printer", symbol="BTCUSD")
    # 50 wins at hour 0 → hour 0 is allowed.
    m.pnl[0] = 50.0
    m.count[0] = 10
    m.wins[0] = 10
    map_path = tmp_path / "maps.json"
    write_hour_maps({("money_printer", "BTCUSD"): m}, map_path)

    strat = MoneyPrinter(
        hour_map_path=str(map_path),
        model_path=str(tmp_path / "y.joblib"),
        bar_seconds=10,
    )
    # Hour 5 UTC = ts that lands at 5h past midnight.
    base_ts = 1_700_000_000  # this happens to be ~midnight on some date
    from datetime import datetime, timezone
    h = datetime.fromtimestamp(base_ts, tz=timezone.utc).hour
    target_offset_hours = (5 - h) % 24
    ts5 = base_ts + target_offset_hours * 3600 + 60
    rng = np.random.default_rng(7)
    price = 1000.0
    for i in range(80):
        ts = ts5 + i * 60
        price *= 1 + rng.normal(0.0, 0.0015)
        market = _make_market("BTCUSD", abs(price), ts=ts)
        strat.scan([market])
    # Final tick is also hour 5 → blocked
    final_market = _make_market("BTCUSD", abs(price), ts=ts5 + 80 * 60)
    sigs = strat.scan([final_market])
    # Hour gate vetoes → no entry. (Close signals would only fire if we'd been in.)
    assert all(any(getattr(l, "reduce_only", False) for l in s.legs) for s in sigs)


# ---------------------------------------------------------------------------
# ATR-tuned exits
# ---------------------------------------------------------------------------


def test_money_printer_exits_on_tp(tmp_path):
    """Manually install an open LONG position; move price up by 2×ATR
    → TP fires."""
    strat = MoneyPrinter(
        hour_map_path=str(tmp_path / "x.json"),
        model_path=str(tmp_path / "y.joblib"),
        bar_seconds=10,
        tp_atr_mult=1.0, sl_atr_mult=1.0,
    )
    st = strat._state_for("BTCUSD")
    st.position_side = "LONG"
    st.entry_mid = 1000.0
    st.entry_size_usd = 100.0
    st.entry_atr = 5.0  # $5
    st.entry_time = 1_700_000_000.0
    market = _make_market("BTCUSD", 1010.0, ts=1_700_000_010)  # +$10 move
    sigs = strat.scan([market])
    assert sigs, "TP at +1×ATR ($5) should fire when move = $10"
    sig = sigs[0]
    assert all(getattr(l, "reduce_only", False) for l in sig.legs)
    assert sig.legs[0].side == "SELL"


def test_money_printer_exits_on_sl(tmp_path):
    strat = MoneyPrinter(
        hour_map_path=str(tmp_path / "x.json"),
        model_path=str(tmp_path / "y.joblib"),
        bar_seconds=10,
        tp_atr_mult=2.0, sl_atr_mult=1.0,
    )
    st = strat._state_for("BTCUSD")
    st.position_side = "LONG"
    st.entry_mid = 1000.0
    st.entry_size_usd = 100.0
    st.entry_atr = 5.0
    st.entry_time = 1_700_000_000.0
    market = _make_market("BTCUSD", 990.0, ts=1_700_000_010)  # -$10 move
    sigs = strat.scan([market])
    assert sigs, "SL at -1×ATR ($5) should fire when move = -$10"
    assert sigs[0].legs[0].side == "SELL"


def test_money_printer_exits_on_hold_timeout(tmp_path):
    strat = MoneyPrinter(
        hour_map_path=str(tmp_path / "x.json"),
        model_path=str(tmp_path / "y.joblib"),
        max_hold_seconds=60.0,
    )
    st = strat._state_for("BTCUSD")
    st.position_side = "LONG"
    st.entry_mid = 1000.0
    st.entry_size_usd = 100.0
    st.entry_atr = 10.0
    st.entry_time = 1_700_000_000.0
    # 200 seconds later, mid hasn't moved — but the timeout has fired.
    market = _make_market("BTCUSD", 1000.0, ts=1_700_000_200)
    sigs = strat.scan([market])
    assert sigs
    assert "hold timeout" in sigs[0].legs[0].reason.lower()


# ---------------------------------------------------------------------------
# phantom-position guard (inherited from base Strategy)
# ---------------------------------------------------------------------------


def test_money_printer_clears_phantom_position_when_portfolio_flat(tmp_path):
    pf = Portfolio(bankroll=100.0)
    strat = MoneyPrinter(
        hour_map_path=str(tmp_path / "x.json"),
        model_path=str(tmp_path / "y.joblib"),
        portfolio=pf,
    )
    st = strat._state_for("BTCUSD")
    st.position_side = "LONG"
    st.entry_mid = 1000.0
    st.entry_size_usd = 100.0
    st.entry_atr = 5.0
    st.entry_time = 1_700_000_000.0
    # Portfolio has no position → strategy's internal state should reset
    # on the next scan via sync_position_state.
    market = _make_market("BTCUSD", 1000.0)
    strat.scan([market])
    assert st.position_side is None
    assert st.entry_mid == 0.0
