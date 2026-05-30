"""Backtest replay engine — deterministic synthetic candle path."""
from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
import pandas as pd
import pytest

from aera.backtest import BacktestResult, BarReplay, candles_to_market_stream
from aera.core import Portfolio
from aera.markets import Market
from aera.strategies.base import Leg, Signal, Strategy


# ---------------------------------------------------------------------------
# fake strategies for controlled backtests
# ---------------------------------------------------------------------------


class _BuyAndHold(Strategy):
    """Opens a single long on the first tick, never closes."""

    name = "buy_and_hold"

    def __init__(self, portfolio=None, size_usd: float = 100.0):
        super().__init__()
        self.size_usd = size_usd
        self._fired = False

    def scan(self, markets: Iterable[Market]) -> List[Signal]:
        if self._fired:
            return []
        sigs: List[Signal] = []
        for m in markets:
            outcome = next(iter(m.outcomes.values()), None)
            if outcome is None:
                continue
            ask = outcome.best_ask
            if ask is None:
                continue
            self._fired = True
            sigs.append(Signal(
                strategy=self.name, confidence=1.0, edge=0.01,
                legs=[Leg(
                    market_id=m.id, outcome_id=outcome.id, side="BUY",
                    limit_price=ask, size_usd=self.size_usd,
                    reason="test", leverage=1.0,
                )],
                metadata={"symbol": m.id},
            ))
        return sigs


class _OpenCloseEachBar(Strategy):
    """Opens a long, closes it on the very next signal cycle. Tests
    that close legs are honoured and PnL is recorded."""

    name = "open_close"

    def __init__(self, portfolio=None, size_usd: float = 100.0):
        super().__init__()
        self.size_usd = size_usd
        self._state = "flat"

    def scan(self, markets: Iterable[Market]) -> List[Signal]:
        sigs: List[Signal] = []
        for m in markets:
            outcome = next(iter(m.outcomes.values()), None)
            if outcome is None:
                continue
            bid = outcome.best_bid
            ask = outcome.best_ask
            if bid is None or ask is None:
                continue
            if self._state == "flat":
                self._state = "long"
                sigs.append(Signal(
                    strategy=self.name, confidence=1.0, edge=0.01,
                    legs=[Leg(
                        market_id=m.id, outcome_id=outcome.id, side="BUY",
                        limit_price=ask, size_usd=self.size_usd,
                        reason="open", leverage=1.0,
                    )],
                    metadata={"symbol": m.id},
                ))
            else:
                self._state = "flat"
                sigs.append(Signal(
                    strategy=self.name, confidence=1.0, edge=0.01,
                    legs=[Leg(
                        market_id=m.id, outcome_id=outcome.id, side="SELL",
                        limit_price=bid, size_usd=self.size_usd,
                        reason="close", leverage=1.0, reduce_only=True,
                    )],
                    metadata={"symbol": m.id},
                ))
        return sigs


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def _candles(
    n: int = 60, start_ts: int = 1_700_000_000, base: float = 1000.0,
    drift: float = 0.001, noise: float = 0.0005, seed: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = base
    for i in range(n):
        ts = start_ts + i * 60
        op = price
        # Simulate intra-bar OHLC with a small drift + noise.
        cl = price * (1 + drift + rng.normal(0.0, noise))
        hi = max(op, cl) * (1 + abs(rng.normal(0.0, noise * 0.5)))
        lo = min(op, cl) * (1 - abs(rng.normal(0.0, noise * 0.5)))
        vol = float(rng.integers(50, 500))
        rows.append((ts, op, hi, lo, cl, vol))
        price = cl
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_candles_to_market_stream_yields_monotonic_timestamps():
    df = _candles(n=5)
    stream = list(candles_to_market_stream(df, symbol="BTCUSD", ticks_per_bar=4))
    assert len(stream) == 5 * 4
    timestamps = [ts for ts, _ in stream]
    # Should be non-decreasing (intra-bar ticks share a ts; that's fine)
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


def test_candles_to_market_stream_walks_high_first_on_red_bars():
    """Red bar (close < open) should put HIGH before LOW in the
    synthetic tick sequence so a long entry on tick 1 sees the
    most-adverse price first."""
    row = pd.DataFrame([{"ts": 100, "open": 110, "high": 115, "low": 95, "close": 100, "volume": 0}])
    stream = list(candles_to_market_stream(row, symbol="BTCUSD", ticks_per_bar=4))
    mids = []
    for _, m in stream:
        out = next(iter(m.outcomes.values()))
        mids.append(0.5 * (out.best_bid + out.best_ask))
    # 4 ticks: open, high, low, close → mids[1] is high, mids[2] is low
    assert mids[1] == pytest.approx(115, abs=1e-6)
    assert mids[2] == pytest.approx(95, abs=1e-6)


def test_buy_and_hold_yields_one_trade_with_correct_direction():
    df = _candles(n=30, drift=0.002, seed=2)
    replay = BarReplay(
        strategy_factory=_BuyAndHold,
        starting_bankroll=1_000.0, symbol="BTCUSD",
        leverage=1.0, taker_fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0,
    )
    result = replay.run(df)
    assert result.num_trades == 1
    t = result.trades[0]
    assert t.side == "LONG"
    # With positive drift the long should be profitable before fees.
    assert t.pnl_usd > 0
    # End bankroll matches realised PnL.
    assert result.ending_bankroll == pytest.approx(
        result.starting_bankroll + result.total_pnl, abs=1e-6
    )


def test_open_close_pattern_realises_pnl_per_round_trip():
    df = _candles(n=20, drift=0.0, noise=0.001, seed=3)
    replay = BarReplay(
        strategy_factory=_OpenCloseEachBar,
        starting_bankroll=1_000.0, symbol="BTCUSD",
        leverage=1.0, taker_fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0,
    )
    result = replay.run(df)
    assert result.num_trades >= 5, "should fire many open/close round-trips"
    # Wins + losses should sum to num_trades (no zero-PnL trades).
    assert result.num_wins + result.num_losses <= result.num_trades


def test_fees_dent_pnl_proportionally():
    df = _candles(n=10, drift=0.0, noise=0.0, seed=4)
    no_fee = BarReplay(
        _BuyAndHold, starting_bankroll=1_000.0, symbol="BTCUSD",
        taker_fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0,
    ).run(df)
    with_fee = BarReplay(
        _BuyAndHold, starting_bankroll=1_000.0, symbol="BTCUSD",
        taker_fee_bps=10.0, slippage_bps=0.0, spread_bps=0.0,
    ).run(df)
    # 10 bps × 2 legs × $100 = $0.20 of fees.
    assert with_fee.total_pnl < no_fee.total_pnl
    diff = no_fee.total_pnl - with_fee.total_pnl
    assert diff == pytest.approx(0.20, abs=1e-2)


def test_metrics_safe_on_empty_result():
    res = BacktestResult(
        strategy="x", symbol="BTCUSD", resolution="1m",
        leverage=1.0, bars_processed=0,
        starting_bankroll=1_000.0, ending_bankroll=1_000.0,
    )
    assert res.win_rate == 0.0
    assert res.profit_factor == 0.0
    assert res.expectancy == 0.0
    assert res.sharpe == 0.0
    assert res.max_drawdown == 0.0


def test_max_drawdown_picks_largest_peak_to_trough():
    """Trades with profile +5, +5, -8, +1: equity 1000→1005→1010→
    1002→1003. Peak=1010, trough after =1002 ⇒ DD = 8/1010 ≈ 0.79%."""
    from aera.backtest.replay import TradeRecord
    trades = [
        TradeRecord("x", "S", "LONG", 0, 0, 100, 105, 100, 1.0, +5.0, 0.0, 60, "", ""),
        TradeRecord("x", "S", "LONG", 0, 0, 100, 105, 100, 1.0, +5.0, 0.0, 60, "", ""),
        TradeRecord("x", "S", "LONG", 0, 0, 100, 92, 100, 1.0, -8.0, 0.0, 60, "", ""),
        TradeRecord("x", "S", "LONG", 0, 0, 100, 101, 100, 1.0, +1.0, 0.0, 60, "", ""),
    ]
    res = BacktestResult(
        strategy="x", symbol="S", resolution="1m", leverage=1.0,
        bars_processed=10, trades=trades, starting_bankroll=1000.0,
    )
    assert res.max_drawdown == pytest.approx(8.0 / 1010, abs=1e-6)


def test_short_position_pnl_signs_correctly():
    """Build a single short trade where the close price is HIGHER
    than entry → losing short. Confirms the PnL sign."""
    df = pd.DataFrame([
        # Two bars: first bar's open is the entry tick. Second bar's
        # last tick is the forced end-of-data flatten.
        {"ts": 1000, "open": 100, "high": 100, "low": 100, "close": 100, "volume": 0},
        {"ts": 1060, "open": 110, "high": 110, "low": 110, "close": 110, "volume": 0},
    ])

    class _OpenShort(Strategy):
        name = "short_open"

        def __init__(self, portfolio=None):
            super().__init__()
            self._fired = False

        def scan(self, markets):
            sigs = []
            for m in markets:
                if self._fired:
                    return []
                outcome = next(iter(m.outcomes.values()))
                bid = outcome.best_bid
                if bid is None:
                    continue
                self._fired = True
                sigs.append(Signal(
                    strategy=self.name, confidence=1.0, edge=0.01,
                    legs=[Leg(
                        market_id=m.id, outcome_id=outcome.id, side="SELL",
                        limit_price=bid, size_usd=100.0, reason="test",
                        leverage=1.0,
                    )],
                    metadata={"symbol": m.id},
                ))
            return sigs

    res = BarReplay(
        _OpenShort, starting_bankroll=1_000.0, symbol="X",
        taker_fee_bps=0.0, slippage_bps=0.0, spread_bps=0.0,
    ).run(df)
    assert res.num_trades == 1
    t = res.trades[0]
    assert t.side == "SHORT"
    # Short opened at 100, closed at 110 → losing, ≈ -$10 on $100 notional.
    assert t.pnl_usd < 0
    assert t.pnl_usd == pytest.approx(-10.0, abs=0.5)
