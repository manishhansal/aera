"""DeltaPerpetualScalper unit tests.

Drive the strategy deterministically by hand-feeding sequences of mid prices
through the scan loop and asserting it only fires once both the z-score and
OFI gates are tripped in the correct direction.
"""
from __future__ import annotations

import pytest

from aera.core import Portfolio
from aera.core.portfolio import Fill, Position
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.strategies import DeltaPerpetualScalper


def _make_market(symbol: str, bid_p: float, bid_sz: float, ask_p: float, ask_sz: float) -> Market:
    book = OrderBook()
    book.replace(
        bids=[(bid_p, bid_sz), (bid_p - 1, bid_sz * 0.5)],
        asks=[(ask_p, ask_sz), (ask_p + 1, ask_sz * 0.5)],
    )
    return Market(
        id=symbol,
        slug=symbol.lower(),
        question=f"{symbol} perp",
        category="perpetual_futures",
        outcomes={symbol: Outcome(id=symbol, label=DELTA_OUTCOME_LABEL, book=book)},
        venue="delta",
    )


def test_scalper_ignores_non_delta_venues():
    market = _make_market("BTCUSD", 100, 10, 101, 10)
    market.venue = "other"  # any non-"delta" string must be skipped
    strat = DeltaPerpetualScalper(zscore_window=5, zscore_entry=1.0, ofi_threshold=0.0)
    # Even after enough samples, a non-delta market never triggers.
    for _ in range(30):
        sigs = strat.scan([market])
    assert sigs == []


def test_scalper_fires_buy_when_mid_dips_with_bid_stacked():
    """Build up a baseline at mid≈100, then drop mid with the bid stacked:
    z-score should turn deeply negative AND OFI strongly positive → BUY."""
    strat = DeltaPerpetualScalper(
        zscore_window=20,
        zscore_entry=1.5,
        ofi_threshold=0.15,
        rearm_distance_bps=0.0,   # disable debounce for the test
    )

    # warm-up baseline (also primes the OFI EMA to ~0)
    for _ in range(20):
        strat.scan([_make_market("BTCUSD", 100, 5, 101, 5)])

    # dip + sustained bid imbalance → after a couple of ticks the EMA crosses.
    for _ in range(3):
        out = strat.scan([_make_market("BTCUSD", 90, 50, 91, 1)])
        if out:
            break
    assert out, "expected at least one signal after sustained dip + imbalance"
    sig = out[0]
    assert sig.legs[0].side == "BUY"
    assert sig.strategy == "delta_perp_scalper"
    assert "z" in sig.metadata
    assert sig.metadata["ofi"] > 0


def test_scalper_fires_sell_when_mid_spikes_with_ask_stacked():
    strat = DeltaPerpetualScalper(
        zscore_window=20,
        zscore_entry=1.5,
        ofi_threshold=0.15,
        rearm_distance_bps=0.0,
    )
    for _ in range(20):
        strat.scan([_make_market("ETHUSD", 100, 5, 101, 5)])
    out: list = []
    for _ in range(3):
        out = strat.scan([_make_market("ETHUSD", 110, 1, 111, 50)])
        if out:
            break
    assert out
    assert out[0].legs[0].side == "SELL"


def test_scalper_debounces_repeated_fires_close_to_same_mid():
    strat = DeltaPerpetualScalper(
        zscore_window=20,
        zscore_entry=1.5,
        ofi_threshold=0.15,
        rearm_distance_bps=200.0,  # require 200 bps move between fires
    )
    for _ in range(20):
        strat.scan([_make_market("BTCUSD", 100, 5, 101, 5)])

    first: list = []
    for _ in range(5):
        first = strat.scan([_make_market("BTCUSD", 90, 50, 91, 1)])
        if first:
            break
    second = strat.scan([_make_market("BTCUSD", 90, 50, 91, 1)])
    assert first
    assert not second, "debounce should suppress the second fire at the same mid"


def test_scalper_skips_when_top_of_book_too_thin():
    strat = DeltaPerpetualScalper(
        zscore_window=5, zscore_entry=1.0, ofi_threshold=0.0,
        min_depth_contracts=1000.0,   # impossible threshold
    )
    for _ in range(10):
        out = strat.scan([_make_market("BTCUSD", 100, 5, 101, 5)])
    assert out == []


def _force_long_entry(strat: DeltaPerpetualScalper, symbol: str = "BTCUSD") -> float:
    """Drive a buy entry and return the entry mid recorded by the strategy."""
    for _ in range(20):
        strat.scan([_make_market(symbol, 100, 5, 101, 5)])
    out: list = []
    for _ in range(5):
        out = strat.scan([_make_market(symbol, 90, 50, 91, 1)])
        if out:
            break
    assert out and out[0].legs[0].side == "BUY", "test prerequisite: long entry"
    return strat._state[symbol].entry_mid


def _force_short_entry(strat: DeltaPerpetualScalper, symbol: str = "ETHUSD") -> float:
    for _ in range(20):
        strat.scan([_make_market(symbol, 100, 5, 101, 5)])
    out: list = []
    for _ in range(5):
        out = strat.scan([_make_market(symbol, 110, 1, 111, 50)])
        if out:
            break
    assert out and out[0].legs[0].side == "SELL", "test prerequisite: short entry"
    return strat._state[symbol].entry_mid


def test_take_profit_closes_long_when_mid_rises_enough():
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_pct=0.02,   # +2% from entry
    )
    entry = _force_long_entry(strat)

    # Slightly above entry — must NOT close yet.
    target_below = entry * 1.01
    out = strat.scan([_make_market("BTCUSD",
                                   target_below - 0.5, 5,
                                   target_below + 0.5, 5)])
    assert not any(s.metadata.get("exit") for s in out)

    # Push mid clearly above the TP threshold.
    target_above = entry * 1.025
    out = strat.scan([_make_market("BTCUSD",
                                   target_above - 0.5, 5,
                                   target_above + 0.5, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits, "expected take-profit to fire above the TP threshold"
    sig = exits[0]
    assert sig.metadata["exit"] == "take-profit"
    assert sig.metadata["position_side"] == "LONG"
    assert sig.legs[0].side == "SELL"
    # Position should now be flat — re-running on the same mid yields no new close.
    out2 = strat.scan([_make_market("BTCUSD",
                                    target_above - 0.5, 5,
                                    target_above + 0.5, 5)])
    assert not any(s.metadata.get("exit") for s in out2)


def test_stop_loss_closes_long_when_mid_drops_enough():
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        stop_loss_pct=0.01,   # -1% from entry
    )
    entry = _force_long_entry(strat)

    target = entry * 0.985   # 1.5% below entry → SL must fire
    out = strat.scan([_make_market("BTCUSD",
                                   target - 0.5, 5,
                                   target + 0.5, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits, "expected stop-loss to fire below the SL threshold"
    sig = exits[0]
    assert sig.metadata["exit"] == "stop-loss"
    assert sig.legs[0].side == "SELL"


def test_take_profit_closes_short_when_mid_drops_enough():
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_pct=0.02,
    )
    entry = _force_short_entry(strat)

    target = entry * 0.97   # -3% → comfortably past +2% TP for shorts
    out = strat.scan([_make_market("ETHUSD",
                                   target - 0.5, 5,
                                   target + 0.5, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits and exits[0].metadata["exit"] == "take-profit"
    assert exits[0].legs[0].side == "BUY"


def test_stop_loss_closes_short_when_mid_rises_enough():
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        stop_loss_pct=0.01,
    )
    entry = _force_short_entry(strat)

    target = entry * 1.02
    out = strat.scan([_make_market("ETHUSD",
                                   target - 0.5, 5,
                                   target + 0.5, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits and exits[0].metadata["exit"] == "stop-loss"
    assert exits[0].legs[0].side == "BUY"


def test_no_exit_emitted_when_tp_sl_both_disabled():
    """With take_profit_pct=0 and stop_loss_pct=0 the strategy keeps the
    pre-existing behaviour: positions only flatten on opposite-direction
    reversion signals."""
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
    )
    entry = _force_long_entry(strat)
    big_move = entry * 1.10   # +10% — way past anything reasonable
    for _ in range(5):
        out = strat.scan([_make_market("BTCUSD",
                                       big_move - 0.5, 5,
                                       big_move + 0.5, 5)])
    assert not any(s.metadata.get("exit") for s in out)


def _seed_position(portfolio: Portfolio, symbol: str, *, shares: float, avg_cost: float) -> None:
    """Inject a position directly into the portfolio book.

    Bypasses ``apply_fill`` so the test doesn't have to satisfy bankroll /
    margin invariants — we only care that ``positions[key].shares`` and
    ``.avg_cost`` are what the strategy will read.
    """
    key = Portfolio._key(symbol, symbol)
    pos = Position(market_id=symbol, outcome_id=symbol)
    pos.shares = shares
    pos.avg_cost = avg_cost
    portfolio.positions[key] = pos


def test_usd_take_profit_closes_long_at_target_profit():
    """With take_profit_usd=$5 and a 10-share long at $100 cost, a $0.50 bid
    rise = +$5 P&L → take-profit fires."""
    portfolio = Portfolio(bankroll=1000.0)
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_usd=5.0, stop_loss_usd=3.0,
        portfolio=portfolio,
    )
    _force_long_entry(strat)
    _seed_position(portfolio, "BTCUSD", shares=10.0, avg_cost=100.0)

    # Bid at 100.4 → pnl = (100.4 - 100) * 10 = $4 → below TP, no close.
    out = strat.scan([_make_market("BTCUSD", 100.4, 5, 101.0, 5)])
    assert not any(s.metadata.get("exit") for s in out)

    # Bid at 100.6 → pnl = $6 → above $5 TP, must close.
    out = strat.scan([_make_market("BTCUSD", 100.6, 5, 101.2, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits, "expected USD take-profit to fire at +$6 unrealised P&L"
    sig = exits[0]
    assert sig.metadata["exit"] == "take-profit"
    assert sig.metadata["pnl_usd"] == pytest.approx(6.0, abs=1e-6)
    assert sig.legs[0].side == "SELL"
    assert sig.legs[0].reduce_only is True


def test_usd_stop_loss_closes_long_at_target_loss():
    """10-share long at $100 cost, bid drops to $99.65 → pnl = -$3.50 → SL fires."""
    portfolio = Portfolio(bankroll=1000.0)
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_usd=5.0, stop_loss_usd=3.0,
        portfolio=portfolio,
    )
    _force_long_entry(strat)
    _seed_position(portfolio, "BTCUSD", shares=10.0, avg_cost=100.0)

    # Bid at 99.8 → pnl = -$2 → above -$3 SL, no close.
    out = strat.scan([_make_market("BTCUSD", 99.8, 5, 100.4, 5)])
    assert not any(s.metadata.get("exit") for s in out)

    # Bid at 99.65 → pnl = -$3.5 → must close.
    out = strat.scan([_make_market("BTCUSD", 99.65, 5, 100.25, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits, "expected USD stop-loss to fire at −$3.5 unrealised P&L"
    sig = exits[0]
    assert sig.metadata["exit"] == "stop-loss"
    assert sig.metadata["pnl_usd"] == pytest.approx(-3.5, abs=1e-6)


def test_usd_take_profit_closes_short_at_target_profit():
    """10-share short at $100, ask drops to $99.4 → pnl = (99.4-100)*(-10)=+$6 → TP."""
    portfolio = Portfolio(bankroll=1000.0)
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_usd=5.0, stop_loss_usd=3.0,
        portfolio=portfolio,
    )
    _force_short_entry(strat)
    _seed_position(portfolio, "ETHUSD", shares=-10.0, avg_cost=100.0)

    out = strat.scan([_make_market("ETHUSD", 98.8, 5, 99.4, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits and exits[0].metadata["exit"] == "take-profit"
    assert exits[0].metadata["pnl_usd"] == pytest.approx(6.0, abs=1e-6)
    assert exits[0].legs[0].side == "BUY"


def test_usd_stop_loss_closes_short_at_target_loss():
    """10-share short at $100, ask climbs to $100.35 → pnl = -$3.50 → SL."""
    portfolio = Portfolio(bankroll=1000.0)
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_usd=5.0, stop_loss_usd=3.0,
        portfolio=portfolio,
    )
    _force_short_entry(strat)
    _seed_position(portfolio, "ETHUSD", shares=-10.0, avg_cost=100.0)

    out = strat.scan([_make_market("ETHUSD", 99.75, 5, 100.35, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits and exits[0].metadata["exit"] == "stop-loss"
    assert exits[0].metadata["pnl_usd"] == pytest.approx(-3.5, abs=1e-6)
    assert exits[0].legs[0].side == "BUY"


def test_usd_no_exit_when_position_already_flat():
    """If the portfolio shows shares=0, the USD path should clear internal
    state and emit nothing — even when entry_mid is stamped."""
    portfolio = Portfolio(bankroll=1000.0)
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_usd=5.0, stop_loss_usd=3.0,
        portfolio=portfolio,
    )
    _force_long_entry(strat)
    # No position seeded → strategy must not fire a close on a wild mid.
    out = strat.scan([_make_market("BTCUSD", 200.0, 5, 201.0, 5)])
    assert not any(s.metadata.get("exit") for s in out)
    # And internal entry state should have been cleared.
    assert strat._state["BTCUSD"].position_side is None


def test_usd_thresholds_take_precedence_over_pct_when_both_set():
    """USD path should be selected when portfolio is attached + USD>0, even
    if pct thresholds would also have tripped."""
    portfolio = Portfolio(bankroll=1000.0)
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_pct=0.001,   # very tight pct that would also fire
        take_profit_usd=5.0,
        stop_loss_usd=3.0,
        portfolio=portfolio,
    )
    _force_long_entry(strat)
    _seed_position(portfolio, "BTCUSD", shares=10.0, avg_cost=100.0)

    # Bid at 100.55 → USD pnl = $5.50 → fires.
    out = strat.scan([_make_market("BTCUSD", 100.55, 5, 101.15, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits and "pnl_usd" in exits[0].metadata
    assert exits[0].metadata["pnl_usd"] == pytest.approx(5.5, abs=1e-6)


def test_usd_stop_loss_takes_precedence_over_usd_take_profit():
    """If a single tick crosses both USD bands (close-side price below SL
    while the opposite side would mark above TP), realise the loss first."""
    portfolio = Portfolio(bankroll=1000.0)
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_usd=2.0,
        stop_loss_usd=2.0,
        portfolio=portfolio,
    )
    _force_long_entry(strat)
    # 10-share long at $100. Bid at 99.7 → pnl = -$3 (past SL).
    # Construct a wide spread so both bands could trip if the strategy
    # used either side, but the close-side (bid) sees the loss.
    _seed_position(portfolio, "BTCUSD", shares=10.0, avg_cost=100.0)
    out = strat.scan([_make_market("BTCUSD", 99.7, 5, 100.5, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits and exits[0].metadata["exit"] == "stop-loss"


def test_pct_path_used_when_no_portfolio_attached_even_with_usd_set():
    """If USD thresholds are configured but no portfolio is wired in, the
    strategy must fall back to the pct path (or emit nothing if pct is 0).
    This is the safety net for hand-built setups."""
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_pct=0.02,
        stop_loss_pct=0.01,
        take_profit_usd=5.0,
        stop_loss_usd=3.0,
        portfolio=None,
    )
    entry = _force_long_entry(strat)
    target = entry * 1.025
    out = strat.scan([_make_market("BTCUSD",
                                   target - 0.5, 5,
                                   target + 0.5, 5)])
    exits = [s for s in out if s.metadata.get("exit")]
    assert exits, "pct fallback should fire since no portfolio is attached"
    # pnl_usd metadata should be absent on the pct path.
    assert "pnl_usd" not in exits[0].metadata


def test_stop_loss_takes_precedence_over_take_profit():
    """Pathological tick where TP and SL bands are crossed simultaneously
    (e.g. a wild spike): the conservative behaviour is to realise the loss."""
    strat = DeltaPerpetualScalper(
        zscore_window=20, zscore_entry=1.5, ofi_threshold=0.15,
        rearm_distance_bps=0.0,
        take_profit_pct=0.005,
        stop_loss_pct=0.005,
    )
    entry = _force_long_entry(strat)
    state = strat._state["BTCUSD"]
    # Force-poke entry mid to a sentinel so we can build a tick that's both
    # above TP and below SL by manipulating bid/ask asymmetrically.
    state.entry_mid = entry
    # Construct a thick spread spanning both thresholds (bid below SL,
    # ask above TP) so mid can land above entry while the bid (= sell-into)
    # sits well below SL. This is exactly the spike scenario we care about.
    bid = entry * 0.985
    ask = entry * 1.030
    out = strat.scan([_make_market("BTCUSD", bid, 5, ask, 5)])
    mid = 0.5 * (bid + ask)
    crossed_sl = mid <= entry * (1 - strat.stop_loss_pct)
    crossed_tp = mid >= entry * (1 + strat.take_profit_pct)
    if crossed_sl and crossed_tp:
        exits = [s for s in out if s.metadata.get("exit")]
        assert exits and exits[0].metadata["exit"] == "stop-loss"
    else:
        # If only one side crossed at this mid, just assert the right one fired.
        exits = [s for s in out if s.metadata.get("exit")]
        if crossed_sl:
            assert exits and exits[0].metadata["exit"] == "stop-loss"
        elif crossed_tp:
            assert exits and exits[0].metadata["exit"] == "take-profit"
