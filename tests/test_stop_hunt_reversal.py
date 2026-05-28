"""StopHuntReversal + BarStream helpers.

Drive both layers deterministically: fake clock, synthetic order books,
manual control over every bar the strategy will see. Covers bar
rotation, taker-volume inference, swing-pivot detection, sweep entry
gates (wick depth, body ratio, recovery speed, volume confirmation,
delta confirmation, swing pivot presence, rearm debounce), and every
exit path (wick-anchored SL, partial TP1, final TP2, hold-timeout,
USD-P&L exits, position-flat reset).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pytest

from aera.core import Portfolio
from aera.core.portfolio import Position
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.signals.bar_stream import Bar, BarStream
from aera.strategies import StopHuntReversal


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
    leverage: float = 8.0,
) -> Market:
    """Single-level book Delta market — enough for bar-stream tests."""
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


def _set_book(market: Market, bid_p: float, bid_sz: float, ask_p: float, ask_sz: float) -> None:
    outcome = next(iter(market.outcomes.values()))
    outcome.book.replace(bids=[(bid_p, bid_sz)], asks=[(ask_p, ask_sz)])


class _Clock:
    """Manually-advanced clock used by both the stream and the strategy."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 0.1) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
# BarStream — rotation + volume inference + swing pivots
# ---------------------------------------------------------------------------


def test_bar_stream_cold_start_opens_first_bar():
    stream = BarStream(bar_seconds=1.0)
    book = OrderBook()
    book.replace(bids=[(100.0, 10.0)], asks=[(101.0, 10.0)])
    closed = stream.update(book, now=1000.25)
    assert closed is None
    cur = stream.current_bar
    assert cur is not None
    assert cur.start == 1000.0
    assert cur.end == 1001.0
    # OHLC all equal to the very first mid (100.5)
    assert cur.open == pytest.approx(100.5)
    assert cur.high == pytest.approx(100.5)
    assert cur.low == pytest.approx(100.5)
    assert cur.close == pytest.approx(100.5)
    assert stream.closed_bars == []


def test_bar_stream_same_bucket_updates_ohlc():
    stream = BarStream(bar_seconds=1.0)
    book = OrderBook()
    book.replace(bids=[(100.0, 10.0)], asks=[(101.0, 10.0)])
    stream.update(book, now=1000.10)
    # mid moves up — high should follow
    book.replace(bids=[(101.0, 10.0)], asks=[(102.0, 10.0)])
    stream.update(book, now=1000.30)
    # mid moves back down — low should track
    book.replace(bids=[(99.0, 10.0)], asks=[(100.0, 10.0)])
    stream.update(book, now=1000.50)
    cur = stream.current_bar
    assert cur is not None
    assert cur.open == pytest.approx(100.5)
    assert cur.high == pytest.approx(101.5)
    assert cur.low == pytest.approx(99.5)
    assert cur.close == pytest.approx(99.5)
    assert stream.closed_bars == []


def test_bar_stream_rotates_on_bucket_boundary():
    stream = BarStream(bar_seconds=1.0)
    book = OrderBook()
    book.replace(bids=[(100.0, 10.0)], asks=[(101.0, 10.0)])
    stream.update(book, now=1000.10)
    # bump into the next bucket
    book.replace(bids=[(102.0, 10.0)], asks=[(103.0, 10.0)])
    closed = stream.update(book, now=1001.10)
    assert closed is not None
    assert closed.start == 1000.0
    assert closed.end == 1001.0
    assert closed.open == pytest.approx(100.5)
    assert closed.close == pytest.approx(100.5)  # last mid before rotation
    cur = stream.current_bar
    assert cur is not None
    assert cur.start == 1001.0
    assert cur.open == pytest.approx(102.5)


def test_bar_stream_infers_taker_buy_volume():
    """Ask size shrinks at the same price → taker buy of the diff."""
    stream = BarStream(bar_seconds=1.0)
    book = OrderBook()
    book.replace(bids=[(100.0, 10.0)], asks=[(101.0, 50.0)])
    stream.update(book, now=1000.10)
    # ask price holds, size drops by 30 — taker buy of 30
    book.replace(bids=[(100.0, 10.0)], asks=[(101.0, 20.0)])
    stream.update(book, now=1000.30)
    cur = stream.current_bar
    assert cur is not None
    assert cur.buy_volume == pytest.approx(30.0)
    assert cur.sell_volume == 0.0
    assert cur.delta == pytest.approx(30.0)


def test_bar_stream_infers_taker_sell_volume():
    """Bid size shrinks at the same price → taker sell of the diff."""
    stream = BarStream(bar_seconds=1.0)
    book = OrderBook()
    book.replace(bids=[(100.0, 50.0)], asks=[(101.0, 10.0)])
    stream.update(book, now=1000.10)
    book.replace(bids=[(100.0, 18.0)], asks=[(101.0, 10.0)])
    stream.update(book, now=1000.30)
    cur = stream.current_bar
    assert cur is not None
    assert cur.sell_volume == pytest.approx(32.0)
    assert cur.buy_volume == 0.0
    assert cur.delta == pytest.approx(-32.0)


def test_bar_stream_swing_pivots_basic():
    """5-bar fractal: a clear high in the middle of identical low bars."""
    stream = BarStream(bar_seconds=1.0)
    # Build 7 closed bars with one obvious pivot high at index 3.
    sequence: List[Tuple[float, float]] = [
        (100.0, 100.5),  # bar 0: mid stays in 100..101 range
        (100.5, 101.0),  # bar 1
        (101.0, 101.5),  # bar 2
        (102.0, 103.0),  # bar 3 — high spikes up
        (101.0, 101.5),  # bar 4 — fallback
        (100.5, 101.0),  # bar 5
        (100.0, 100.5),  # bar 6
    ]
    t = 1000.0
    for low, high in sequence:
        # start each bar at the LOW
        book = OrderBook()
        book.replace(bids=[(low - 0.25, 5.0)], asks=[(low + 0.25, 5.0)])
        stream.update(book, now=t)
        # push the mid up to the HIGH inside the same bar
        book.replace(bids=[(high - 0.25, 5.0)], asks=[(high + 0.25, 5.0)])
        stream.update(book, now=t + 0.5)
        t += 1.0
    # rotate the last bar off by stepping into a new bucket
    book = OrderBook()
    book.replace(bids=[(99.0, 5.0)], asks=[(100.0, 5.0)])
    stream.update(book, now=t + 1.0)

    highs, lows = stream.swing_pivots(pivot_strength=2, lookback_bars=60)
    # The pivot high at index 3 (mid of bid 102.75 / ask 103.25 = 103.0)
    # should be detected.
    assert len(highs) >= 1
    top = max(b.high for b in highs)
    assert top == pytest.approx(103.0)


def test_bar_stream_recent_swing_lows_returns_newest_first():
    stream = BarStream(bar_seconds=1.0)
    # Build sequence with two clear swing lows
    sequence: List[Tuple[float, float]] = [
        (100.0, 101.0),
        (99.0, 100.0),
        (98.0, 99.0),
        (95.0, 96.0),   # pivot low #1 at index 3
        (98.0, 99.0),
        (99.0, 100.0),
        (98.0, 99.0),
        (94.0, 95.0),   # pivot low #2 at index 7
        (97.0, 98.0),
        (98.0, 99.0),
        (99.0, 100.0),
    ]
    t = 1000.0
    for low, high in sequence:
        book = OrderBook()
        book.replace(bids=[(high - 0.25, 5.0)], asks=[(high + 0.25, 5.0)])
        stream.update(book, now=t)
        book.replace(bids=[(low - 0.25, 5.0)], asks=[(low + 0.25, 5.0)])
        stream.update(book, now=t + 0.5)
        t += 1.0
    # rotate the last bar off
    book = OrderBook()
    book.replace(bids=[(99.0, 5.0)], asks=[(100.0, 5.0)])
    stream.update(book, now=t + 1.0)

    lows = stream.recent_swing_lows(pivot_strength=2, lookback_bars=60, max_count=3)
    assert len(lows) >= 2
    # newest first — pivot at index 7 (low_mid = 94.0) should come
    # before pivot at index 3 (low_mid = 95.0).
    assert lows[0] < lows[1]
    assert lows[0] == pytest.approx(94.0)
    assert lows[1] == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# helpers for driving the strategy
# ---------------------------------------------------------------------------


def _make_strategy(*, clock: _Clock, **overrides) -> StopHuntReversal:
    """Build a StopHuntReversal wired with sane test defaults.

    Volume + delta gates are disabled by default so tests can focus on
    a single dimension at a time; opt-in via overrides.
    """
    base = dict(
        bar_seconds=1.0,
        max_bars=200,
        swing_lookback_bars=30,
        swing_pivot_strength=2,
        swing_count=3,
        wick_size_pct=0.0015,
        body_ratio_max=0.30,
        recovery_seconds=3.0,
        volume_multiple=0.0,                  # disabled by default
        volume_lookback_bars=10,
        require_delta_confirmation=False,     # disabled by default
        delta_flip_threshold=0.0,
        take_profit_pct=0.0020,
        tp1_pct=0.0010,
        tp1_fraction=0.60,
        stop_extra_pct=0.0008,
        stop_loss_pct=0.0,
        max_hold_seconds=60.0,
        leverage_override=5.0,
        notional_usd=1000.0,
        rearm_distance_bps=0.0,               # disable debounce in tests
        clock=clock,
    )
    base.update(overrides)
    return StopHuntReversal(**base)


def _seed_swing_low(
    strategy: StopHuntReversal,
    clock: _Clock,
    *,
    symbol: str = "BTCUSD",
    pivot_low: float = 100.0,
    pivot_high: float = 105.0,
    bid_size: float = 50.0,
    ask_size: float = 50.0,
) -> Market:
    """Drive enough bars to create at least one confirmed swing-low pivot.

    Sequence: a clear "valley" in the middle of a flat-ish series so
    the 5-bar fractal at index 3 qualifies as a swing low. Returns the
    Market object (always the same, mutated in-place per bar).
    """
    market = _make_market(
        symbol, pivot_high - 0.5, bid_size, pivot_high + 0.5, ask_size
    )
    # 7 bars: 0..2 flat-ish high, bar 3 dips to pivot_low, bars 4..6 recover
    bars = [
        (pivot_high - 0.5, pivot_high + 0.5),     # bar 0
        (pivot_high - 0.5, pivot_high + 0.5),     # bar 1
        (pivot_high - 0.5, pivot_high + 0.5),     # bar 2
        (pivot_low - 0.5, pivot_low + 0.5),       # bar 3 — pivot low here
        (pivot_high - 0.5, pivot_high + 0.5),     # bar 4
        (pivot_high - 0.5, pivot_high + 0.5),     # bar 5
        (pivot_high - 0.5, pivot_high + 0.5),     # bar 6
    ]
    for bid_p, ask_p in bars:
        _set_book(market, bid_p, bid_size, ask_p, ask_size)
        strategy.scan([market])
        clock.tick(1.05)            # cross a bar boundary
    return market


def _seed_swing_high(
    strategy: StopHuntReversal,
    clock: _Clock,
    *,
    symbol: str = "BTCUSD",
    pivot_low: float = 100.0,
    pivot_high: float = 110.0,
    bid_size: float = 50.0,
    ask_size: float = 50.0,
) -> Market:
    """Mirror of :func:`_seed_swing_low`: build a pivot HIGH at index 3."""
    market = _make_market(
        symbol, pivot_low - 0.5, bid_size, pivot_low + 0.5, ask_size
    )
    bars = [
        (pivot_low - 0.5, pivot_low + 0.5),
        (pivot_low - 0.5, pivot_low + 0.5),
        (pivot_low - 0.5, pivot_low + 0.5),
        (pivot_high - 0.5, pivot_high + 0.5),     # pivot high at index 3
        (pivot_low - 0.5, pivot_low + 0.5),
        (pivot_low - 0.5, pivot_low + 0.5),
        (pivot_low - 0.5, pivot_low + 0.5),
    ]
    for bid_p, ask_p in bars:
        _set_book(market, bid_p, bid_size, ask_p, ask_size)
        strategy.scan([market])
        clock.tick(1.05)
    return market


def _drive_wick_bar(
    strategy: StopHuntReversal,
    clock: _Clock,
    market: Market,
    *,
    open_mid: float,
    wick_low: Optional[float] = None,
    wick_high: Optional[float] = None,
    close_mid: float,
    bid_size: float = 50.0,
    ask_size: float = 50.0,
) -> None:
    """Drive a wick-and-snap-back bar followed by a boundary roll.

    Sets the book to ``open_mid``, then to the wick extreme, then to
    the close, and finally rolls into the next bar (so the wick bar
    closes and the strategy sees it as a candidate).
    """
    # open
    _set_book(market, open_mid - 0.5, bid_size, open_mid + 0.5, ask_size)
    strategy.scan([market])
    clock.tick(0.1)
    # wick
    if wick_low is not None:
        _set_book(market, wick_low - 0.5, bid_size, wick_low + 0.5, ask_size)
        strategy.scan([market])
        clock.tick(0.1)
    if wick_high is not None:
        _set_book(market, wick_high - 0.5, bid_size, wick_high + 0.5, ask_size)
        strategy.scan([market])
        clock.tick(0.1)
    # close
    _set_book(market, close_mid - 0.5, bid_size, close_mid + 0.5, ask_size)
    strategy.scan([market])
    clock.tick(0.6)    # take us through the bar boundary on the NEXT scan
    # roll into the new bar so the wick bar closes
    _set_book(market, close_mid - 0.5, bid_size, close_mid + 0.5, ask_size)


# ---------------------------------------------------------------------------
# strategy: bullish sweep entry
# ---------------------------------------------------------------------------


def test_bullish_sweep_fires_long():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, wick_size_pct=0.0015, body_ratio_max=0.30)

    market = _seed_swing_low(strat, clock, pivot_low=100.0, pivot_high=105.0)

    # Now drive a wick bar that dips below 100 by > 0.15%:
    # 100 × 0.0015 = 0.15 → wick must be ≤ 99.85.
    # We open near the pivot (≈ 100), spike down to 99.5, close back at 100.4.
    # body = |100.4 − 100| = 0.4 ; range = 100.4 − 99.5 = 0.9 → body_ratio ≈ 0.44 :( 
    # Need body_ratio < 0.30 — push the wick deeper to 99.0:
    # body = 0.4 ; range = 100.4 − 99.0 = 1.4 → body_ratio ≈ 0.286 ✓
    _drive_wick_bar(
        strat, clock, market,
        open_mid=100.0,
        wick_low=99.0,
        close_mid=100.4,
    )
    signals = strat.scan([market])

    assert len(signals) == 1
    sig = signals[0]
    assert sig.strategy == "stop_hunt_reversal"
    assert len(sig.legs) == 1
    leg = sig.legs[0]
    assert leg.side == "BUY"
    assert leg.reduce_only is False
    assert leg.size_usd == pytest.approx(1000.0)
    meta = sig.metadata
    assert meta["side"] == "BUY"
    assert meta["swept_level"] == pytest.approx(100.0)
    assert meta["stop_price"] < 99.0
    # tp targets are positive against entry
    assert meta["tp1_target"] > meta["mid"]
    assert meta["tp2_target"] > meta["tp1_target"]


def test_bearish_sweep_fires_short():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock)

    market = _seed_swing_high(strat, clock, pivot_low=100.0, pivot_high=110.0)

    # Pivot high ≈ 110. Wick depth needs ≥ 110 × 0.0015 = 0.165, so wick
    # high ≥ 110.165. Open 110, spike to 111.0, close at 109.6.
    # body = 0.4 ; range = 111.0 − 109.6 = 1.4 → body_ratio ≈ 0.286 ✓
    _drive_wick_bar(
        strat, clock, market,
        open_mid=110.0,
        wick_high=111.0,
        close_mid=109.6,
    )
    signals = strat.scan([market])

    assert len(signals) == 1
    sig = signals[0]
    leg = sig.legs[0]
    assert leg.side == "SELL"
    meta = sig.metadata
    assert meta["side"] == "SELL"
    assert meta["swept_level"] == pytest.approx(110.0)
    assert meta["stop_price"] > 111.0
    # for shorts, tp targets are BELOW entry
    assert meta["tp1_target"] < meta["mid"]
    assert meta["tp2_target"] < meta["tp1_target"]


# ---------------------------------------------------------------------------
# strategy: entry gates
# ---------------------------------------------------------------------------


def test_no_signal_when_wick_doesnt_pierce_far_enough():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, wick_size_pct=0.0015)
    market = _seed_swing_low(strat, clock, pivot_low=100.0)
    # Wick only dips to 99.95 — only 5 bps below 100, fails 0.15% gate.
    _drive_wick_bar(
        strat, clock, market,
        open_mid=100.0,
        wick_low=99.95,
        close_mid=100.4,
    )
    signals = strat.scan([market])
    assert signals == []


def test_no_signal_when_close_fails_recovery():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock)
    market = _seed_swing_low(strat, clock, pivot_low=100.0)
    # Wick dips deep enough, but close is BELOW the level — sweep
    # without snap-back, just a directional break.
    _drive_wick_bar(
        strat, clock, market,
        open_mid=100.0,
        wick_low=99.0,
        close_mid=99.6,
    )
    signals = strat.scan([market])
    assert signals == []


def test_no_signal_when_body_too_big():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, body_ratio_max=0.30)
    market = _seed_swing_low(strat, clock, pivot_low=100.0)
    # Body = 0.9, range = 1.0 → ratio = 0.9. Too big.
    _drive_wick_bar(
        strat, clock, market,
        open_mid=100.0,
        wick_low=99.5,
        close_mid=100.4,
    )
    # Pre-check: body 0.4 / range 0.9 = 0.44, still fails 0.30.
    signals = strat.scan([market])
    assert signals == []


def test_volume_confirmation_required_when_enabled():
    clock = _Clock(t=1000.0)
    # Require 5× the avg per-bar volume on the wick.
    strat = _make_strategy(
        clock=clock,
        volume_multiple=5.0,
        volume_lookback_bars=5,
    )
    market = _seed_swing_low(strat, clock, pivot_low=100.0)
    # Wick prints but with default bid/ask sizes the inferred volume
    # won't suddenly 5× the baseline (sizes haven't changed). Setup
    # must be vetoed.
    _drive_wick_bar(
        strat, clock, market,
        open_mid=100.0,
        wick_low=99.0,
        close_mid=100.4,
    )
    signals = strat.scan([market])
    assert signals == []


def test_delta_confirmation_required_for_bullish():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(
        clock=clock,
        require_delta_confirmation=True,
        delta_flip_threshold=0.0001,
    )
    market = _seed_swing_low(strat, clock, pivot_low=100.0)

    # Drive a wick where the inferred delta turns deeply NEGATIVE on
    # the wick close (the bid side gets eaten heavily during the
    # close). That should veto the bullish sweep, which wants
    # positive snap-back flow.
    _set_book(market, 99.5, 50.0, 100.5, 50.0)
    strat.scan([market])
    clock.tick(0.1)
    # wick down
    _set_book(market, 98.5, 50.0, 99.5, 50.0)
    strat.scan([market])
    clock.tick(0.1)
    # close: bid drops in size massively (= big taker sell) — net delta red
    _set_book(market, 99.5, 5.0, 100.5, 50.0)
    strat.scan([market])
    clock.tick(0.6)
    # roll the bar
    _set_book(market, 99.9, 50.0, 100.9, 50.0)
    signals = strat.scan([market])
    # Either no signal (delta veto) — at minimum the strategy must NOT
    # produce a BUY when delta is red.
    for sig in signals:
        for leg in sig.legs:
            assert leg.side != "BUY"


def test_rearm_debounce_blocks_back_to_back_fires():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, rearm_distance_bps=50.0)  # very strict
    market = _seed_swing_low(strat, clock, pivot_low=100.0)
    _drive_wick_bar(
        strat, clock, market,
        open_mid=100.0,
        wick_low=99.0,
        close_mid=100.4,
    )
    first = strat.scan([market])
    assert len(first) == 1

    # Force the strategy back to flat WITHOUT the proper exit pipeline
    # so we can test the rearm path on a fresh bar that satisfies all
    # sweep gates but lands within rearm_distance_bps of last firing.
    st = strat._state["BTCUSD"]
    StopHuntReversal._reset_position(st)
    # last_signal_mid was stamped at the firing mid (~100.4); a fresh
    # wick at the same mid range is within 5 bps of it → debounce.

    # Drive ANOTHER sweep at essentially the same mid.
    _drive_wick_bar(
        strat, clock, market,
        open_mid=100.0,
        wick_low=99.0,
        close_mid=100.4,
    )
    second = strat.scan([market])
    assert second == []


# ---------------------------------------------------------------------------
# strategy: exits
# ---------------------------------------------------------------------------


def _open_long_position(strat: StopHuntReversal, clock: _Clock) -> Market:
    """Convenience: drive a clean bullish sweep, return the market."""
    market = _seed_swing_low(strat, clock, pivot_low=100.0)
    _drive_wick_bar(
        strat, clock, market,
        open_mid=100.0,
        wick_low=99.0,
        close_mid=100.4,
    )
    sigs = strat.scan([market])
    assert len(sigs) == 1 and sigs[0].legs[0].side == "BUY"
    return market


def test_tp1_partial_close_then_tp2_flatten():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(
        clock=clock,
        tp1_pct=0.001,        # +0.1%
        tp1_fraction=0.6,
        take_profit_pct=0.002,
    )
    market = _open_long_position(strat, clock)

    st = strat._state["BTCUSD"]
    entry_mid = st.entry_mid
    assert entry_mid > 0

    # Push mid up to TP1 — entry_mid × (1 + 0.001).
    tp1_price = entry_mid * 1.001 + 0.01
    _set_book(market, tp1_price - 0.5, 50.0, tp1_price + 0.5, 50.0)
    clock.tick(0.2)
    sigs = strat.scan([market])
    assert len(sigs) == 1
    leg = sigs[0].legs[0]
    assert leg.reduce_only is True
    assert leg.side == "SELL"          # close a LONG
    assert leg.size_usd == pytest.approx(1000.0 * 0.60)
    assert sigs[0].metadata["exit"] == "take-profit-1"
    # Position should NOT be reset — strategy still riding the remaining 40%.
    assert strat._state["BTCUSD"].position_side == "LONG"
    assert strat._state["BTCUSD"].tp1_taken is True

    # Push mid up to TP2 — entry × 1.002.
    tp2_price = entry_mid * 1.002 + 0.01
    _set_book(market, tp2_price - 0.5, 50.0, tp2_price + 0.5, 50.0)
    clock.tick(0.2)
    sigs = strat.scan([market])
    assert len(sigs) == 1
    leg = sigs[0].legs[0]
    assert leg.reduce_only is True
    assert leg.side == "SELL"
    # Remaining 40% notional flattened.
    assert leg.size_usd == pytest.approx(1000.0 * 0.40)
    assert sigs[0].metadata["exit"] == "take-profit-2"
    assert strat._state["BTCUSD"].position_side is None


def test_wick_anchored_stop_loss():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, stop_extra_pct=0.0008)
    market = _open_long_position(strat, clock)

    st = strat._state["BTCUSD"]
    stop_price = st.stop_price
    assert stop_price > 0
    assert stop_price < st.wick_extreme   # below the wick low

    # Drop mid below the stop price.
    bad_price = stop_price - 0.5
    _set_book(market, bad_price - 0.5, 50.0, bad_price + 0.5, 50.0)
    clock.tick(0.2)
    sigs = strat.scan([market])
    assert len(sigs) == 1
    leg = sigs[0].legs[0]
    assert leg.reduce_only is True
    assert leg.side == "SELL"
    # Full remaining notional (no partial taken yet)
    assert leg.size_usd == pytest.approx(1000.0)
    assert sigs[0].metadata["exit"] == "stop-loss"
    assert strat._state["BTCUSD"].position_side is None


def test_stop_loss_only_closes_remaining_after_tp1():
    """If TP1 already took 60%, a later stop should flatten the 40% left."""
    clock = _Clock(t=1000.0)
    strat = _make_strategy(
        clock=clock,
        tp1_pct=0.001,
        tp1_fraction=0.6,
        stop_extra_pct=0.0008,
    )
    market = _open_long_position(strat, clock)
    st = strat._state["BTCUSD"]
    entry_mid = st.entry_mid

    # Hit TP1 first.
    tp1_price = entry_mid * 1.001 + 0.01
    _set_book(market, tp1_price - 0.5, 50.0, tp1_price + 0.5, 50.0)
    clock.tick(0.2)
    sigs = strat.scan([market])
    assert len(sigs) == 1
    assert sigs[0].legs[0].size_usd == pytest.approx(1000.0 * 0.60)

    # Now mid crashes below the wick-anchored stop.
    bad_price = st.stop_price - 0.5
    _set_book(market, bad_price - 0.5, 50.0, bad_price + 0.5, 50.0)
    clock.tick(0.2)
    sigs = strat.scan([market])
    assert len(sigs) == 1
    leg = sigs[0].legs[0]
    # Should close ONLY the remaining 40% (entry_size − tp1_size_usd).
    assert leg.size_usd == pytest.approx(1000.0 * 0.40)
    assert sigs[0].metadata["exit"] == "stop-loss"


def test_hold_timeout_flattens_position():
    clock = _Clock(t=1000.0)
    strat = _make_strategy(clock=clock, max_hold_seconds=10.0)
    market = _open_long_position(strat, clock)

    # Advance well past the timeout WITHOUT triggering TP/SL.
    clock.tick(30.0)
    # nudge the book a tick so the scan has something to do
    _set_book(market, 100.45, 50.0, 100.55, 50.0)
    sigs = strat.scan([market])
    assert len(sigs) == 1
    leg = sigs[0].legs[0]
    assert leg.reduce_only is True
    assert sigs[0].metadata["exit"] == "hold-timeout"
    assert sigs[0].metadata["hold_seconds"] > 10.0
    assert strat._state["BTCUSD"].position_side is None


def test_usd_pnl_exits_skip_partial_path():
    """USD-PnL mode flattens the whole position, never partial."""
    clock = _Clock(t=1000.0)
    portfolio = Portfolio(bankroll=100.0)
    strat = _make_strategy(
        clock=clock,
        portfolio=portfolio,
        take_profit_usd=5.0,
        stop_loss_usd=3.0,
        tp1_pct=0.001,
        tp1_fraction=0.6,
    )
    market = _open_long_position(strat, clock)
    st = strat._state["BTCUSD"]

    # Manually seed a position the portfolio knows about so the USD-PnL
    # path has something to read.
    key = Portfolio._key("BTCUSD", "BTCUSD")
    portfolio.positions[key] = Position(
        market_id="BTCUSD",
        outcome_id="BTCUSD",
        shares=10.0,
        avg_cost=st.entry_mid,
    )

    # Push mid up enough that pnl_usd > take_profit_usd.
    new_mid = st.entry_mid + 1.0   # 10 shares × $1 = +$10 of pnl
    _set_book(market, new_mid - 0.5, 50.0, new_mid + 0.5, 50.0)
    clock.tick(0.2)
    sigs = strat.scan([market])
    assert len(sigs) == 1
    leg = sigs[0].legs[0]
    assert sigs[0].metadata["exit"] == "take-profit"
    # Full notional (no partial path in USD mode)
    assert leg.size_usd == pytest.approx(1000.0)
    assert strat._state["BTCUSD"].position_side is None


def test_position_resets_when_portfolio_shows_flat():
    """If portfolio shows zero shares, USD-mode exits skip and reset state."""
    clock = _Clock(t=1000.0)
    portfolio = Portfolio(bankroll=100.0)
    strat = _make_strategy(
        clock=clock,
        portfolio=portfolio,
        take_profit_usd=5.0,
        stop_loss_usd=3.0,
    )
    market = _open_long_position(strat, clock)
    # Portfolio has NO position recorded (we never seeded one).
    _set_book(market, 100.5, 50.0, 101.5, 50.0)
    clock.tick(0.2)
    sigs = strat.scan([market])
    assert sigs == []
    assert strat._state["BTCUSD"].position_side is None
