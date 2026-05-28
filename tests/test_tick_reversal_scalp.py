"""TickReversalScalp + TickStream helpers.

Drive the strategy and the underlying tick stream deterministically:
fake clock, synthetic order books, manual control over every tick the
strategy will see. Covers entry conditions, every filter, and every exit
path.
"""
from __future__ import annotations

import pytest

from aera.core import Portfolio
from aera.core.portfolio import Position
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.signals.tick_stream import TickStream
from aera.strategies import TickReversalScalp


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
    """Single-level book Delta market — enough for tick-stream tests."""
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
    """Manually-advanced clock used by both the stream and the strategy."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 0.1) -> None:
        self.t += dt


def _drive_downtick_streak(
    sniper: TickReversalScalp,
    clock: _Clock,
    *,
    symbol: str = "BTCUSD",
    streak: int = 6,
    start_mid: float = 100.0,
    bid_size_seed: float = 100.0,
    decay_per_tick: float = 0.5,
    step_bps: float = 1.0,
) -> Market:
    """Drive a controlled exhaustion-DOWN streak.

    Mid drops by ``step_bps`` each iteration (well under the
    50-bps news-spike threshold). Bid price drops each tick → the tape
    inferrer treats the prior best-bid as fully eaten, so the per-tick
    eaten size = the *prior* recorded bid_size.

    By shrinking ``bid_size`` geometrically (``decay_per_tick`` < 1)
    we make the eaten series decay — satisfying the size-decay gate.
    Note that this means depth_trend on top-of-book bid_size will be
    NEGATIVE, so tests using this helper must set
    ``require_depth_trend=False`` (covered in a dedicated test).

    Returns the last market fed in.
    """
    market = _make_market(
        symbol, start_mid - 0.5, bid_size_seed, start_mid + 0.5, 50.0,
    )
    sniper.scan([market])
    clock.tick()

    bid_size = bid_size_seed
    mid = start_mid
    last_market = market
    for _ in range(streak):
        mid = mid * (1.0 - step_bps / 1e4)
        bid_size = max(1.0, bid_size * decay_per_tick)
        last_market = _make_market(
            symbol, mid - 0.5, bid_size, mid + 0.5, 50.0,
        )
        sniper.scan([last_market])
        clock.tick()
    return last_market


def _drive_uptick_streak(
    sniper: TickReversalScalp,
    clock: _Clock,
    *,
    symbol: str = "ETHUSD",
    streak: int = 6,
    start_mid: float = 100.0,
    ask_size_seed: float = 100.0,
    decay_per_tick: float = 0.5,
    step_bps: float = 1.0,
) -> Market:
    """Mirror of ``_drive_downtick_streak`` for upticks.

    Each iteration the ask price climbs (so the tape inferrer treats
    the prior best-ask as fully eaten) and the new best-ask size
    shrinks — producing a decaying eaten series.
    """
    market = _make_market(
        symbol, start_mid - 0.5, 50.0, start_mid + 0.5, ask_size_seed,
    )
    sniper.scan([market])
    clock.tick()

    ask_size = ask_size_seed
    mid = start_mid
    last_market = market
    for _ in range(streak):
        mid = mid * (1.0 + step_bps / 1e4)
        ask_size = max(1.0, ask_size * decay_per_tick)
        last_market = _make_market(
            symbol, mid - 0.5, 50.0, mid + 0.5, ask_size,
        )
        sniper.scan([last_market])
        clock.tick()
    return last_market


# ---------------------------------------------------------------------------
# TickStream
# ---------------------------------------------------------------------------


def test_tickstream_returns_none_on_first_observation():
    stream = TickStream()
    book = OrderBook()
    book.replace(bids=[(100, 10)], asks=[(101, 10)])
    assert stream.update(book, now=0.0) is None


def test_tickstream_returns_none_on_flat_mid():
    stream = TickStream()
    book = OrderBook()
    book.replace(bids=[(100, 10)], asks=[(101, 10)])
    stream.update(book, now=0.0)
    # Same mid → no tick recorded.
    stream.update(book, now=0.1)
    direction, length, _ = stream.current_streak()
    assert (direction, length) == (0, 0)


def test_tickstream_records_downtick_with_size_eaten_from_bid():
    stream = TickStream()
    book = OrderBook()
    book.replace(bids=[(100, 20)], asks=[(101, 10)])
    stream.update(book, now=0.0)
    book.replace(bids=[(100, 5)], asks=[(101, 10)])    # bid eaten 20 -> 5
    # Wait, mid is the same! Both bid_price and ask_price unchanged → mid unchanged → no tick.
    # We need to shrink the bid AND drop the mid. Drop both bid and ask.
    book.replace(bids=[(99.5, 5)], asks=[(100.5, 10)])
    tick = stream.update(book, now=0.1)
    assert tick is not None
    assert tick.direction == -1
    # Eaten on the leading (bid) side: bid_price dropped 100 -> 99.5, so the
    # 100-level was cleared. The inferrer charges the prior bid_size (20).
    assert tick.size == 20.0


def test_tickstream_records_uptick_with_size_eaten_from_ask():
    stream = TickStream()
    book = OrderBook()
    book.replace(bids=[(100, 10)], asks=[(101, 30)])
    stream.update(book, now=0.0)
    # Ask price held but size shrunk 30 -> 7 → ask was eaten by 23.
    # Bid price held but bid size grew (the new buy lifted offers, then
    # rejoined the bid → mid up).
    book.replace(bids=[(100.5, 12)], asks=[(101, 7)])
    tick = stream.update(book, now=0.1)
    assert tick is not None
    assert tick.direction == 1
    # Ask price unchanged, size shrank 30 -> 7 → eaten = 23.
    assert tick.size == 23.0


def test_tickstream_current_streak_tracks_consecutive_direction():
    stream = TickStream()
    book = OrderBook()

    # Prime
    book.replace(bids=[(100, 10)], asks=[(101, 10)])
    stream.update(book, now=0.0)

    # 3 downticks
    for i, mid_offset in enumerate([0.5, 1.0, 1.5]):
        book.replace(
            bids=[(100 - mid_offset, 10)],
            asks=[(101 - mid_offset, 10)],
        )
        stream.update(book, now=0.1 * (i + 1))

    direction, length, sizes = stream.current_streak()
    assert direction == -1
    assert length == 3
    assert len(sizes) == 3


def test_tickstream_streak_resets_on_direction_flip():
    stream = TickStream()
    book = OrderBook()
    book.replace(bids=[(100, 10)], asks=[(101, 10)])
    stream.update(book, now=0.0)

    # 2 downticks
    for offset in (0.5, 1.0):
        book.replace(bids=[(100 - offset, 10)], asks=[(101 - offset, 10)])
        stream.update(book, now=0.1)
    # 1 uptick
    book.replace(bids=[(99.5, 10)], asks=[(100.5, 10)])
    stream.update(book, now=0.5)

    direction, length, _ = stream.current_streak()
    assert direction == 1
    assert length == 1   # only the last uptick counts


def test_size_decay_returns_zero_for_short_streaks():
    assert TickStream.size_decay([]) == 0.0
    assert TickStream.size_decay([10.0]) == 0.0


def test_size_decay_computes_end_to_end_fraction():
    assert TickStream.size_decay([100.0, 80.0, 60.0, 20.0]) == pytest.approx(0.80)
    # No decay → 0
    assert TickStream.size_decay([100.0, 100.0]) == 0.0


def test_tickstream_recent_extreme_lookback():
    stream = TickStream()
    book = OrderBook()
    mids = [100, 99, 98, 99, 101, 102]
    for i, m in enumerate(mids):
        book.replace(bids=[(m - 0.5, 10)], asks=[(m + 0.5, 10)])
        stream.update(book, now=float(i))
    # min across last 5 ticks (excluding the priming first one)
    assert stream.recent_extreme(-1, 5) == pytest.approx(98.0)
    assert stream.recent_extreme(+1, 5) == pytest.approx(102.0)


def test_tickstream_spread_multiple_tracks_ema():
    stream = TickStream(spread_ema_alpha=0.5)
    book = OrderBook()
    # Prime with spread = 1.0
    book.replace(bids=[(100, 10)], asks=[(101, 10)])
    stream.update(book, now=0.0)
    # Now blow the spread to 5
    book.replace(bids=[(99, 10)], asks=[(104, 10)])
    stream.update(book, now=0.1)
    mult = stream.current_spread_multiple()
    assert mult is not None
    # EMA = 0.5*5 + 0.5*1 = 3. Spread = 5. Multiple = 5/3 ≈ 1.67.
    assert mult == pytest.approx(5.0 / 3.0, rel=0.01)


def test_tickstream_max_tick_move_bps_window():
    stream = TickStream()
    book = OrderBook()
    book.replace(bids=[(100, 10)], asks=[(101, 10)])
    stream.update(book, now=0.0)
    # Small move 100.5 -> 100.0 (~5 bps)
    book.replace(bids=[(99.5, 10)], asks=[(100.5, 10)])
    stream.update(book, now=1.0)
    # Big move 100 -> 110 (~1000 bps)
    book.replace(bids=[(109.5, 10)], asks=[(110.5, 10)])
    stream.update(book, now=2.0)
    move = stream.max_tick_move_bps(5.0, now=2.5)
    assert move > 900   # roughly 1000 bps


def test_tickstream_depth_trend_compares_endpoints():
    stream = TickStream()
    book = OrderBook()
    # Prime + 3 downticks with bid size growing each tick → +1 trend.
    book.replace(bids=[(100, 10)], asks=[(101, 10)])
    stream.update(book, now=0.0)
    bids = [20, 30, 40]
    for i, sz in enumerate(bids):
        book.replace(bids=[(99 - i, sz)], asks=[(100 - i, 10)])
        stream.update(book, now=float(i + 1))
    # 3 ticks in buffer, lookback 3 → compare bid_size[0] (=20) to [-1] (=40).
    assert stream.depth_trend("bid", lookback=3) > 0
    assert stream.depth_trend("ask", lookback=3) == 0  # flat ask size


def test_tickstream_volume_spike_ratio_compares_windows():
    stream = TickStream()
    book = OrderBook()
    # Establish a baseline of small ticks over 50 s
    book.replace(bids=[(100, 50)], asks=[(101, 50)])
    stream.update(book, now=0.0)
    for i in range(5):
        book.replace(bids=[(99 - i * 0.1, 49 - i)], asks=[(100 - i * 0.1, 50)])
        stream.update(book, now=float(10 * (i + 1)))   # spaced 10s apart, small ticks

    # Now slam a flurry of LARGE eats in the last 3s
    base_mid = 98.5
    for j in range(4):
        base_mid -= 0.5
        book.replace(bids=[(base_mid - 0.5, 1)], asks=[(base_mid + 0.5, 50)])
        stream.update(book, now=60.0 + j * 0.5)
    ratio = stream.volume_spike_ratio(
        short_seconds=5.0, long_seconds=60.0, now=62.0,
    )
    assert ratio is not None
    assert ratio > 1.0


# ---------------------------------------------------------------------------
# TickReversalScalp — basic firing
# ---------------------------------------------------------------------------


def test_strategy_skips_non_delta_markets():
    clock = _Clock()
    strat = TickReversalScalp(clock=clock)
    market = _make_market("BTCUSD", 99.5, 10, 100.5, 10)
    market.venue = "other"
    for _ in range(20):
        out = strat.scan([market])
    assert out == []


def test_strategy_fires_long_after_downtick_exhaustion():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,                  # disable S/R for this happy-path test
        require_depth_trend=False,        # see helper docstring
        max_spread_multiple=0.0,          # disable spread filter
        news_max_tick_bps=0.0,            # disable news filter
        volume_spike_multiple=0.0,        # disable volume spike
        max_hold_seconds=0.0,             # disable hold timeout
        rearm_distance_bps=0.0,
        clock=clock,
    )
    _drive_downtick_streak(strat, clock, streak=6, decay_per_tick=0.5)
    state = strat._state["BTCUSD"]
    assert state.position_side == "LONG"


def test_strategy_does_not_fire_when_streak_too_short():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=10,                     # require 10 ticks
        size_decay_threshold=0.0,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    _drive_downtick_streak(strat, clock, streak=4)
    assert strat._state["BTCUSD"].position_side is None


def test_strategy_does_not_fire_when_size_not_decaying():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.30,         # demand 30% decay
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    # decay_per_tick = 1.0 → bid_size stays flat → eaten size flat → 0% decay
    _drive_downtick_streak(strat, clock, streak=6, decay_per_tick=1.0)
    assert strat._state["BTCUSD"].position_side is None


def test_strategy_does_not_fire_when_bid_depth_not_increasing():
    """require_depth_trend on, but the favoured-side depth shrinks.

    Single-level books inherently couple "what was eaten" with "what's
    now resting" on the same side, so when bid_size shrinks across the
    streak (which our decaying-eaten helper requires), depth_trend
    necessarily reports < 0. This is the case we want: decay passes,
    depth_trend fails → strategy must NOT fire.
    """
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=True,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    _drive_downtick_streak(strat, clock, streak=6, decay_per_tick=0.5)
    assert strat._state["BTCUSD"].position_side is None


def test_strategy_fires_short_after_uptick_exhaustion():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    _drive_uptick_streak(strat, clock, streak=6, decay_per_tick=0.5)
    assert strat._state["ETHUSD"].position_side == "SHORT"


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_strategy_vetoes_on_news_spike():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_lookback_seconds=60.0,
        news_max_tick_bps=20.0,           # ANY tick > 20 bps trips
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    # Inject a 50-bps single-tick move right at the start, then drive
    # the exhaustion streak — filter must veto.
    market = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    strat.scan([market])
    clock.tick()
    # 50 bps drop in a single tick — flash-event signature.
    market = _make_market("BTCUSD", 99.0, 100, 100.0, 100)
    strat.scan([market])
    clock.tick()
    # Now drive an otherwise-valid downtick streak in calmer steps.
    _drive_downtick_streak(strat, clock, streak=6, start_mid=99.5,
                           decay_per_tick=0.5)
    assert strat._state["BTCUSD"].position_side is None


def test_strategy_vetoes_on_wide_spread():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=3.0,
        spread_ema_alpha=0.05,            # slow EMA → blow-out trips the filter
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    # Prime tight, then drive a downtick streak. min_streak=5 → the
    # strategy will TRY to fire on the 5th downtick (i=4). Widen the
    # spread on exactly that tick so the filter trips. 4 tight ticks
    # let the spread EMA converge near 1.0; the 5th's spread of 10.0
    # is then ~10× the EMA → filter trips at threshold 3.
    market = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    strat.scan([market])
    clock.tick()
    bid_size = 100.0
    mid = 100.0
    for i in range(6):
        mid *= 1.0 - 1.0 / 1e4
        bid_size = max(1.0, bid_size * 0.5)
        # Keep the spread wide from the firing tick onward — otherwise
        # the next tick (whose streak is also >= min_streak) would re-
        # attempt entry with a tight spread and the filter wouldn't fire.
        half_spread = 5.0 if i >= 4 else 0.5
        market = _make_market(
            "BTCUSD",
            mid - half_spread, bid_size,
            mid + half_spread, 50.0,
        )
        strat.scan([market])
        clock.tick()
    assert strat._state["BTCUSD"].position_side is None


def test_strategy_vetoes_on_sr_band_mismatch():
    """Tight sr_band_bps + lookback that includes an earlier deep dip →
    the local extreme is far from the current mid → filter trips."""
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=1.0,                  # very tight — 1 bp band
        sr_lookback_ticks=20,             # long enough to retain the dip
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    # Prime + a deep dip + bounce back, then drive a downtick streak.
    # The buffer's min mid (the dip) is ~90, current mid ~100 → SR fails.
    market = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    strat.scan([market])
    clock.tick()
    market = _make_market("BTCUSD", 89.5, 100, 90.5, 100)
    strat.scan([market])
    clock.tick()
    market = _make_market("BTCUSD", 99.5, 100, 100.5, 100)
    strat.scan([market])
    clock.tick()
    _drive_downtick_streak(strat, clock, streak=6, start_mid=100.0,
                           decay_per_tick=0.5)
    assert strat._state["BTCUSD"].position_side is None


# ---------------------------------------------------------------------------
# Entry mechanics
# ---------------------------------------------------------------------------


def test_strategy_emits_buy_signal_with_metadata():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    _drive_downtick_streak(strat, clock, streak=6, decay_per_tick=0.5)
    state = strat._state["BTCUSD"]
    assert state.position_side == "LONG"
    assert state.entry_size_usd == strat.notional_usd
    assert state.entry_mid > 0
    # entry_time was stamped at the firing tick; the exact value depends
    # on which tick of the streak crossed min_streak. Just confirm it's
    # within the streak window.
    assert 1000.0 < state.entry_time < clock.t


def test_strategy_does_not_stack_entries():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_downtick_streak(strat, clock, streak=6)
    # Keep feeding the same favourable setup — no second entry.
    for _ in range(5):
        out = strat.scan([market])
        clock.tick()
        non_reduce = [s for s in out if not s.legs[0].reduce_only]
        assert not non_reduce


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------


def test_strategy_take_profit_closes_long():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        take_profit_pct=0.0004,
        stop_loss_pct=0.00025,
        max_hold_seconds=0.0,           # disable hold timeout for this test
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_downtick_streak(strat, clock, streak=6)
    entry_mid = strat._state[market.id].entry_mid

    # Push mid clearly above entry * (1 + TP)
    target = entry_mid * 1.001          # 10 bps move, well above 4-bps TP
    push = _make_market(market.id, target - 0.5, 100, target + 0.5, 100)
    out = strat.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "take-profit"]
    assert exits, "expected a take-profit close"
    assert exits[0].legs[0].side == "SELL"
    assert exits[0].legs[0].reduce_only is True


def test_strategy_stop_loss_closes_long():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        take_profit_pct=0.0004,
        stop_loss_pct=0.00025,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_downtick_streak(strat, clock, streak=6)
    entry_mid = strat._state[market.id].entry_mid

    target = entry_mid * 0.999          # 10 bps below, past 2.5-bp SL
    push = _make_market(market.id, target - 0.5, 100, target + 0.5, 100)
    out = strat.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "stop-loss"]
    assert exits, "expected a stop-loss close"


def test_strategy_hold_timeout_forces_exit():
    clock = _Clock()
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        max_hold_seconds=30.0,
        rearm_distance_bps=0.0,
        clock=clock,
    )
    market = _drive_downtick_streak(strat, clock, streak=6)
    # Advance well past the 30s hold limit while price barely moves.
    clock.tick(dt=35.0)
    out = strat.scan([market])
    exits = [s for s in out if s.metadata.get("exit") == "hold-timeout"]
    assert exits, "expected a hold-timeout close"
    assert exits[0].metadata["hold_seconds"] > 30.0


# ---------------------------------------------------------------------------
# USD-PnL exit path (mirrors the other strategies' contract)
# ---------------------------------------------------------------------------


def _seed_position(portfolio: Portfolio, symbol: str, *, shares: float, avg_cost: float) -> None:
    key = Portfolio._key(symbol, symbol)
    pos = Position(market_id=symbol, outcome_id=symbol)
    pos.shares = shares
    pos.avg_cost = avg_cost
    portfolio.positions[key] = pos


def test_strategy_usd_take_profit_closes_long_at_target_profit():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    strat = TickReversalScalp(
        min_streak=5,
        size_decay_threshold=0.20,
        sr_band_bps=0.0,
        require_depth_trend=False,
        max_spread_multiple=0.0,
        news_max_tick_bps=0.0,
        volume_spike_multiple=0.0,
        take_profit_pct=0.0,
        stop_loss_pct=0.0,
        take_profit_usd=5.0,
        stop_loss_usd=3.0,
        max_hold_seconds=0.0,
        rearm_distance_bps=0.0,
        portfolio=portfolio,
        clock=clock,
    )
    market = _drive_downtick_streak(strat, clock, streak=6)
    _seed_position(portfolio, market.id, shares=10.0, avg_cost=100.0)

    push = _make_market(market.id, 100.6, 100, 101.2, 100)
    out = strat.scan([push])
    exits = [s for s in out if s.metadata.get("exit") == "take-profit"]
    assert exits, "expected USD take-profit close"
    assert exits[0].metadata["pnl_usd"] == pytest.approx(6.0, abs=1e-6)
