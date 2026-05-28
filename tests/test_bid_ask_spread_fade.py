"""BidAskSpreadFade (market making lite) tests.

The strategy emits two independent single-leg signals (BUY + SELL) per
quote cycle when all gates pass. Tests drive a manual clock + synthetic
order books so every gate (kill switch, spread, fee, refresh, inventory
cap, inventory skew) can be exercised without sleeping.
"""
from __future__ import annotations

import pytest

from aera.core import Portfolio
from aera.core.portfolio import Position
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.strategies import BidAskSpreadFade


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


def _wide_spread_market(
    symbol: str = "BTCUSD",
    mid: float = 100.0,
    spread_pct: float = 0.002,        # 0.2% spread; capture=60% → 12 bps gross,
                                      # net 8 bps after 2 × 2 bps maker fees,
                                      # clears the 4 bps net-edge floor.
) -> Market:
    half = mid * spread_pct * 0.5
    return _make_market(
        symbol,
        bids=[(mid - half, 50.0)],
        asks=[(mid + half, 50.0)],
        tick=0.01,
    )


def _tight_spread_market(
    symbol: str = "BTCUSD",
    mid: float = 100.0,
    spread_pct: float = 0.0001,       # 0.01% — below 0.03% gate
) -> Market:
    half = mid * spread_pct * 0.5
    return _make_market(
        symbol,
        bids=[(mid - half, 50.0)],
        asks=[(mid + half, 50.0)],
        tick=0.01,
    )


class _Clock:
    """Manually-advanced clock so refresh-rate / kill-switch tests are deterministic."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 0.6) -> None:
        # default tick is > 500 ms refresh gate so each scan can emit a quote
        self.t += dt


def _seed_position(
    portfolio: Portfolio,
    symbol: str,
    *,
    shares: float,
    avg_cost: float,
) -> None:
    key = Portfolio._key(symbol, symbol)
    pos = Position(market_id=symbol, outcome_id=symbol)
    pos.shares = shares
    pos.avg_cost = avg_cost
    portfolio.positions[key] = pos


def _default_mm(
    *,
    clock: _Clock,
    portfolio: Portfolio | None = None,
    **overrides,
) -> BidAskSpreadFade:
    """Build an MM with spec-default tunables, easy to override per-test."""
    kwargs = dict(
        min_spread_pct=0.0003,
        capture_target=0.60,
        quote_size_usd=5.0,
        max_inventory_usd=15.0,
        inventory_skew_threshold_usd=10.0,
        inventory_skew_ticks=1,
        refresh_rate_ms=500.0,
        kill_move_pct=0.0008,
        kill_window_seconds=5.0,
        maker_fee_bps=2.0,
        min_net_edge_bps=4.0,
        leverage_override=1.0,
        portfolio=portfolio,
        clock=clock,
    )
    kwargs.update(overrides)
    return BidAskSpreadFade(**kwargs)


# ---------------------------------------------------------------------------
# basic gating
# ---------------------------------------------------------------------------


def test_mm_skips_non_delta_markets():
    clock = _Clock()
    mm = _default_mm(clock=clock)
    market = _wide_spread_market()
    market.venue = "other"
    out = mm.scan([market])
    assert out == []


def test_mm_emits_buy_and_sell_when_gates_pass():
    clock = _Clock()
    mm = _default_mm(clock=clock)
    market = _wide_spread_market(mid=100.0, spread_pct=0.002)
    out = mm.scan([market])
    sides = sorted(s.legs[0].side for s in out)
    assert sides == ["BUY", "SELL"]
    assert all(s.legs[0].size_usd == 5.0 for s in out)
    assert all(s.legs[0].leverage == 1.0 for s in out)


def test_mm_quote_prices_capture_inner_fraction_of_spread():
    """With capture=0.60 and spread=0.20, quotes sit at mid ± 0.06."""
    clock = _Clock()
    mm = _default_mm(clock=clock, capture_target=0.60)
    market = _make_market(
        "BTCUSD",
        bids=[(99.9, 50.0)],
        asks=[(100.1, 50.0)],
        tick=0.01,
    )
    out = mm.scan([market])
    by_side = {s.legs[0].side: s.legs[0] for s in out}
    assert by_side["BUY"].limit_price == pytest.approx(99.94)
    assert by_side["SELL"].limit_price == pytest.approx(100.06)


def test_mm_skips_when_spread_below_floor():
    clock = _Clock()
    mm = _default_mm(clock=clock, min_spread_pct=0.0003)
    market = _tight_spread_market(spread_pct=0.00005)  # well under 0.03%
    out = mm.scan([market])
    assert out == []


def test_mm_skips_when_net_edge_below_floor():
    """If capture × spread − 2 × maker_fee < min_net_edge_bps, no quote."""
    clock = _Clock()
    # min_net_edge=8 bps; spread=10 bps; capture=0.60 -> 6 bps; -2*2=2 bps net.
    # That's below 8 bps so the cycle must be vetoed.
    mm = _default_mm(
        clock=clock,
        maker_fee_bps=2.0,
        min_net_edge_bps=8.0,
        capture_target=0.60,
    )
    market = _wide_spread_market(spread_pct=0.001)
    out = mm.scan([market])
    assert out == []


def test_mm_emits_when_net_edge_passes():
    """Loosen the net-edge gate enough that the default 0.1% spread clears."""
    clock = _Clock()
    mm = _default_mm(
        clock=clock,
        maker_fee_bps=2.0,
        min_net_edge_bps=1.0,         # very permissive
        capture_target=0.60,
    )
    market = _wide_spread_market(spread_pct=0.001)
    out = mm.scan([market])
    assert len(out) == 2


# ---------------------------------------------------------------------------
# refresh-rate gate
# ---------------------------------------------------------------------------


def test_mm_refresh_rate_gate_throttles_quotes():
    clock = _Clock()
    mm = _default_mm(clock=clock, refresh_rate_ms=500.0)
    market = _wide_spread_market()
    # first call quotes
    first = mm.scan([market])
    assert len(first) == 2
    # next call inside the 500ms window — no quotes
    clock.tick(dt=0.1)
    second = mm.scan([market])
    assert second == []
    # after the window — quotes again
    clock.tick(dt=0.5)
    third = mm.scan([market])
    assert len(third) == 2


# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------


def test_mm_kill_switch_trips_on_fast_move():
    clock = _Clock()
    mm = _default_mm(
        clock=clock,
        kill_move_pct=0.0008,         # 0.08%
        kill_window_seconds=5.0,
    )
    # First scan: mid=100, fills the buffer with one point
    out1 = mm.scan([_wide_spread_market(mid=100.0)])
    assert len(out1) == 2
    # Advance just 1s and jump mid by 0.2% — exceeds 0.08% threshold
    clock.tick(dt=1.0)
    out2 = mm.scan([_wide_spread_market(mid=100.2)])
    # Kill switch should have tripped — no quotes this cycle.
    assert out2 == []


def test_mm_kill_switch_does_not_trip_on_calm_book():
    clock = _Clock()
    mm = _default_mm(clock=clock)
    # Same mid across multiple ticks — vol = 0, no kill.
    out = []
    for _ in range(4):
        out = mm.scan([_wide_spread_market(mid=100.0)])
        clock.tick(dt=0.6)
    assert len(out) == 2


def test_mm_kill_switch_resets_after_window_passes():
    """Once the spike rolls off the window, quoting must resume."""
    clock = _Clock()
    mm = _default_mm(clock=clock, kill_window_seconds=2.0)
    mm.scan([_wide_spread_market(mid=100.0)])
    # Jump price + within window
    clock.tick(dt=0.5)
    out_jump = mm.scan([_wide_spread_market(mid=100.5)])
    assert out_jump == []   # killed
    # Far outside window: only the latest mid remains -> vol=0, quoting resumes
    clock.tick(dt=5.0)
    out_calm = mm.scan([_wide_spread_market(mid=100.5)])
    assert len(out_calm) == 2


# ---------------------------------------------------------------------------
# inventory skew
# ---------------------------------------------------------------------------


def test_mm_skew_shifts_quotes_down_when_long():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    # Long $12 of inventory (0.12 shares × $100 mid) — past $10 skew
    # threshold, under $15 hard cap. Both quotes should shift down by
    # 1 tick (= 0.01).
    _seed_position(portfolio, "BTCUSD", shares=0.12, avg_cost=100.0)
    mm = _default_mm(
        clock=clock,
        portfolio=portfolio,
        inventory_skew_threshold_usd=10.0,
        max_inventory_usd=15.0,
        inventory_skew_ticks=1,
    )
    market = _make_market(
        "BTCUSD",
        bids=[(99.9, 50.0)],
        asks=[(100.1, 50.0)],
        tick=0.01,
    )
    out = mm.scan([market])
    by_side = {s.legs[0].side: s.legs[0] for s in out}
    # Without skew: bid=99.94, ask=100.06. With −1 tick: bid=99.93, ask=100.05.
    assert by_side["BUY"].limit_price == pytest.approx(99.93, abs=1e-6)
    assert by_side["SELL"].limit_price == pytest.approx(100.05, abs=1e-6)


def test_mm_skew_shifts_quotes_up_when_short():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    _seed_position(portfolio, "BTCUSD", shares=-0.12, avg_cost=100.0)
    mm = _default_mm(
        clock=clock,
        portfolio=portfolio,
        inventory_skew_threshold_usd=10.0,
        max_inventory_usd=15.0,
        inventory_skew_ticks=1,
    )
    market = _make_market(
        "BTCUSD",
        bids=[(99.9, 50.0)],
        asks=[(100.1, 50.0)],
        tick=0.01,
    )
    out = mm.scan([market])
    by_side = {s.legs[0].side: s.legs[0] for s in out}
    # +1 tick shift on both: bid=99.95, ask=100.07.
    assert by_side["BUY"].limit_price == pytest.approx(99.95, abs=1e-6)
    assert by_side["SELL"].limit_price == pytest.approx(100.07, abs=1e-6)


def test_mm_no_skew_when_inventory_inside_band():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    # $5 long — inside the $10 threshold, no skew expected
    _seed_position(portfolio, "BTCUSD", shares=0.05, avg_cost=100.0)
    mm = _default_mm(clock=clock, portfolio=portfolio)
    market = _make_market(
        "BTCUSD",
        bids=[(99.9, 50.0)],
        asks=[(100.1, 50.0)],
        tick=0.01,
    )
    out = mm.scan([market])
    by_side = {s.legs[0].side: s.legs[0] for s in out}
    assert by_side["BUY"].limit_price == pytest.approx(99.94)
    assert by_side["SELL"].limit_price == pytest.approx(100.06)


# ---------------------------------------------------------------------------
# inventory cap
# ---------------------------------------------------------------------------


def test_mm_suppresses_buy_when_long_exceeds_cap():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    # +$20 long, past the $15 cap. BUY suppressed; SELL must still post
    # so the inventory can walk back.
    _seed_position(portfolio, "BTCUSD", shares=0.2, avg_cost=100.0)
    mm = _default_mm(
        clock=clock,
        portfolio=portfolio,
        max_inventory_usd=15.0,
    )
    out = mm.scan([_wide_spread_market(mid=100.0)])
    sides = [s.legs[0].side for s in out]
    assert sides == ["SELL"]


def test_mm_suppresses_sell_when_short_exceeds_cap():
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    _seed_position(portfolio, "BTCUSD", shares=-0.2, avg_cost=100.0)
    mm = _default_mm(
        clock=clock,
        portfolio=portfolio,
        max_inventory_usd=15.0,
    )
    out = mm.scan([_wide_spread_market(mid=100.0)])
    sides = [s.legs[0].side for s in out]
    assert sides == ["BUY"]


def test_mm_quotes_both_sides_when_no_portfolio_attached():
    """Without a portfolio, inventory tracking degrades to 0 — both sides quote."""
    clock = _Clock()
    mm = _default_mm(clock=clock, portfolio=None)
    out = mm.scan([_wide_spread_market()])
    assert sorted(s.legs[0].side for s in out) == ["BUY", "SELL"]


# ---------------------------------------------------------------------------
# signal shape / metadata
# ---------------------------------------------------------------------------


def test_mm_signal_metadata_describes_quote_cycle():
    clock = _Clock()
    mm = _default_mm(clock=clock)
    market = _wide_spread_market(mid=100.0, spread_pct=0.002)
    out = mm.scan([market])
    buy = next(s for s in out if s.legs[0].side == "BUY")
    md = buy.metadata
    assert md["symbol"] == "BTCUSD"
    assert md["quote_side"] == "BUY"
    assert md["mid"] == pytest.approx(100.0)
    assert md["spread"] == pytest.approx(0.2, abs=1e-6)
    # capture = 60% of 20 bps = 12 bps
    assert md["capture_bps"] == pytest.approx(12.0, abs=1e-6)
    # net = 12 − 2 × 2 = 8 bps
    assert md["net_bps_after_fees"] == pytest.approx(8.0, abs=1e-6)
    assert md["inventory_usd"] == pytest.approx(0.0)
    assert "limit_price" in md


def test_mm_signal_legs_are_not_reduce_only():
    """MM quotes open positions; they must never be reduce-only."""
    clock = _Clock()
    mm = _default_mm(clock=clock)
    out = mm.scan([_wide_spread_market()])
    assert all(s.legs[0].reduce_only is False for s in out)


def test_mm_leverage_override_stamps_on_legs():
    clock = _Clock()
    mm = _default_mm(clock=clock, leverage_override=1.0)
    market = _wide_spread_market()
    market.metadata["leverage"] = 50.0     # venue at 50x, but override to 1x
    out = mm.scan([market])
    assert all(s.legs[0].leverage == 1.0 for s in out)


def test_mm_uses_venue_leverage_when_override_is_none():
    clock = _Clock()
    mm = _default_mm(clock=clock, leverage_override=None)
    market = _wide_spread_market()
    market.metadata["leverage"] = 25.0
    out = mm.scan([market])
    assert all(s.legs[0].leverage == 25.0 for s in out)


# ---------------------------------------------------------------------------
# edge case: skewed quote must not cross the spread
# ---------------------------------------------------------------------------


def test_mm_skew_never_crosses_the_spread():
    """A pathological skew must keep BUY < best_ask and SELL > best_bid.

    Crossing would turn the quote into a taker order (eating the other
    side's liquidity), defeating the entire point of MM. The strategy
    clamps to best_ask − 1 tick / best_bid + 1 tick when the skew is
    aggressive enough to push the quote across the spread.
    """
    clock = _Clock()
    portfolio = Portfolio(bankroll=1000.0)
    # Long $14 — past skew threshold ($10), under cap ($15) so both
    # quotes emit. With 100 ticks of downward skew the quotes would
    # land far below mid; we only assert non-crossing here.
    _seed_position(portfolio, "BTCUSD", shares=0.14, avg_cost=100.0)
    mm = _default_mm(
        clock=clock,
        portfolio=portfolio,
        inventory_skew_threshold_usd=1.0,
        max_inventory_usd=15.0,
        inventory_skew_ticks=100,      # absurd shift
    )
    market = _make_market(
        "BTCUSD",
        bids=[(99.9, 50.0)],
        asks=[(100.1, 50.0)],
        tick=0.01,
    )
    out = mm.scan([market])
    for s in out:
        if s.legs[0].side == "BUY":
            assert s.legs[0].limit_price < 100.1
        if s.legs[0].side == "SELL":
            assert s.legs[0].limit_price > 99.9
