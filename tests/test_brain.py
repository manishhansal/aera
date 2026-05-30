"""Adaptive Brain — live edge tracker + regime router + circuit-breakers.

Covers:

* :class:`RegimeDetector` classification (RANGE / TREND_* / HIGH_VOL /
  NEWS_SPIKE) from a streaming sequence of mid prints.
* :class:`AdaptiveBrain` performance tracker: rolling win-rate, expectancy,
  mute on losing streak, mute on bad rolling stats, mute expiry +
  probation, probation graduation.
* :meth:`AdaptiveBrain.filter_signals`: regime veto, mute veto, daily-loss
  veto (passes reduce-only legs), correlation cap veto, size shrink.
* End-to-end: a healthy strategy survives, a losing strategy is muted
  after the threshold trips and re-engaged on probation at half size.
"""
from __future__ import annotations

import pytest

from aera.core import AdaptiveBrain, Portfolio
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.settings import BrainConfig
from aera.signals.regime import Regime, RegimeDetector
from aera.strategies import Leg, Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_market(symbol: str = "BTCUSD", price: float = 1000.0) -> Market:
    half = price * 5e-5  # ~10 bps spread
    book = OrderBook()
    book.replace(bids=[(price - half, 100.0)], asks=[(price + half, 100.0)])
    return Market(
        id=symbol, slug=symbol.lower(), question=f"{symbol} perp",
        category="perpetual_futures",
        outcomes={symbol: Outcome(id=symbol, label=DELTA_OUTCOME_LABEL, book=book)},
        venue="delta",
        metadata={"contract_value": 1.0, "leverage": 25.0},
    )


def _make_signal(
    strategy: str = "delta_perp_scalper",
    *,
    market_id: str = "BTCUSD",
    side: str = "BUY",
    size_usd: float = 100.0,
    reduce_only: bool = False,
) -> Signal:
    leg = Leg(
        market_id=market_id, outcome_id=market_id, side=side,
        limit_price=1000.0, size_usd=size_usd, reason="test",
        leverage=25.0, reduce_only=reduce_only,
    )
    return Signal(
        strategy=strategy, confidence=0.5, edge=0.005, legs=[leg],
        metadata={"symbol": market_id},
    )


def _cfg(**overrides) -> BrainConfig:
    base = dict(
        enabled=True,
        min_trades_for_eval=5,
        perf_window=10,
        min_win_rate=0.40,
        min_expectancy_usd=0.0,
        max_strategy_loss_streak=3,
        mute_seconds=600.0,
        # Disabled by default in test cfgs so the existing test
        # suite (which fires the same strategy on the same symbol
        # multiple times in a row) keeps passing. The dedicated
        # post-loss-cool-down test re-enables it.
        post_loss_cooldown_seconds=0.0,
        probation_trades=3,
        probation_size_mult=0.5,
        regime_short_window=10,
        regime_long_window=50,
        regime_trend_threshold=0.30,
        regime_high_vol_ratio=2.0,
        regime_news_tick_bps=25.0,
        high_vol_size_mult=0.5,
        daily_loss_pct=0.10,
        daily_window_seconds=86400.0,
        max_gross_exposure_mult=2.0,
        # Cost-aware edge gate disabled by default for the existing
        # suite — most fixtures use a 50 bps edge with 0 fees so the
        # default 5 bps floor is a no-op anyway, but explicitly
        # disabling keeps regression tests honest. The dedicated
        # cost-gate tests below re-enable it.
        min_edge_after_costs_bps=0.0,
        cost_round_trip_legs=2,
        cost_assumed_slippage_bps=0.0,
        # Lifetime kill switch disabled in default fixture cfg —
        # existing tests feed many losing trades to test other gates
        # and shouldn't trip the lifetime kill. Dedicated tests
        # re-enable it.
        lifetime_pnl_kill_floor_usd=0.0,
    )
    base.update(overrides)
    return BrainConfig(**base)


class _FixedClock:
    """Manual clock that advances only when explicitly tick()'d."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------


def test_regime_detector_returns_unknown_until_warm():
    d = RegimeDetector(short_window=5, long_window=20)
    # Cold start, no samples
    snap = d.snapshot()
    assert snap.regime == Regime.UNKNOWN

    # 3 samples isn't enough (gate is 5)
    d.observe(100.0)
    d.observe(100.01)
    d.observe(100.02)
    assert d.snapshot().regime == Regime.UNKNOWN


def test_regime_detector_classifies_pure_uptrend():
    d = RegimeDetector(short_window=5, long_window=30, trend_threshold=0.2)
    # Pure +1 bps per tick = clean trend, very low vol noise → high
    # drift/vol ratio after warm-up.
    for i in range(40):
        d.observe(100.0 * (1.0 + 0.0001 * (i + 1)))
    snap = d.snapshot()
    assert snap.regime == Regime.TREND_UP
    assert snap.trend_score > 0


def test_regime_detector_classifies_pure_downtrend():
    d = RegimeDetector(short_window=5, long_window=30, trend_threshold=0.2)
    for i in range(40):
        d.observe(100.0 * (1.0 - 0.0001 * (i + 1)))
    snap = d.snapshot()
    assert snap.regime == Regime.TREND_DOWN
    assert snap.trend_score < 0


def test_regime_detector_classifies_range_when_chop():
    d = RegimeDetector(short_window=5, long_window=30, trend_threshold=0.5)
    # Alternating up/down ticks of equal size cancel each other in the
    # drift EMA so the trend_score stays close to 0 → RANGE.
    base = 100.0
    for i in range(40):
        d.observe(base + (0.05 if i % 2 == 0 else -0.05))
    snap = d.snapshot()
    assert snap.regime == Regime.RANGE


def test_regime_detector_news_spike_dominates_trend():
    d = RegimeDetector(
        short_window=5, long_window=30,
        news_tick_bps=20.0, trend_threshold=0.2,
    )
    for i in range(10):
        d.observe(100.0 + i * 0.001)  # small drift, warm-up
    # 50 bps single-tick jump
    d.observe(100.5)
    snap = d.snapshot()
    assert snap.regime == Regime.NEWS_SPIKE
    assert snap.last_tick_bps > 20.0


def test_regime_detector_high_vol_when_short_atr_blows_out():
    # Long-window calm baseline, then a burst of large moves.
    d = RegimeDetector(
        short_window=5, long_window=40,
        high_vol_ratio=2.0, trend_threshold=10.0,
        news_tick_bps=200.0,  # disable news veto so we hit HIGH_VOL
    )
    for i in range(45):
        # alternating tiny moves to keep long-window ATR small but
        # avoid creating drift
        d.observe(100.0 + (0.0001 if i % 2 == 0 else -0.0001))
    # Now spike the short window with much larger moves (no news veto).
    for i in range(6):
        d.observe(100.0 + (0.01 if i % 2 == 0 else -0.01))
    snap = d.snapshot()
    assert snap.regime == Regime.HIGH_VOL
    assert snap.vol_ratio >= 2.0


# ---------------------------------------------------------------------------
# AdaptiveBrain — performance tracker
# ---------------------------------------------------------------------------


def test_brain_tracks_rolling_win_rate_and_expectancy():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(_cfg(perf_window=5), pf, clock=clock)
    name = "delta_perp_scalper"

    # Feed 4 wins + 1 loss
    for v in [1.0, 1.0, 1.0, 1.0, -2.0]:
        brain.on_trade_closed(name, v)

    p = brain.perf(name)
    assert p.n == 5
    assert p.wins == 4
    assert p.losses == 1
    assert p.win_rate == pytest.approx(0.8)
    assert p.expectancy == pytest.approx((4 - 2) / 5)


def test_brain_perf_window_evicts_old_pnls():
    pf = Portfolio(bankroll=100.0)
    brain = AdaptiveBrain(_cfg(perf_window=3), pf, clock=_FixedClock())
    name = "delta_perp_scalper"
    for v in [-1.0, -1.0, -1.0, 5.0]:
        brain.on_trade_closed(name, v)
    p = brain.perf(name)
    # Window is 3 → the first loss falls off, only [-1, -1, 5] remain.
    assert p.n == 3
    assert p.wins == 1
    assert p.losses == 2
    # Lifetime counter is unaffected.
    assert p.total_trades == 4
    assert p.total_pnl == pytest.approx(2.0)


def test_brain_mutes_on_consecutive_loss_streak():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(max_strategy_loss_streak=3, min_trades_for_eval=2), pf, clock=clock,
    )
    name = "delta_perp_scalper"

    # 3 losses in a row → mute trips.
    for _ in range(3):
        brain.on_trade_closed(name, -1.0)

    p = brain.perf(name)
    assert p.muted_until > clock.t
    assert p.probation is True
    assert p.size_mult == pytest.approx(0.5)


def test_brain_mutes_on_bad_rolling_win_rate():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            min_trades_for_eval=5, min_win_rate=0.6,
            max_strategy_loss_streak=999,  # disable streak gate
        ),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    # 2 wins, 3 losses = 40% win rate (below 60% floor) over 5 trades.
    for v in [+1, +1, -1, -1, -1]:
        brain.on_trade_closed(name, float(v))
    p = brain.perf(name)
    assert p.muted_until > clock.t


def test_brain_mute_expires_and_strategy_returns_on_probation():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            max_strategy_loss_streak=3, min_trades_for_eval=2,
            mute_seconds=60.0, probation_size_mult=0.5,
        ),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    for _ in range(3):
        brain.on_trade_closed(name, -1.0)
    p = brain.perf(name)
    assert p.muted_until > clock.t

    # Time passes → mute lifts on next filter_signals.
    clock.tick(120.0)
    market = _make_market("BTCUSD")
    sig = _make_signal(name, market_id="BTCUSD")
    # Warm up the regime detector so the signal isn't vetoed for UNKNOWN.
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": market})
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert len(out) == 1
    # Size was shrunk by the probation multiplier.
    assert out[0].legs[0].size_usd == pytest.approx(100.0 * 0.5)


def test_brain_probation_graduates_after_positive_trades():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            max_strategy_loss_streak=2, min_trades_for_eval=2,
            probation_trades=3, probation_size_mult=0.5,
        ),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    # Trip the mute
    brain.on_trade_closed(name, -1.0)
    brain.on_trade_closed(name, -1.0)
    p = brain.perf(name)
    assert p.probation is True
    assert p.size_mult == 0.5

    # 3 wins on probation → graduate back to 1.0
    for _ in range(3):
        brain.on_trade_closed(name, +1.0)
    assert p.probation is False
    assert p.size_mult == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AdaptiveBrain — signal filter (regime + mute + circuit breakers)
# ---------------------------------------------------------------------------


def test_brain_vetoes_signals_when_strategy_is_muted():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(max_strategy_loss_streak=3, min_trades_for_eval=2),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    for _ in range(3):
        brain.on_trade_closed(name, -1.0)

    market = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": market})

    sig = _make_signal(name, market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert out == []
    assert brain.stats.signals_vetoed_mute == 1


def test_brain_passes_reduce_only_legs_when_muted():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(max_strategy_loss_streak=3, min_trades_for_eval=2),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    for _ in range(3):
        brain.on_trade_closed(name, -1.0)

    market = _make_market("BTCUSD")
    close_sig = _make_signal(name, market_id="BTCUSD", side="SELL", reduce_only=True)
    out = brain.filter_signals([close_sig], {"BTCUSD": market})
    assert len(out) == 1, "closes must always flow even when muted"


def test_brain_hard_vetoes_mean_reversion_in_strong_uptrend_when_soft_veto_off():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            regime_short_window=5, regime_long_window=20,
            regime_trend_threshold=0.2,
            regime_news_tick_bps=200.0,  # avoid news veto on the warm-up
            regime_soft_veto=False,      # opt back into hard veto
        ),
        pf, clock=clock,
    )
    market = _make_market("BTCUSD", price=1000.0)
    base = 1000.0
    for i in range(40):
        base *= 1.0005
        outcome = next(iter(market.outcomes.values()))
        half = base * 5e-5
        outcome.book.replace(bids=[(base - half, 100.0)], asks=[(base + half, 100.0)])
        brain.regimes.observe_markets({"BTCUSD": market})

    assert brain.regimes.snapshot("BTCUSD").regime == Regime.TREND_UP

    sig = _make_signal("delta_perp_scalper", market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert out == []
    assert brain.stats.signals_vetoed_regime == 1


def test_brain_soft_vetoes_mean_reversion_in_strong_uptrend_by_default():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            regime_short_window=5, regime_long_window=20,
            regime_trend_threshold=0.2,
            regime_news_tick_bps=200.0,
            # soft veto on by default — wrong regime SHRINKS size
            wrong_regime_size_mult=0.25,
        ),
        pf, clock=clock,
    )
    market = _make_market("BTCUSD", price=1000.0)
    base = 1000.0
    for i in range(40):
        base *= 1.0005
        outcome = next(iter(market.outcomes.values()))
        half = base * 5e-5
        outcome.book.replace(bids=[(base - half, 100.0)], asks=[(base + half, 100.0)])
        brain.regimes.observe_markets({"BTCUSD": market})

    assert brain.regimes.snapshot("BTCUSD").regime == Regime.TREND_UP

    sig = _make_signal("delta_perp_scalper", market_id="BTCUSD", size_usd=100.0)
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert len(out) == 1, "soft veto should let the signal through at reduced size"
    assert out[0].legs[0].size_usd == pytest.approx(25.0)
    assert brain.stats.signals_shrunk == 1


def test_brain_allows_flow_scalp_in_strong_uptrend():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            regime_short_window=5, regime_long_window=20,
            regime_trend_threshold=0.2,
            regime_news_tick_bps=200.0,
        ),
        pf, clock=clock,
    )
    market = _make_market("BTCUSD", price=1000.0)
    base = 1000.0
    for i in range(40):
        base *= 1.0005
        outcome = next(iter(market.outcomes.values()))
        half = base * 5e-5
        outcome.book.replace(bids=[(base - half, 100.0)], asks=[(base + half, 100.0)])
        brain.regimes.observe_markets({"BTCUSD": market})

    assert brain.regimes.snapshot("BTCUSD").regime == Regime.TREND_UP

    sig = _make_signal("flow_scalp", market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert len(out) == 1


def test_brain_daily_loss_cap_vetoes_new_entries():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            daily_loss_pct=0.10,
            max_strategy_loss_streak=99,  # avoid mute interference
            min_trades_for_eval=99,
        ),
        pf, clock=clock,
    )
    # 24h PnL = -$15 on a $100 bankroll = -15% > 10% cap.
    brain.on_trade_closed("delta_perp_scalper", -15.0)

    market = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": market})

    sig = _make_signal("delta_perp_scalper", market_id="BTCUSD")
    close_sig = _make_signal(
        "delta_perp_scalper", market_id="BTCUSD", side="SELL", reduce_only=True
    )
    out = brain.filter_signals([sig, close_sig], {"BTCUSD": market})
    assert len(out) == 1
    assert all(leg.reduce_only for leg in out[0].legs)
    assert brain.stats.signals_vetoed_daily_loss == 1


def test_brain_daily_loss_window_evicts_old_pnls():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            daily_loss_pct=0.10,
            daily_window_seconds=10.0,  # tiny window so the test is quick
            max_strategy_loss_streak=99, min_trades_for_eval=99,
        ),
        pf, clock=clock,
    )
    brain.on_trade_closed("delta_perp_scalper", -20.0)
    assert brain._daily_pnls

    clock.tick(20.0)
    # filter_signals triggers the evict; force a call.
    market = _make_market("BTCUSD")
    brain.filter_signals([], {"BTCUSD": market})
    assert not brain._daily_pnls
    assert brain.stats.daily_pnl == 0.0


def test_brain_correlation_cap_vetoes_oversized_batch():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    # The cap = max_gross_exposure_mult × bankroll × leverage.
    # leverage = 25 (set on _make_signal's legs). With mult=0.1
    # the cap is 0.1 × 100 × 25 = $250. Two $200 trades = $400 > cap.
    brain = AdaptiveBrain(
        _cfg(
            max_gross_exposure_mult=0.1,
            min_trades_for_eval=99,
            max_strategy_loss_streak=99,
        ),
        pf, clock=clock,
    )
    btc = _make_market("BTCUSD")
    eth = _make_market("ETHUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc, "ETHUSD": eth})

    sigs = [
        _make_signal("delta_perp_scalper", market_id="BTCUSD", size_usd=200.0),
        _make_signal("delta_perp_scalper", market_id="ETHUSD", size_usd=200.0),
    ]
    out = brain.filter_signals(sigs, {"BTCUSD": btc, "ETHUSD": eth})
    assert len(out) == 1
    assert brain.stats.signals_vetoed_correlation == 1


def test_brain_correlation_cap_scales_with_leverage():
    """A leveraged perp ($100 bankroll, 25× lev) at mult=2.0 should
    let multiple concurrent $375 scalps through (cap = $5,000),
    not block the first one as the pre-fix code did."""
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            max_gross_exposure_mult=2.0,
            min_trades_for_eval=99,
            max_strategy_loss_streak=99,
        ),
        pf, clock=clock,
    )
    btc = _make_market("BTCUSD")
    eth = _make_market("ETHUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc, "ETHUSD": eth})
    # cap = 2.0 × 100 × 25 = $5,000. Three $375 trades = $1,125.
    sigs = [
        _make_signal("delta_perp_scalper", market_id="BTCUSD", size_usd=375.0),
        _make_signal("delta_perp_scalper", market_id="ETHUSD", size_usd=375.0),
        _make_signal("flow_scalp", market_id="BTCUSD", size_usd=375.0),
    ]
    out = brain.filter_signals(sigs, {"BTCUSD": btc, "ETHUSD": eth})
    assert len(out) == 3, "all three should fit under the $5,000 cap"
    assert brain.stats.signals_vetoed_correlation == 0


def test_brain_disabled_passes_everything_through():
    pf = Portfolio(bankroll=100.0)
    brain = AdaptiveBrain(_cfg(enabled=False), pf, clock=_FixedClock())
    # Force a loss streak that would normally mute.
    for _ in range(10):
        brain.on_trade_closed("delta_perp_scalper", -1.0)

    market = _make_market("BTCUSD")
    sig = _make_signal("delta_perp_scalper", market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert len(out) == 1
    assert out[0].legs[0].size_usd == pytest.approx(100.0)


def test_brain_shrinks_size_in_high_vol_regime():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            regime_short_window=5, regime_long_window=40,
            regime_high_vol_ratio=2.0,
            regime_trend_threshold=10.0,    # disable trend classifier
            regime_news_tick_bps=200.0,     # disable news veto
            high_vol_size_mult=0.25,
        ),
        pf, clock=clock,
    )
    market = _make_market("BTCUSD", price=1000.0)
    outcome = next(iter(market.outcomes.values()))

    # Long-window calm, then short-window spike, both non-trending.
    base = 1000.0
    for i in range(45):
        nxt = base + (0.001 if i % 2 == 0 else -0.001)
        outcome.book.replace(
            bids=[(nxt - 0.05, 100.0)], asks=[(nxt + 0.05, 100.0)]
        )
        brain.regimes.observe_markets({"BTCUSD": market})
    for i in range(8):
        nxt = base + (1.0 if i % 2 == 0 else -1.0)
        outcome.book.replace(
            bids=[(nxt - 0.05, 100.0)], asks=[(nxt + 0.05, 100.0)]
        )
        brain.regimes.observe_markets({"BTCUSD": market})

    assert brain.regimes.snapshot("BTCUSD").regime == Regime.HIGH_VOL

    sig = _make_signal("delta_perp_scalper", market_id="BTCUSD", size_usd=100.0)
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert len(out) == 1
    assert out[0].legs[0].size_usd == pytest.approx(25.0)
    assert brain.stats.signals_shrunk == 1


def test_brain_shrinks_size_on_loss_streak_even_when_not_muted():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            max_strategy_loss_streak=99,  # don't mute
            min_trades_for_eval=99,
        ),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    # 3 losses → consecutive_losses = 3 → size shrunk by 1/(1+0.5*2)=0.5
    for _ in range(3):
        brain.on_trade_closed(name, -1.0)

    market = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": market})

    sig = _make_signal(name, market_id="BTCUSD", size_usd=100.0)
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert len(out) == 1
    # 1 / (1 + 0.5 × (3 - 1)) = 1 / 2 = 0.5
    assert out[0].legs[0].size_usd == pytest.approx(50.0)


def test_brain_tags_vetoed_signals_with_reason():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(max_strategy_loss_streak=3, min_trades_for_eval=2),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    for _ in range(3):
        brain.on_trade_closed(name, -1.0)

    market = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": market})

    sig = _make_signal(name, market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert out == []
    # The original signal object was tagged with the veto reason in
    # metadata so the engine can surface it via the dashboard.
    assert sig.metadata.get("brain_vetoed") is True
    assert "mute" in sig.metadata.get("brain_veto_reason", "").lower()


def test_strategy_phantom_position_state_is_reconciled_with_portfolio():
    """If the strategy thinks it has a position but the portfolio is
    flat (e.g. the brain vetoed the entry), the next ``scan()`` must
    clear internal state so we don't fire phantom TP/SL closes
    forever."""
    from aera.strategies import DeltaPerpetualScalper

    pf = Portfolio(bankroll=100.0)
    strat = DeltaPerpetualScalper(
        zscore_window=5, zscore_entry=99.0,  # entry gate effectively off
        take_profit_pct=0.0, stop_loss_pct=0.0,
        take_profit_usd=5.0, stop_loss_usd=3.0,
        portfolio=pf,
    )
    market = _make_market("BTCUSD", price=1000.0)
    # Manually install phantom internal state simulating a previously
    # vetoed entry: the strategy "thinks" it's LONG at 1000 but the
    # portfolio holds nothing.
    st = strat._state_for(market.id)
    st.position_side = "LONG"
    st.entry_mid = 1000.0
    st.entry_size_usd = 100.0

    # First scan triggers reconciliation — state should reset to flat.
    strat.scan([market])
    assert st.position_side is None
    assert st.entry_mid == 0.0


def test_brain_news_spike_vetoes_even_flow_scalp():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            regime_short_window=5, regime_long_window=20,
            regime_news_tick_bps=20.0,
        ),
        pf, clock=clock,
    )
    market = _make_market("BTCUSD", price=1000.0)
    outcome = next(iter(market.outcomes.values()))

    # Warm up
    for i in range(15):
        nxt = 1000.0 + i * 0.01
        outcome.book.replace(
            bids=[(nxt - 0.05, 100.0)], asks=[(nxt + 0.05, 100.0)]
        )
        brain.regimes.observe_markets({"BTCUSD": market})
    # Single tick spike of 50 bps
    outcome.book.replace(bids=[(1005.0, 100.0)], asks=[(1005.1, 100.0)])
    brain.regimes.observe_markets({"BTCUSD": market})
    assert brain.regimes.snapshot("BTCUSD").regime == Regime.NEWS_SPIKE

    sig = _make_signal("flow_scalp", market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": market})
    assert out == []
    assert brain.stats.signals_vetoed_regime == 1


# ---------------------------------------------------------------------------
# post-loss cool-down (per strategy × symbol)
# ---------------------------------------------------------------------------


def test_post_loss_cooldown_blocks_same_strategy_on_same_symbol():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            post_loss_cooldown_seconds=60.0,
            max_strategy_loss_streak=99,  # don't auto-mute, isolate the cool-down
            min_trades_for_eval=99,
        ),
        pf, clock=clock,
    )
    btc = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc})

    brain.on_trade_closed("flow_scalp", -1.0, symbol="BTCUSD")

    sig = _make_signal("flow_scalp", market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": btc})
    assert out == [], "same (strategy, symbol) within cool-down → blocked"
    assert brain.stats.signals_vetoed_post_loss == 1
    assert "post-loss" in sig.metadata.get("brain_veto_reason", "").lower()


def test_post_loss_cooldown_does_not_block_other_symbol():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            post_loss_cooldown_seconds=60.0,
            max_strategy_loss_streak=99,
            min_trades_for_eval=99,
        ),
        pf, clock=clock,
    )
    btc = _make_market("BTCUSD")
    eth = _make_market("ETHUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc, "ETHUSD": eth})

    brain.on_trade_closed("flow_scalp", -1.0, symbol="BTCUSD")

    sig = _make_signal("flow_scalp", market_id="ETHUSD")
    out = brain.filter_signals([sig], {"BTCUSD": btc, "ETHUSD": eth})
    assert len(out) == 1, "different symbol should not be blocked"


def test_post_loss_cooldown_expires():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock(t=1000.0)
    brain = AdaptiveBrain(
        _cfg(
            post_loss_cooldown_seconds=60.0,
            max_strategy_loss_streak=99,
            min_trades_for_eval=99,
        ),
        pf, clock=clock,
    )
    btc = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc})

    brain.on_trade_closed("flow_scalp", -1.0, symbol="BTCUSD")
    clock.tick(61.0)

    sig = _make_signal("flow_scalp", market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": btc})
    assert len(out) == 1, "cool-down should expire after the configured window"


def test_winning_close_does_not_arm_cooldown():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(post_loss_cooldown_seconds=60.0),
        pf, clock=clock,
    )
    btc = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc})

    brain.on_trade_closed("flow_scalp", +1.0, symbol="BTCUSD")

    sig = _make_signal("flow_scalp", market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": btc})
    assert len(out) == 1
    assert brain.stats.signals_vetoed_post_loss == 0


def test_daily_loss_veto_tags_signal_with_reason():
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock(t=1000.0)
    brain = AdaptiveBrain(
        _cfg(daily_loss_pct=0.05),  # cap = -$5 on $100 bankroll
        pf, clock=clock,
    )
    btc = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc})

    brain.on_trade_closed("flow_scalp", -10.0)  # blows past the cap

    sig = _make_signal("flow_scalp", market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": btc})
    assert out == []
    reason = sig.metadata.get("brain_veto_reason", "")
    assert "daily loss" in reason.lower(), reason


# ---------------------------------------------------------------------------
# Cost-aware edge gate
# ---------------------------------------------------------------------------


def test_cost_aware_edge_gate_vetoes_below_fee_floor():
    """Regression for the "81 % loss rate" failure mode.

    Four of the seven strategies shipped with ``min_edge`` below the
    venue's 10 bps round-trip taker fee, so every fire at the strategy's
    low-edge tail was a guaranteed loss before slippage even hit. The
    brain now enforces a global floor independent of strategy
    configuration: edge minus round-trip fees minus assumed slippage
    must clear ``min_edge_after_costs_bps``.

    Concrete numbers: taker_fee_bps=5 (Delta), RT legs=2 → 10 bps RT fee.
    Slippage assumed 2 bps. With ``min_edge_after_costs_bps=5`` the
    required gross edge is 17 bps. A 10 bps signal must be vetoed; a
    20 bps signal must pass.
    """
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            min_edge_after_costs_bps=5.0,
            cost_round_trip_legs=2,
            cost_assumed_slippage_bps=2.0,
            min_trades_for_eval=999,  # disable mute logic
        ),
        pf, clock=clock, taker_fee_bps=5.0,
    )
    btc = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc})

    # 10 bps edge — fails (10 - 10 - 2 = -2 < 5).
    sig_low = _make_signal("flow_scalp", market_id="BTCUSD")
    sig_low.edge = 0.0010
    # 20 bps edge — passes (20 - 10 - 2 = 8 ≥ 5).
    sig_high = _make_signal("delta_perp_scalper", market_id="BTCUSD")
    sig_high.edge = 0.0020

    out = brain.filter_signals([sig_low, sig_high], {"BTCUSD": btc})
    out_strategies = [s.strategy for s in out]
    assert "delta_perp_scalper" in out_strategies, (
        "high-edge signal must pass the cost gate"
    )
    assert "flow_scalp" not in out_strategies, (
        "low-edge signal must be vetoed by the cost gate"
    )
    veto_reason = sig_low.metadata.get("brain_veto_reason", "")
    assert "edge" in veto_reason.lower() and "fee" in veto_reason.lower(), veto_reason
    assert brain.stats.signals_vetoed_cost == 1


def test_cost_aware_edge_gate_disabled_by_zero_floor():
    """Setting ``min_edge_after_costs_bps=0`` reverts to legacy behaviour
    where any non-zero edge passes the cost gate."""
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            min_edge_after_costs_bps=0.0,
            min_trades_for_eval=999,
        ),
        pf, clock=clock, taker_fee_bps=5.0,
    )
    btc = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc})

    sig = _make_signal("flow_scalp", market_id="BTCUSD")
    sig.edge = 0.0001  # 1 bp — would be vetoed under any positive floor
    out = brain.filter_signals([sig], {"BTCUSD": btc})
    assert len(out) == 1
    assert brain.stats.signals_vetoed_cost == 0


def test_lifetime_pnl_kill_switch_permanently_mutes_bleeder():
    """Regression for the mute → probation → mute → probation cycle.

    Without the kill switch, a structurally broken strategy can keep
    cycling through the rolling-window mute forever: each round of
    probation lets it bleed more, the size_mult shrink doesn't stop
    losses (it just shrinks them), and after ``mute_seconds`` the
    cycle restarts. The lifetime PnL kill switch ends the loop —
    once cumulative bleed exceeds the floor, the strategy is
    permanently muted for the rest of the process.
    """
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            lifetime_pnl_kill_floor_usd=-3.0,
            max_strategy_loss_streak=999,  # disable rolling mute
            min_trades_for_eval=999,
            min_edge_after_costs_bps=0.0,
        ),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    # Three losing trades of -$1 each → cumulative -$3 ≤ floor -$3.
    for _ in range(3):
        brain.on_trade_closed(name, -1.0)
    perf = brain.perf(name)
    assert perf.killed is True, (
        f"strategy should be killed after lifetime PnL "
        f"${perf.total_pnl:.2f} ≤ floor -$3.00"
    )

    # Even AFTER waiting longer than any mute window, a signal stays
    # vetoed because killed=True. (Verifies the kill is not just a
    # huge mute_until that auto-expires.)
    clock.t += 10_000_000.0
    btc = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc})
    sig = _make_signal(name, market_id="BTCUSD")
    out = brain.filter_signals([sig], {"BTCUSD": btc})
    assert out == []
    reason = sig.metadata.get("brain_veto_reason", "")
    assert "killed" in reason.lower(), reason


def test_lifetime_kill_does_not_fire_on_winning_strategy():
    """A strategy that's net-positive (or just slightly negative but
    above the floor) must NOT be killed. Kill switch only fires when
    cumulative PnL drops at or below the configured floor."""
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            lifetime_pnl_kill_floor_usd=-3.0,
            max_strategy_loss_streak=999,
            min_trades_for_eval=999,
            min_edge_after_costs_bps=0.0,
        ),
        pf, clock=clock,
    )
    name = "tick_reversal_scalp"
    # Two losses then a big win → cumulative still positive.
    brain.on_trade_closed(name, -1.0)
    brain.on_trade_closed(name, -1.0)
    brain.on_trade_closed(name, +5.0)
    assert brain.perf(name).killed is False
    assert brain.perf(name).total_pnl == pytest.approx(3.0)


def test_lifetime_kill_disabled_when_floor_is_zero():
    """``lifetime_pnl_kill_floor_usd=0`` (or any non-negative value)
    disables the kill switch entirely. Useful for backtests where
    you want to measure raw strategy expectancy without the gate."""
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            lifetime_pnl_kill_floor_usd=0.0,
            max_strategy_loss_streak=999,
            min_trades_for_eval=999,
            min_edge_after_costs_bps=0.0,
        ),
        pf, clock=clock,
    )
    name = "delta_perp_scalper"
    for _ in range(20):
        brain.on_trade_closed(name, -1.0)
    assert brain.perf(name).killed is False
    assert brain.perf(name).total_pnl == pytest.approx(-20.0)


def test_cost_aware_edge_gate_lets_reduce_only_closes_through():
    """Closing legs have no notion of edge and must always flow,
    regardless of the cost floor or any signal-level edge value."""
    pf = Portfolio(bankroll=100.0)
    clock = _FixedClock()
    brain = AdaptiveBrain(
        _cfg(
            min_edge_after_costs_bps=100.0,  # extreme floor
            min_trades_for_eval=999,
        ),
        pf, clock=clock, taker_fee_bps=5.0,
    )
    btc = _make_market("BTCUSD")
    for _ in range(10):
        brain.regimes.observe_markets({"BTCUSD": btc})

    close = _make_signal("greedy", market_id="BTCUSD", reduce_only=True)
    close.edge = 0.0  # closes don't carry edge
    out = brain.filter_signals([close], {"BTCUSD": btc})
    assert len(out) == 1
    assert brain.stats.signals_vetoed_cost == 0
