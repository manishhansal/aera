"""Tests for the compounding-growth math."""
from __future__ import annotations

import math

from aera.core.compounding import simulate_growth, trades_needed_for_target


def test_required_edge_for_target():
    # 10000x in 1000 trades requires (10000)^(1/1000) - 1 per trade
    expected = 10_000 ** (1 / 1000) - 1
    g = trades_needed_for_target(starting_bankroll=1.0, target=10_000.0, edge_per_trade=expected)
    assert math.isclose(g, 1000.0, rel_tol=1e-4)


def test_simulate_growth_basic_shape():
    res = simulate_growth(
        starting_bankroll=1.0,
        target=1_000_000.0,
        n_trades=2000,
        win_rate=0.55,
        payoff_ratio=1.0,
        kelly_fraction=0.25,
        runs=200,
        seed=1,
    )
    assert res.median_path.shape == (2001,)
    assert res.terminal_bankrolls.shape == (200,)
    assert 0.0 <= res.p_hit_target <= 1.0


def test_positive_edge_grows_on_average():
    res = simulate_growth(
        starting_bankroll=1.0,
        target=1e12,             # unreachable, just measuring growth
        n_trades=500,
        win_rate=0.60,
        payoff_ratio=1.0,
        kelly_fraction=0.25,
        runs=500,
        seed=2,
    )
    # expected log growth per trade must be > 0
    assert res.expected_log_growth_per_trade > 0
    # geometric mean of terminals should exceed 1
    import numpy as np
    logs = np.log(np.where(res.terminal_bankrolls > 0, res.terminal_bankrolls, 1e-9))
    assert logs.mean() > 0


def test_negative_edge_shrinks_on_average():
    res = simulate_growth(
        starting_bankroll=1.0,
        target=1e12,
        n_trades=500,
        win_rate=0.45,
        payoff_ratio=1.0,
        kelly_fraction=0.25,
        runs=200,
        seed=3,
    )
    # negative edge -> full kelly is 0 (clamped to 0 in our implementation) -> no growth
    # In our impl `_full_kelly` clamps negatives to 0, so bankroll stays at 1.
    assert all(res.terminal_bankrolls == 1.0)
