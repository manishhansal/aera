"""Monte-Carlo compounding simulator.

Models the bot as a sequence of independent bets, each with:
    - win probability `p`
    - net payoff +b on win, -1 on loss (so a coin-flip with edge has p > 0.5 and b = 1)
    - Kelly-sized stake based on (p, b) and the *live* bankroll

The simulator returns the distribution of terminal bankrolls and the median path,
so you can see how realistic the "$1 -> $1M" claim is for a given edge profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from aera.logging import get_logger


log = get_logger(__name__)


@dataclass
class GrowthResult:
    starting_bankroll: float
    target: float
    n_trades: int
    win_rate: float
    payoff_ratio: float
    kelly_fraction: float

    terminal_bankrolls: np.ndarray         # shape (runs,)
    p_hit_target: float                    # probability of reaching target
    median_path: np.ndarray                # shape (n_trades + 1,)
    p5_path: np.ndarray
    p95_path: np.ndarray
    expected_log_growth_per_trade: float
    required_edge_per_trade_for_target: float


def _full_kelly(p: float, b: float) -> float:
    """Full-Kelly fraction for win prob `p`, payoff `b:1`."""
    q = 1.0 - p
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - q) / b)


def simulate_growth(
    *,
    starting_bankroll: float = 1.0,
    target: float = 1_000_000.0,
    n_trades: int = 37_000,
    win_rate: float = 0.535,
    payoff_ratio: float = 1.0,
    kelly_fraction: float = 0.25,
    runs: int = 1000,
    min_bankroll: float = 1e-4,
    seed: Optional[int] = 42,
) -> GrowthResult:
    """Run `runs` Monte-Carlo paths and aggregate.

    Args:
        starting_bankroll: e.g. 1.0
        target: e.g. 1_000_000.0
        n_trades: number of bets per path
        win_rate: per-bet win probability
        payoff_ratio: b in b:1 payoff (1.0 = symmetric)
        kelly_fraction: 0..1 multiplier on full Kelly (0.25 = quarter Kelly)
        runs: monte-carlo paths
        min_bankroll: stop a path if bankroll falls below this (treat as ruin)
    """
    rng = np.random.default_rng(seed)
    p = win_rate
    b = payoff_ratio
    f = _full_kelly(p, b) * kelly_fraction

    # Each bet, bankroll evolves multiplicatively:
    #   win:  B *= (1 + f * b)
    #   loss: B *= (1 - f)
    log_gain_win = np.log(1.0 + f * b) if 1.0 + f * b > 0 else -np.inf
    log_gain_loss = np.log(1.0 - f) if 1.0 - f > 0 else -np.inf
    expected_log_growth = p * log_gain_win + (1.0 - p) * log_gain_loss

    # required average geometric growth per trade to hit target:
    required_g = (target / starting_bankroll) ** (1.0 / n_trades) - 1.0

    bankrolls = np.full(runs, starting_bankroll, dtype=np.float64)
    paths = np.empty((runs, n_trades + 1), dtype=np.float64)
    paths[:, 0] = starting_bankroll

    for t in range(1, n_trades + 1):
        wins = rng.random(runs) < p
        multipliers = np.where(wins, 1.0 + f * b, 1.0 - f)
        bankrolls = bankrolls * multipliers
        # ruin
        ruined = bankrolls < min_bankroll
        bankrolls = np.where(ruined, 0.0, bankrolls)
        paths[:, t] = bankrolls

    hit = (paths.max(axis=1) >= target).mean() if target > 0 else 0.0
    median_path = np.median(paths, axis=0)
    p5_path = np.quantile(paths, 0.05, axis=0)
    p95_path = np.quantile(paths, 0.95, axis=0)

    return GrowthResult(
        starting_bankroll=starting_bankroll,
        target=target,
        n_trades=n_trades,
        win_rate=win_rate,
        payoff_ratio=payoff_ratio,
        kelly_fraction=kelly_fraction,
        terminal_bankrolls=bankrolls,
        p_hit_target=float(hit),
        median_path=median_path,
        p5_path=p5_path,
        p95_path=p95_path,
        expected_log_growth_per_trade=float(expected_log_growth),
        required_edge_per_trade_for_target=float(required_g),
    )


def trades_needed_for_target(
    *, starting_bankroll: float, target: float, edge_per_trade: float
) -> float:
    """Deterministic answer: how many trades at average edge `g` to reach `target`."""
    if edge_per_trade <= 0:
        return float("inf")
    import math
    return math.log(target / starting_bankroll) / math.log(1.0 + edge_per_trade)
