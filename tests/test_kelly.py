"""Sanity tests for the Kelly criterion implementation."""
from __future__ import annotations

import math

from aera.core.risk import kelly_fraction, fractional_kelly_bet
from aera.settings import RiskConfig


def test_kelly_zero_edge():
    # if true_prob == market_price you have no edge -> 0 stake
    assert kelly_fraction(0.50, 0.50) == 0.0


def test_kelly_known_value():
    # classic example: p = 0.6, payoff b = 1 -> f* = 2p - 1 = 0.2
    p, price = 0.60, 0.50
    f = kelly_fraction(p, price)
    assert math.isclose(f, 0.2, abs_tol=1e-9)


def test_kelly_negative_edge():
    # p < price => bet NO; full kelly returns negative fraction
    f = kelly_fraction(0.40, 0.55)
    assert f < 0


def test_kelly_invalid_inputs():
    assert kelly_fraction(0.5, 0.0) == 0.0
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert kelly_fraction(1.5, 0.5) == 0.0


def test_fractional_kelly_respects_cap():
    cfg = RiskConfig(kelly_fraction=1.0, max_trade_fraction=0.05)
    # Full kelly here would be 0.2 of bankroll, but cap is 0.05
    stake = fractional_kelly_bet(0.60, 0.50, bankroll=100.0, risk=cfg)
    assert math.isclose(stake, 5.0, abs_tol=1e-9)


def test_fractional_kelly_quarter():
    cfg = RiskConfig(kelly_fraction=0.25, max_trade_fraction=1.0)
    # 0.2 * 0.25 = 0.05 of bankroll
    stake = fractional_kelly_bet(0.60, 0.50, bankroll=200.0, risk=cfg)
    assert math.isclose(stake, 10.0, abs_tol=1e-9)
