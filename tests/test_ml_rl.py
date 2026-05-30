"""RL agent — env semantics, DQN training smoke, scorer integration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aera.ml import FEATURE_COLUMNS
from aera.ml.rl import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    DQNAgent,
    DQNConfig,
    N_OBS,
    POS_FLAT,
    POS_LONG,
    POS_SHORT,
    RLScorer,
    TradingEnv,
    TradingEnvConfig,
    load_rl_scorer,
    train_dqn_agent,
)


def _toy_candles(n: int = 200, drift: float = 0.001, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic OHLCV series with a tunable drift."""
    rng = np.random.default_rng(seed)
    ts0 = 1_700_000_000
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift + rng.normal(0, 0.002)))
    closes = np.array(closes)
    opens = np.roll(closes, 1); opens[0] = closes[0]
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.001, n)))
    vols = np.abs(rng.normal(100, 20, n))
    return pd.DataFrame({
        "ts":    [ts0 + i * 60 for i in range(n)],
        "open":  opens,
        "high":  highs,
        "low":   lows,
        "close": closes,
        "volume": vols,
    })


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def test_env_reset_returns_obs_of_correct_shape():
    env = TradingEnv(_toy_candles(100))
    obs = env.reset()
    assert obs.shape == (N_OBS,)
    assert obs.dtype == np.float32


def test_env_hold_doesnt_open_position():
    env = TradingEnv(_toy_candles(100))
    env.reset()
    obs, r, done, info = env.step(ACTION_HOLD)
    assert info["position"] == POS_FLAT
    assert env.trade_log == []


def test_env_buy_then_hold_opens_long():
    env = TradingEnv(_toy_candles(100, drift=0.0))
    env.reset()
    obs, r, done, info = env.step(ACTION_BUY)
    assert info["position"] == POS_LONG
    obs, r, done, info = env.step(ACTION_HOLD)
    assert info["position"] == POS_LONG


def test_env_buy_then_sell_closes_with_realised_pnl():
    env = TradingEnv(_toy_candles(100))
    env.reset()
    env.step(ACTION_BUY)
    # Force enough holds for a realised PnL to accumulate.
    for _ in range(5):
        env.step(ACTION_HOLD)
    obs, r, done, info = env.step(ACTION_SELL)
    # Flipping long→short closes the long and opens a short
    assert info["position"] == POS_SHORT
    assert len(env.trade_log) == 1


def test_env_max_hold_forces_close():
    cfg = TradingEnvConfig(max_hold_bars=3)
    env = TradingEnv(_toy_candles(200), config=cfg)
    env.reset()
    env.step(ACTION_BUY)
    for _ in range(5):
        env.step(ACTION_HOLD)
    # Position must have been force-flattened by max_hold_bars=3
    assert len(env.trade_log) >= 1
    assert env.trade_log[0]["reason_close"] == "max-hold timeout"


def test_env_terminal_force_flatten_on_done():
    env = TradingEnv(_toy_candles(20))
    env.reset()
    env.step(ACTION_BUY)
    done = False
    while not done:
        _, _, done, _ = env.step(ACTION_HOLD)
    # End of data should leave us flat.
    assert env._pos.side == POS_FLAT
    assert any(t["reason_close"] == "end-of-data flatten" for t in env.trade_log)


def test_env_rejects_empty_candles():
    with pytest.raises(ValueError):
        TradingEnv(pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]))


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------


def test_dqn_agent_select_action_in_range():
    agent = DQNAgent(n_obs=N_OBS, seed=0)
    obs = np.zeros(N_OBS, dtype=np.float32)
    a = agent.select_action(obs, greedy=True)
    assert a in (ACTION_HOLD, ACTION_BUY, ACTION_SELL)


def test_dqn_agent_policy_proba_normalises_to_one():
    agent = DQNAgent(n_obs=N_OBS, seed=0)
    obs = np.zeros(N_OBS, dtype=np.float32)
    p = agent.policy_proba(obs)
    assert p.shape == (3,)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)


def test_dqn_agent_smoke_train_runs_and_persists(tmp_path):
    candles = _toy_candles(100, drift=0.001, seed=1)
    cfg = DQNConfig(hidden=8, epsilon_decay_steps=200, warmup_steps=10)
    agent, stats = train_dqn_agent(candles, episodes=2, agent_config=cfg, seed=0)
    assert stats.episodes == 2
    assert stats.total_steps > 0

    out = tmp_path / "policy.npz"
    agent.save(out)
    assert out.exists()

    loaded = DQNAgent.load(out)
    # Forward pass should reproduce the trained weights exactly.
    obs = np.zeros(N_OBS, dtype=np.float32)
    assert np.allclose(loaded.q.forward(obs), agent.q.forward(obs))


# ---------------------------------------------------------------------------
# RLScorer (registry adapter)
# ---------------------------------------------------------------------------


def test_rl_scorer_with_no_agent_returns_neutral():
    scorer = RLScorer(agent=None)
    val = scorer.predict_proba_win(np.zeros(len(FEATURE_COLUMNS), dtype=np.float32))
    assert val == pytest.approx(0.5)


def test_rl_scorer_with_agent_returns_unit_interval(tmp_path):
    candles = _toy_candles(80, drift=0.001, seed=2)
    cfg = DQNConfig(hidden=8, epsilon_decay_steps=100, warmup_steps=10)
    agent, _ = train_dqn_agent(candles, episodes=1, agent_config=cfg, seed=0)
    scorer = RLScorer(agent=agent)
    feats = pd.Series({c: 0.0 for c in FEATURE_COLUMNS})
    val = scorer.predict_proba_win(feats)
    assert 0.0 <= val <= 1.0


def test_load_rl_scorer_round_trip(tmp_path):
    candles = _toy_candles(80, drift=0.001, seed=3)
    cfg = DQNConfig(hidden=8, epsilon_decay_steps=100, warmup_steps=10)
    agent, _ = train_dqn_agent(candles, episodes=1, agent_config=cfg, seed=0)
    out = tmp_path / "rl_policy.npz"
    agent.save(out)
    loaded = load_rl_scorer(out)
    assert isinstance(loaded, RLScorer)
    feats = pd.Series({c: 0.0 for c in FEATURE_COLUMNS})
    val = loaded.predict_proba_win(feats)
    assert 0.0 <= val <= 1.0
