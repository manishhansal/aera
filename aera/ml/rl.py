"""Reinforcement-learning agent for adaptive trading policy.

The classifier in :mod:`aera.ml.model` answers a per-trade
"will this win?" question; this module learns a *policy* — at each
bar, choose one of ``{HOLD, BUY, SELL}`` so as to maximise
discounted future PnL. The agent sees the same 15-feature vector
that the GBT does, plus a one-hot of the current position state
(flat / long / short), and is rewarded with the marked-to-market
PnL delta on every step plus a terminal realised-PnL settlement
when a position is closed.

Design choices
==============

* **Numpy-only DQN.** No torch needed. The Q-network is a tiny
  2-layer MLP (~3k parameters) trained with experience replay +
  a soft-updated target network. This is fast enough to converge
  on a single laptop CPU in minutes, and keeps the RL component
  ``pip install``-friendly with zero new dependencies.
* **Reward shaping.** Per-step reward = unrealised-PnL delta +
  small holding penalty + large realised-PnL bonus on close. This
  discourages whip-saw entries and prefers trades that build PnL
  monotonically.
* **Episode = one historical replay.** The trainer runs the agent
  through a full candle history, repeatedly. Each episode starts
  flat at the first bar and ends on a forced flatten at the last.

Tests cover the env's step semantics + a tiny smoke training run
that confirms the Q-network learns *something* on a deterministic
toy series. The agent's actual edge on real Delta data is for the
user to validate by running :mod:`scripts.train_rl_agent` against
their sweep cache.

Integration with the money printer
----------------------------------

When the trained policy file is present on disk
(``data/money_printer/rl_policy.npz``), :class:`MoneyPrinter`
loads it as one of several scorers via :mod:`aera.ml.registry`. The
RL agent contributes a "BUY confidence" derived from its Q-values
that gets fused with the GBT and ensemble probabilities into the
final fire / no-fire decision.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import pandas as pd

from aera.logging import get_logger

from .features import FEATURE_COLUMNS, extract_features

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# action / state space
# ---------------------------------------------------------------------------


ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
N_ACTIONS = 3

POS_FLAT = 0
POS_LONG = 1
POS_SHORT = -1


# Observation = FEATURE_COLUMNS + 3 one-hot position bits + 1 unrealised-PnL
# scalar. We keep the position info as a one-hot so the Q-net can learn
# "don't BUY again if already long" without numeric ambiguity.
EXTRA_DIMS = 4   # one-hot (flat, long, short) + unreal_pnl_pct
N_OBS = len(FEATURE_COLUMNS) + EXTRA_DIMS


def _position_block(side: int, unreal_pnl_pct: float) -> np.ndarray:
    block = np.zeros(EXTRA_DIMS, dtype=np.float32)
    if side == POS_FLAT:
        block[0] = 1.0
    elif side == POS_LONG:
        block[1] = 1.0
    elif side == POS_SHORT:
        block[2] = 1.0
    # Clip to ±1 so unrealised PnL doesn't blow up the linear layer's
    # input distribution on a runaway trade.
    block[3] = float(np.clip(unreal_pnl_pct, -1.0, 1.0))
    return block


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


@dataclass
class TradingEnvConfig:
    fee_bps: float = 5.0                # taker fee on each leg
    slippage_bps: float = 2.0
    hold_penalty: float = 0.0001        # per-step cost while in a position
    max_hold_bars: int = 60             # force-flatten window
    notional_usd: float = 100.0
    leverage: float = 1.0
    realise_bonus_mult: float = 1.0     # extra weight on realised PnL on close


@dataclass
class _EnvPosition:
    side: int = POS_FLAT
    entry_price: float = 0.0
    entry_bar: int = -1


class TradingEnv:
    """A minimal Gym-style environment over a candle DataFrame.

    The agent observes (FEATURE_COLUMNS + position state) at each
    bar; it chooses HOLD / BUY / SELL and receives a reward equal to
    the unrealised-PnL delta plus a closing-PnL settlement when a
    round-trip ends.
    """

    def __init__(self, candles: pd.DataFrame, config: Optional[TradingEnvConfig] = None) -> None:
        if candles is None or candles.empty:
            raise ValueError("TradingEnv requires non-empty candles")
        self.config = config or TradingEnvConfig()
        self.candles = candles.reset_index(drop=True)
        self.features = extract_features(self.candles)
        self.n_bars = len(self.candles)
        self._cursor = 0
        self._pos = _EnvPosition()
        self._last_mark: float = 0.0
        self._trade_log: List[dict] = []

    # ------------------------------------------------------------------
    # gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self._cursor = 0
        self._pos = _EnvPosition()
        self._last_mark = float(self.candles.iloc[0]["close"])
        self._trade_log.clear()
        return self._observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Advance one bar. Returns ``(obs, reward, done, info)``."""
        reward = 0.0
        info: dict = {}

        bar = self.candles.iloc[self._cursor]
        price = float(bar["close"])

        # ---- apply action ------------------------------------------
        if action == ACTION_BUY:
            if self._pos.side == POS_SHORT:
                reward += self._close_position(price, "agent flip→long") * self.config.realise_bonus_mult
            if self._pos.side == POS_FLAT:
                self._open_position(POS_LONG, price)
        elif action == ACTION_SELL:
            if self._pos.side == POS_LONG:
                reward += self._close_position(price, "agent flip→short") * self.config.realise_bonus_mult
            if self._pos.side == POS_FLAT:
                self._open_position(POS_SHORT, price)
        # HOLD: nothing

        # ---- per-step reward = mark-to-market PnL delta ------------
        unreal_now = self._unreal_pnl(price)
        unreal_prev = self._unreal_pnl(self._last_mark)
        reward += unreal_now - unreal_prev
        if self._pos.side != POS_FLAT:
            reward -= self.config.hold_penalty
            # Forced flatten after max_hold_bars to keep episodes
            # bounded in worst-case trade length.
            if self._cursor - self._pos.entry_bar >= self.config.max_hold_bars:
                reward += self._close_position(price, "max-hold timeout") * self.config.realise_bonus_mult

        self._last_mark = price
        self._cursor += 1
        done = self._cursor >= self.n_bars
        if done and self._pos.side != POS_FLAT:
            reward += self._close_position(price, "end-of-data flatten") * self.config.realise_bonus_mult

        info["position"] = self._pos.side
        info["bar"] = self._cursor - 1
        info["mid"] = price
        return self._observation(), float(reward), done, info

    # ------------------------------------------------------------------
    # position management
    # ------------------------------------------------------------------

    def _open_position(self, side: int, price: float) -> None:
        # Apply slippage to entry mid so the env's PnL aligns with the
        # paper exchange's expected fills.
        slip = price * self.config.slippage_bps / 1e4
        entry = price + (slip if side == POS_LONG else -slip)
        self._pos = _EnvPosition(side=side, entry_price=entry, entry_bar=self._cursor)

    def _close_position(self, price: float, reason: str) -> float:
        slip = price * self.config.slippage_bps / 1e4
        fill = price - (slip if self._pos.side == POS_LONG else -slip)
        pnl = self._raw_pnl(fill)
        fees = self.config.notional_usd * (self.config.fee_bps / 1e4) * 2.0
        net = pnl - fees
        self._trade_log.append({
            "side": "LONG" if self._pos.side == POS_LONG else "SHORT",
            "entry_bar": self._pos.entry_bar,
            "exit_bar": self._cursor,
            "entry_price": self._pos.entry_price,
            "exit_price": fill,
            "pnl_usd": net,
            "reason_close": reason,
        })
        self._pos = _EnvPosition()
        # Normalise the bonus to a per-unit-of-bankroll scale so reward
        # magnitudes are comparable across notional sizes.
        return float(net / max(1.0, self.config.notional_usd))

    def _raw_pnl(self, fill_price: float) -> float:
        if self._pos.side == POS_FLAT or self._pos.entry_price <= 0:
            return 0.0
        notional = self.config.notional_usd
        if self._pos.side == POS_LONG:
            return notional * (fill_price - self._pos.entry_price) / self._pos.entry_price
        return notional * (self._pos.entry_price - fill_price) / self._pos.entry_price

    def _unreal_pnl(self, mark: float) -> float:
        """Unrealised PnL as a fraction of notional (so reward scale
        is independent of `notional_usd`)."""
        if self._pos.side == POS_FLAT or self._pos.entry_price <= 0:
            return 0.0
        if self._pos.side == POS_LONG:
            return (mark - self._pos.entry_price) / self._pos.entry_price
        return (self._pos.entry_price - mark) / self._pos.entry_price

    # ------------------------------------------------------------------
    # observation helpers
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        if self._cursor >= self.n_bars:
            cur = self.n_bars - 1
        else:
            cur = self._cursor
        feat_row = self.features.iloc[cur]
        feat_vec = feat_row.fillna(0.0).to_numpy(dtype=np.float32)
        price = float(self.candles.iloc[cur]["close"])
        pos_block = _position_block(self._pos.side, self._unreal_pnl(price))
        return np.concatenate([feat_vec, pos_block], axis=0).astype(np.float32)

    @property
    def trade_log(self) -> List[dict]:
        return list(self._trade_log)


# ---------------------------------------------------------------------------
# Q-network (numpy-only MLP)
# ---------------------------------------------------------------------------


def _xavier(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0] if len(shape) >= 2 else 1
    bound = math.sqrt(6.0 / max(1, fan_in))
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


@dataclass
class QNetwork:
    """Two-layer MLP with ReLU. Tiny enough to train without autograd
    — we backprop by hand. Hidden layer default is 32 units; bump
    ``hidden`` to 64 / 128 if the model underfits noticeably."""

    n_in: int
    n_out: int
    hidden: int = 32
    seed: int = 0
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.W1 = _xavier((self.n_in, self.hidden), rng)
        self.b1 = np.zeros(self.hidden, dtype=np.float32)
        self.W2 = _xavier((self.hidden, self.n_out), rng)
        self.b2 = np.zeros(self.n_out, dtype=np.float32)

    # --- forward ----------------------------------------------------
    def _forward_cache(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X[np.newaxis, :]
        z1 = X @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        q = h1 @ self.W2 + self.b2
        return q, h1, X

    def forward(self, X: np.ndarray) -> np.ndarray:
        q, _, _ = self._forward_cache(X)
        return q

    # --- backward (MSE on a sub-set of action heads) ----------------
    def fit_batch(
        self, X: np.ndarray, target_q: np.ndarray, actions: np.ndarray, lr: float = 5e-4,
    ) -> float:
        """Single gradient step on ``(X, target_q, actions)``.

        Only the entry of the network's output corresponding to the
        taken ``action`` is updated against ``target_q`` — that's
        standard Q-learning loss. Returns the mean-squared error
        on the action heads.
        """
        q, h1, X_safe = self._forward_cache(X)
        # MSE on the action-specific head
        idx = np.arange(X_safe.shape[0])
        pred = q[idx, actions]
        diff = pred - target_q
        loss = float(np.mean(diff * diff))

        dq = np.zeros_like(q)
        dq[idx, actions] = 2.0 * diff / max(1, X_safe.shape[0])

        dW2 = h1.T @ dq
        db2 = dq.sum(axis=0)
        dh1 = dq @ self.W2.T
        dz1 = dh1 * (h1 > 0)
        dW1 = X_safe.T @ dz1
        db1 = dz1.sum(axis=0)

        # Plain SGD with gradient clipping; Adam would converge faster
        # but adds state we'd have to (de)serialise.
        for grad in (dW1, db1, dW2, db2):
            np.clip(grad, -1.0, 1.0, out=grad)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss

    def copy_from(self, other: "QNetwork") -> None:
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    # --- persistence -----------------------------------------------
    def state_dict(self) -> dict:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "n_in": self.n_in, "n_out": self.n_out, "hidden": self.hidden,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "QNetwork":
        net = cls(
            n_in=int(state["n_in"]),
            n_out=int(state["n_out"]),
            hidden=int(state["hidden"]),
        )
        net.W1 = state["W1"].astype(np.float32)
        net.b1 = state["b1"].astype(np.float32)
        net.W2 = state["W2"].astype(np.float32)
        net.b2 = state["b2"].astype(np.float32)
        return net


# ---------------------------------------------------------------------------
# replay buffer
# ---------------------------------------------------------------------------


@dataclass
class _Transition:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, max_size: int = 50_000, seed: int = 0) -> None:
        self.buf: Deque[_Transition] = deque(maxlen=max_size)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool) -> None:
        self.buf.append(_Transition(s.astype(np.float32), int(a), float(r),
                                    s_next.astype(np.float32), bool(done)))

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, len(self.buf), size=n)
        batch = [self.buf[int(i)] for i in idx]
        S = np.stack([t.s for t in batch], axis=0)
        A = np.array([t.a for t in batch], dtype=np.int64)
        R = np.array([t.r for t in batch], dtype=np.float32)
        S2 = np.stack([t.s_next for t in batch], axis=0)
        D = np.array([t.done for t in batch], dtype=np.float32)
        return S, A, R, S2, D


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------


@dataclass
class DQNConfig:
    hidden: int = 32
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    target_sync_steps: int = 500
    warmup_steps: int = 1_000


@dataclass
class TrainStats:
    episodes: int = 0
    total_steps: int = 0
    final_epsilon: float = 0.0
    avg_episode_reward: float = 0.0
    last_episode_reward: float = 0.0
    last_n_trade_pnls: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "final_epsilon": self.final_epsilon,
            "avg_episode_reward": self.avg_episode_reward,
            "last_episode_reward": self.last_episode_reward,
            "last_n_trade_pnls": self.last_n_trade_pnls,
        }


class DQNAgent:
    """Tiny DQN: behaviour net + target net with periodic sync."""

    def __init__(self, n_obs: int = N_OBS, n_actions: int = N_ACTIONS,
                 config: Optional[DQNConfig] = None, seed: int = 0) -> None:
        self.config = config or DQNConfig()
        self.rng = np.random.default_rng(seed)
        self.q = QNetwork(n_in=n_obs, n_out=n_actions, hidden=self.config.hidden, seed=seed)
        self.q_target = QNetwork(n_in=n_obs, n_out=n_actions, hidden=self.config.hidden, seed=seed + 1)
        self.q_target.copy_from(self.q)
        self.buffer = ReplayBuffer(max_size=self.config.buffer_size, seed=seed + 2)
        self.steps = 0

    # ---- action selection -----------------------------------------
    def _epsilon(self) -> float:
        frac = min(1.0, self.steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(self, s: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self._epsilon():
            return int(self.rng.integers(0, N_ACTIONS))
        q = self.q.forward(s)[0]
        return int(np.argmax(q))

    def policy_proba(self, s: np.ndarray) -> np.ndarray:
        """Softmax over Q-values — useful for the registry's
        confidence-blend step. Temperature is fixed (1.0) on purpose:
        the agent is the one calibrated component."""
        q = self.q.forward(s)[0]
        q = q - q.max()  # softmax stability
        e = np.exp(q)
        return e / e.sum()

    # ---- learning -------------------------------------------------
    def remember(self, s, a, r, s_next, done) -> None:
        self.buffer.add(s, a, r, s_next, done)

    def maybe_learn(self) -> Optional[float]:
        self.steps += 1
        if len(self.buffer) < max(self.config.batch_size, self.config.warmup_steps):
            return None
        S, A, R, S2, D = self.buffer.sample(self.config.batch_size)
        next_q = self.q_target.forward(S2)
        next_max = np.max(next_q, axis=1)
        target = R + (1.0 - D) * self.config.gamma * next_max
        loss = self.q.fit_batch(S, target, A, lr=self.config.lr)
        if self.steps % self.config.target_sync_steps == 0:
            self.q_target.copy_from(self.q)
        return loss

    # ---- persistence ----------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.q.state_dict()
        np.savez(
            path,
            W1=state["W1"], b1=state["b1"], W2=state["W2"], b2=state["b2"],
            n_in=state["n_in"], n_out=state["n_out"], hidden=state["hidden"],
            steps=self.steps,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DQNAgent":
        path = Path(path)
        blob = np.load(path)
        cfg = DQNConfig(hidden=int(blob["hidden"]))
        agent = cls(n_obs=int(blob["n_in"]), n_actions=int(blob["n_out"]),
                    config=cfg, seed=0)
        agent.q = QNetwork.from_state_dict({
            "W1": blob["W1"], "b1": blob["b1"],
            "W2": blob["W2"], "b2": blob["b2"],
            "n_in": int(blob["n_in"]), "n_out": int(blob["n_out"]),
            "hidden": int(blob["hidden"]),
        })
        agent.q_target = QNetwork.from_state_dict({
            "W1": blob["W1"], "b1": blob["b1"],
            "W2": blob["W2"], "b2": blob["b2"],
            "n_in": int(blob["n_in"]), "n_out": int(blob["n_out"]),
            "hidden": int(blob["hidden"]),
        })
        agent.steps = int(blob["steps"])
        return agent


# ---------------------------------------------------------------------------
# trainer + scorer
# ---------------------------------------------------------------------------


def train_dqn_agent(
    candles: pd.DataFrame,
    *,
    episodes: int = 20,
    env_config: Optional[TradingEnvConfig] = None,
    agent_config: Optional[DQNConfig] = None,
    seed: int = 42,
) -> Tuple[DQNAgent, TrainStats]:
    """Train a DQN agent on a candle history. Returns the trained
    agent + summary stats."""
    env = TradingEnv(candles, config=env_config)
    agent = DQNAgent(n_obs=N_OBS, n_actions=N_ACTIONS,
                     config=agent_config, seed=seed)
    episode_rewards: List[float] = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            a = agent.select_action(s)
            s_next, r, done, _ = env.step(a)
            agent.remember(s, a, r, s_next, done)
            agent.maybe_learn()
            s = s_next
            ep_reward += r
        episode_rewards.append(ep_reward)
        if (ep + 1) % max(1, episodes // 10) == 0:
            log.info(
                "rl: episode %d/%d  reward=%.4f  eps=%.3f  trades=%d",
                ep + 1, episodes, ep_reward, agent._epsilon(), len(env.trade_log),
            )
    last_trade_pnls = [float(t["pnl_usd"]) for t in env.trade_log[-10:]]
    stats = TrainStats(
        episodes=episodes,
        total_steps=agent.steps,
        final_epsilon=agent._epsilon(),
        avg_episode_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        last_episode_reward=episode_rewards[-1] if episode_rewards else 0.0,
        last_n_trade_pnls=last_trade_pnls,
    )
    return agent, stats


class RLScorer:
    """Wraps a trained :class:`DQNAgent` so it can plug into the
    money-printer's scoring registry alongside the GBT / ensemble /
    sequence scorers.

    ``predict_proba_win(features)`` returns a 0–1 confidence drawn
    from the agent's policy over (HOLD, BUY, SELL): the score is
    ``max(P(BUY), P(SELL))`` — i.e. "how confident is the policy
    that some directional trade is good right now?". Direction
    selection is left to the strategy (it already has its own
    bias signal from RSI / EMA-dev). This lets the RL contribution
    act as a YES / NO confirmation while the GBT picks the side.
    """

    def __init__(self, agent: Optional[DQNAgent] = None) -> None:
        self.agent = agent

    @staticmethod
    def available() -> bool:
        return True   # numpy-only, always available

    def predict_proba_win(self, features) -> float:
        if self.agent is None:
            return 0.5
        if isinstance(features, pd.Series):
            features = features.to_numpy(dtype=np.float32)
        elif isinstance(features, pd.DataFrame):
            features = features.iloc[-1].to_numpy(dtype=np.float32)
        else:
            features = np.asarray(features, dtype=np.float32)

        # If we got just the 15 GBT features (no position block), append
        # a synthetic "flat / no position" block. This is the right
        # default for "should I open?" queries.
        if features.shape[-1] == len(FEATURE_COLUMNS):
            extra = _position_block(POS_FLAT, 0.0)
            features = np.concatenate([features, extra], axis=-1)

        proba = self.agent.policy_proba(features)
        # P(any directional action) = P(BUY) + P(SELL)
        # We want a "confidence the agent wants to trade" score.
        return float(max(proba[ACTION_BUY], proba[ACTION_SELL]))


def load_rl_scorer(path: str | Path) -> RLScorer:
    agent = DQNAgent.load(path)
    return RLScorer(agent=agent)
