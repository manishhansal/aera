"""Train the DQN trading policy.

Numpy-only — no torch dependency. The agent walks a per-symbol
candle history repeatedly (``--episodes``), each episode starting
flat and ending with a forced flatten at the last bar. Reward at
each step is unrealised-PnL delta + a small holding penalty;
realised PnL on close is amplified by ``realise_bonus_mult``.

Persists ``data/money_printer/rl_policy.npz``; the registry
auto-loads it as one of MoneyPrinter's scorers.

Train per-symbol or on the union — pass ``--symbol BTCUSD`` to
focus on a single market.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

from aera.data import CandleStore
from aera.logging import get_logger
from aera.ml.rl import DQNConfig, TradingEnvConfig, train_dqn_agent

log = get_logger("scripts.train_rl_agent")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the DQN trading policy.")
    ap.add_argument("--symbol", default="BTCUSD",
                    help="Symbol whose candles to train on (must exist in --history-dir).")
    ap.add_argument("--resolution", default="5m")
    ap.add_argument("--history-dir", default="data/history")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    ap.add_argument("--notional-usd", type=float, default=100.0)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--max-hold-bars", type=int, default=60)
    ap.add_argument("--out", default="data/money_printer/rl_policy.npz")
    ap.add_argument("--report", default="data/money_printer/rl_report.json")
    args = ap.parse_args()

    console = Console()
    store = CandleStore(root=Path(args.history_dir))
    candles = store.load(args.symbol, args.resolution)
    if candles.empty:
        console.print(
            f"[red]No cached candles for {args.symbol} {args.resolution}. "
            f"Run `python -m scripts.fetch_history --symbols {args.symbol} "
            f"--resolutions {args.resolution}` first.[/]"
        )
        return

    console.rule(f"[bold cyan]DQN trainer[/]  {args.symbol} {args.resolution}")
    console.print(f"[dim]{len(candles)} bars  ({args.episodes} episodes)[/]")

    env_cfg = TradingEnvConfig(
        fee_bps=args.fee_bps,
        notional_usd=args.notional_usd,
        max_hold_bars=args.max_hold_bars,
    )
    agent_cfg = DQNConfig(
        hidden=args.hidden,
        gamma=args.gamma,
        epsilon_decay_steps=args.epsilon_decay_steps,
        lr=args.lr,
    )
    agent, stats = train_dqn_agent(
        candles, episodes=args.episodes,
        env_config=env_cfg, agent_config=agent_cfg, seed=42,
    )

    out = Path(args.out)
    agent.save(out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "symbol": args.symbol, "resolution": args.resolution,
        "episodes": stats.episodes, "total_steps": stats.total_steps,
        "final_epsilon": stats.final_epsilon,
        "avg_episode_reward": stats.avg_episode_reward,
        "last_episode_reward": stats.last_episode_reward,
        "last_n_trade_pnls": stats.last_n_trade_pnls,
    }, indent=2))

    console.print(f"[green]Saved DQN agent → {out}[/]")
    console.print(
        f"\n[bold]Stats[/]  episodes={stats.episodes}  steps={stats.total_steps}  "
        f"final_eps={stats.final_epsilon:.3f}  avg_reward={stats.avg_episode_reward:.4f}  "
        f"last_reward={stats.last_episode_reward:.4f}"
    )
    if stats.last_n_trade_pnls:
        console.print(
            "  last trades pnl: "
            + ", ".join(f"${p:+.2f}" for p in stats.last_n_trade_pnls)
        )


if __name__ == "__main__":
    main()
