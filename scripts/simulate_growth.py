"""Monte-Carlo simulation of $1 -> $1,000,000 under various edges.

Run:
    python -m scripts.simulate_growth --start 1 --target 1000000 \
        --win-rate 0.535 --trades 37000 --kelly 0.25 --runs 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aera.core.compounding import simulate_growth, trades_needed_for_target


def main() -> None:
    ap = argparse.ArgumentParser(description="Monte-Carlo $1->$1M compounding sim")
    ap.add_argument("--start", type=float, default=1.0)
    ap.add_argument("--target", type=float, default=1_000_000.0)
    ap.add_argument("--trades", type=int, default=37_000)
    ap.add_argument("--win-rate", type=float, default=0.535)
    ap.add_argument("--payoff", type=float, default=1.0)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--edge", type=float, default=None,
                    help="optional: solve for win-rate that gives this average edge per trade")
    ap.add_argument("--plot", action="store_true", help="save matplotlib growth plot to growth.png")
    args = ap.parse_args()

    console = Console()
    console.rule("[bold cyan]aera compounding simulator")

    # Deterministic required-edge table
    table = Table(title="Required average geometric edge per trade")
    table.add_column("Trades N")
    table.add_column("Required g - 1  (per trade)")
    table.add_column("As bps")
    for n in [100, 1000, 5000, 10_000, 37_000, 100_000]:
        g = (args.target / args.start) ** (1.0 / n) - 1.0
        table.add_row(f"{n:,}", f"{g*100:.6f} %", f"{g*10_000:.4f} bps")
    console.print(table)

    if args.edge is not None:
        n = trades_needed_for_target(
            starting_bankroll=args.start, target=args.target, edge_per_trade=args.edge
        )
        console.print(f"[bold]At edge {args.edge*100:.4f}% per trade, you need ~{n:,.0f} trades.[/]\n")

    console.print(
        f"[bold]Monte-Carlo:[/] start=${args.start} target=${args.target:,.0f} "
        f"trades={args.trades:,} win_rate={args.win_rate} kelly={args.kelly}"
    )

    result = simulate_growth(
        starting_bankroll=args.start,
        target=args.target,
        n_trades=args.trades,
        win_rate=args.win_rate,
        payoff_ratio=args.payoff,
        kelly_fraction=args.kelly,
        runs=args.runs,
    )

    out = Table(title="Monte-Carlo results")
    out.add_column("metric")
    out.add_column("value")
    out.add_row("P(reach target)", f"{result.p_hit_target*100:.2f}%")
    out.add_row("median terminal $", f"${np.median(result.terminal_bankrolls):,.2f}")
    out.add_row("mean terminal $",   f"${np.mean(result.terminal_bankrolls):,.2f}")
    out.add_row("p95 terminal $",    f"${np.quantile(result.terminal_bankrolls, 0.95):,.2f}")
    out.add_row("p5 terminal $",     f"${np.quantile(result.terminal_bankrolls, 0.05):,.2f}")
    out.add_row("E[log growth]/trade", f"{result.expected_log_growth_per_trade:.6f}")
    out.add_row("required g/trade",   f"{result.required_edge_per_trade_for_target*100:.6f}%")
    console.print(out)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(args.trades + 1)
            ax.plot(x, result.median_path, label="median", color="#22d3ee")
            ax.fill_between(x, result.p5_path, result.p95_path, alpha=0.2,
                            color="#22d3ee", label="5/95 percentile")
            ax.axhline(args.target, color="#ef4444", linestyle="--", label=f"target ${args.target:,.0f}")
            ax.set_yscale("log")
            ax.set_xlabel("trade #")
            ax.set_ylabel("bankroll ($, log scale)")
            ax.set_title(
                f"$1 -> $1M  |  win={args.win_rate}  Kelly={args.kelly}  "
                f"P(hit)={result.p_hit_target*100:.1f}%"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = ROOT / "growth.png"
            fig.savefig(out_path, dpi=140)
            console.print(f"saved plot to [bold]{out_path}[/]")
        except ImportError:
            console.print("[yellow]matplotlib not installed, skipping plot[/]")


if __name__ == "__main__":
    main()
