"""Grid backtest sweep + analysis output.

Usage::

    python -m scripts.sweep_backtest \
        --symbols BTCUSD,ETHUSD,SOLUSD \
        --resolutions 1m,5m \
        --leverages 10,25,50 \
        --days 30

Backtests every (strategy × symbol × resolution × leverage) combo,
writes:

* ``data/backtest/sweep_summary.csv`` — one row per config with
  headline metrics, sorted by total PnL.
* ``data/backtest/sweep_trades.csv``  — flat trade list (every
  round-trip across all configs).
* ``data/money_printer/hour_maps.json`` — per (strategy, symbol)
  profitable-hour map consumed by the MoneyPrinter at live-trade
  time.

The MoneyPrinter strategy is excluded from the sweep by default
(it's the consumer of the sweep output, not a participant). Pass
``--include-money-printer`` to evaluate it too — useful AFTER
training to measure end-to-end uplift.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
from rich.console import Console
from rich.table import Table

from aera.backtest import (
    SweepConfig,
    build_hour_map,
    run_sweep,
    summarise_results,
    write_summary_csv,
)
from aera.backtest.analysis import build_all_hour_maps, write_hour_maps
from aera.data import CandleStore, DeltaHistoryClient, fetch_history
from aera.logging import get_logger

# We reuse the live runner's strategy wiring so the sweep tests every
# strategy with its tuned ``config.yaml`` parameters — not the bare
# ``Strategy.__init__`` defaults. Without this, backtest results don't
# predict live performance (the live bot e.g. runs ``zscore_entry=2.5``
# while the sweep was running with ``2.0``, producing wildly different
# fire rates and PnL distributions).
from scripts.run_delta import STRATEGY_NAMES, make_strategy


log = get_logger("scripts.sweep_backtest")


def _factories_for(names) -> Dict[str, Callable]:
    """Build the ``{name: factory(portfolio) -> Strategy}`` map the
    sweep needs. Each factory delegates to ``make_strategy`` so the
    sweep stays in lock-step with whatever ``run_delta`` runs live.
    """
    return {
        n: (lambda pf, _n=n: make_strategy(_n, pf))
        for n in names
        if n in STRATEGY_NAMES
    }


async def _prefetch(symbols, resolutions, days) -> None:
    async with DeltaHistoryClient() as client:
        for sym in symbols:
            for res in resolutions:
                await fetch_history(sym, res, days=days, client=client)


def _candles_loader(store: CandleStore):
    def _load(symbol: str, resolution: str) -> pd.DataFrame:
        # The prefetch above guarantees these files exist.
        return store.load(symbol, resolution)
    return _load


async def _main() -> None:
    ap = argparse.ArgumentParser(description="Run a grid sweep backtest.")
    ap.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD",
                    help="Comma-separated Delta symbols.")
    ap.add_argument("--resolutions", default="1m,5m",
                    help="Comma-separated resolutions to test.")
    ap.add_argument("--leverages", default="10,25,50",
                    help="Comma-separated leverages.")
    ap.add_argument("--strategies", default="all",
                    help="Comma-separated strategies, or 'all'.")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include-money-printer", action="store_true",
                    help="Also evaluate the MoneyPrinter strategy.")
    ap.add_argument("--out-dir", default="data/backtest")
    ap.add_argument("--money-printer-dir", default="data/money_printer")
    args = ap.parse_args()

    console = Console()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    resolutions = [r.strip() for r in args.resolutions.split(",") if r.strip()]
    leverages = [float(l.strip()) for l in args.leverages.split(",") if l.strip()]

    if args.strategies == "all":
        # Default: every hand-tuned strategy EXCEPT MoneyPrinter
        # (MoneyPrinter is the consumer of the sweep, not a
        # participant — it would just train on its own un-trained
        # fallback unless ``--include-money-printer`` is set).
        strat_names = [n for n in STRATEGY_NAMES if n != "money_printer"]
    else:
        strat_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if args.include_money_printer and "money_printer" not in strat_names:
        strat_names.append("money_printer")

    factories = _factories_for(strat_names)
    if not factories:
        console.print("[red]No valid strategies selected; aborting.[/]")
        return

    console.rule(
        f"[bold cyan]sweep[/]  {len(strat_names)} strategies × {len(symbols)} symbols "
        f"× {len(resolutions)} resolutions × {len(leverages)} leverages"
    )

    # Prefetch all candles into the cache so the sweep itself is
    # CPU-bound, not network-bound.
    console.print(f"[dim]Pre-fetching {len(symbols) * len(resolutions)} (symbol × resolution) candle sets...[/]")
    await _prefetch(symbols, resolutions, args.days)

    store = CandleStore()
    cfg = SweepConfig(
        strategies=factories,
        symbols=symbols,
        resolutions=resolutions,
        leverages=leverages,
        starting_bankroll=args.bankroll,
        max_workers=args.workers,
    )

    out = run_sweep(cfg=cfg, candles_for=_candles_loader(store))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_df = out.to_df()
    trades_df = out.trades_df()
    summary_path = write_summary_csv(rows_df, out_dir / "sweep_summary.csv")
    if not trades_df.empty:
        trades_df.to_csv(out_dir / "sweep_trades.csv", index=False)

    # Hour maps for the money printer.
    maps = build_all_hour_maps(out.trades)
    if maps:
        mp_dir = Path(args.money_printer_dir)
        mp_dir.mkdir(parents=True, exist_ok=True)
        write_hour_maps(maps, mp_dir / "hour_maps.json")
        console.print(f"[green]Wrote {len(maps)} hour maps → {mp_dir / 'hour_maps.json'}[/]")

    # Top-10 leaderboard
    if not rows_df.empty:
        top = out.top(10)
        table = Table(title="Top 10 configurations (by total PnL)")
        for col in top.columns:
            table.add_column(col)
        for _, row in top.iterrows():
            table.add_row(*[
                f"{v:.4f}" if isinstance(v, float) and v is not None else str(v)
                for v in row.values
            ])
        console.print(table)
    console.print(f"[green]Summary CSV → {summary_path}[/]")


if __name__ == "__main__":
    asyncio.run(_main())
