"""Backtest a single strategy on a single symbol.

Usage::

    python -m scripts.backtest --strategy delta_perp_scalper \
        --symbol BTCUSD --resolution 1m --days 30 --leverage 25

Outputs the headline metrics + saves the trade list to
``data/backtest/<strategy>_<symbol>_<resolution>_lev<lev>.csv``.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from aera.backtest import BarReplay
from aera.core import Portfolio
from aera.data import CandleStore, fetch_history
from aera.logging import get_logger
from aera.strategies import (
    BidAskSpreadFade,
    DeltaPerpetualScalper,
    FlowScalp,
    MicroVWAPSniper,
    MoneyPrinter,
    OrderBookSniper,
    StopHuntReversal,
    TickReversalScalp,
)


log = get_logger("scripts.backtest")


# Factory map: strategy name → factory(portfolio) -> Strategy.
# These mirror the production ``build_strategies`` builders but use
# the strategy's DEFAULT params so the backtest evaluates the same
# config the live runner would use unless ``--params-from-config``
# is opted into.
STRATEGY_FACTORIES: dict[str, Callable] = {
    "delta_perp_scalper": lambda pf: DeltaPerpetualScalper(portfolio=pf),
    "order_book_sniper":  lambda pf: OrderBookSniper(portfolio=pf),
    "tick_reversal_scalp": lambda pf: TickReversalScalp(portfolio=pf),
    "bid_ask_spread_fade": lambda pf: BidAskSpreadFade(portfolio=pf),
    "flow_scalp":         lambda pf: FlowScalp(portfolio=pf),
    "micro_vwap_sniper":  lambda pf: MicroVWAPSniper(portfolio=pf),
    "stop_hunt_reversal": lambda pf: StopHuntReversal(portfolio=pf),
    "money_printer":      lambda pf: MoneyPrinter(portfolio=pf),
}


async def _main() -> None:
    ap = argparse.ArgumentParser(description="Single-config backtest.")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGY_FACTORIES))
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--resolution", default="1m")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--leverage", type=float, default=25.0)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--taker-fee-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--spread-bps", type=float, default=5.0)
    ap.add_argument("--out-dir", default="data/backtest")
    args = ap.parse_args()

    console = Console()
    console.rule(
        f"[bold cyan]backtest[/]  {args.strategy} / {args.symbol} / "
        f"{args.resolution} / lev={args.leverage}x / {args.days}d"
    )

    candles = await fetch_history(args.symbol, args.resolution, days=args.days)
    if candles.empty:
        console.print("[red]no candles in cache for that window — run fetch_history first[/]")
        return

    replay = BarReplay(
        STRATEGY_FACTORIES[args.strategy],
        starting_bankroll=args.bankroll,
        symbol=args.symbol,
        resolution=args.resolution,
        leverage=args.leverage,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        taker_fee_bps=args.taker_fee_bps,
    )
    result = replay.run(candles)

    table = Table(title="Backtest summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in result.to_dict().items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (
        f"{args.strategy}_{args.symbol}_{args.resolution}_lev{int(args.leverage)}.csv"
    )
    df = result.trades_df()
    if not df.empty:
        df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved {len(df)} trades → {csv_path}[/]")
    else:
        console.print("[yellow]No trades fired during this backtest.[/]")


if __name__ == "__main__":
    asyncio.run(_main())
