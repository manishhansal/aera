"""Fetch historical OHLCV candles into the local cache.

Usage::

    python -m scripts.fetch_history --symbols BTCUSD,ETHUSD,SOLUSD \
        --resolutions 1m,5m --days 30

The cache lives at ``data/history/<SYMBOL>/<resolution>.parquet``.
Re-runs are incremental — only the gap between the cache's last
timestamp and ``now`` is pulled, unless ``--refresh`` is set.
"""
from __future__ import annotations

import argparse
import asyncio
import time

from rich.console import Console
from rich.table import Table

from aera.data import CandleStore, DeltaHistoryClient, fetch_history
from aera.logging import get_logger

log = get_logger("scripts.fetch_history")


async def _main() -> None:
    ap = argparse.ArgumentParser(description="Fetch historical OHLCV into the local cache.")
    ap.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD",
                    help="Comma-separated Delta symbols.")
    ap.add_argument("--resolutions", default="1m",
                    help="Comma-separated resolutions (1m, 5m, 15m, 1h, ...).")
    ap.add_argument("--days", type=int, default=30,
                    help="How many days of history to fetch (rolling from now).")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-fetch the entire window even if some bars are cached.")
    ap.add_argument("--root", default="data/history",
                    help="Cache root directory.")
    args = ap.parse_args()

    console = Console()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    resolutions = [r.strip() for r in args.resolutions.split(",") if r.strip()]
    store = CandleStore(root=args.root)
    console.rule(
        f"[bold cyan]history fetch[/]  symbols={len(symbols)} "
        f"resolutions={len(resolutions)} days={args.days}"
    )

    async with DeltaHistoryClient() as client:
        rows = []
        for sym in symbols:
            for res in resolutions:
                t0 = time.perf_counter()
                df = await fetch_history(
                    sym, res, days=args.days, store=store, client=client,
                    refresh=args.refresh,
                )
                dt = time.perf_counter() - t0
                first_ts, last_ts, total_rows = store.coverage(sym, res)
                rows.append((sym, res, len(df), total_rows, first_ts, last_ts, dt))

    table = Table(title="Cache state after fetch")
    table.add_column("symbol")
    table.add_column("res")
    table.add_column("new bars", justify="right")
    table.add_column("cached rows", justify="right")
    table.add_column("first ts", justify="right")
    table.add_column("last ts", justify="right")
    table.add_column("took", justify="right")
    for sym, res, new, total, first_ts, last_ts, dt in rows:
        table.add_row(
            sym, res, f"{new:,}", f"{total:,}",
            "—" if first_ts is None else str(first_ts),
            "—" if last_ts is None else str(last_ts),
            f"{dt:.2f}s",
        )
    console.print(table)


if __name__ == "__main__":
    asyncio.run(_main())
