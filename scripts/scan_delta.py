"""Read-only Delta Exchange market scanner.

Lists the configured Delta products (default: BTCUSD, ETHUSD perpetuals),
fetches their L2 order books, and prints best bid/ask + spread + 24h
volume. No credentials needed. No orders placed.

Run:
    python -m scripts.scan_delta
    python -m scripts.scan_delta --symbols BTCUSD,ETHUSD,SOLUSD
    python -m scripts.scan_delta --contract-types perpetual_futures,futures
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aera.markets import DeltaClient
from aera.settings import get_settings


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None, help="comma-separated symbol list (e.g. BTCUSD,ETHUSD)")
    ap.add_argument(
        "--contract-types",
        default=None,
        help="comma-separated contract types (perpetual_futures, futures, call_options, put_options)",
    )
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    settings = get_settings()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    contract_types = (
        [c.strip() for c in args.contract_types.split(",")] if args.contract_types else None
    )

    console = Console()
    console.rule(
        f"[bold cyan]delta scan[/] base={settings.markets.delta.base_url}"
    )

    async with DeltaClient(settings.markets.delta) as delta:
        markets = await delta.list_active_markets(
            symbols=symbols, contract_types=contract_types, limit=args.limit
        )
        if not markets:
            console.print("[red]no markets matched filters[/]")
            return

        books = await delta.fetch_books_batch([m.id for m in markets])
        table = Table(title="delta exchange — top markets")
        table.add_column("symbol", style="cyan")
        table.add_column("category", style="magenta")
        table.add_column("bid")
        table.add_column("ask")
        table.add_column("spread bps")
        table.add_column("bid size")
        table.add_column("ask size")

        for m in markets:
            book = books.get(m.id)
            if book is None:
                table.add_row(m.id, m.category, "—", "—", "—", "—", "—")
                continue
            bb = book.best_bid()
            ba = book.best_ask()
            if bb is None or ba is None:
                table.add_row(m.id, m.category, "—", "—", "—", "—", "—")
                continue
            mid = 0.5 * (bb.price + ba.price)
            sp_bps = (ba.price - bb.price) / mid * 1e4 if mid > 0 else 0
            table.add_row(
                m.id,
                m.category,
                f"{bb.price:,.2f}",
                f"{ba.price:,.2f}",
                f"{sp_bps:.2f}",
                f"{bb.size:.0f}",
                f"{ba.size:.0f}",
            )

        console.print(table)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
