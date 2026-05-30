"""Train the per-symbol ensemble of profitability classifiers.

Run AFTER ``scripts/sweep_backtest.py`` has produced
``data/backtest/sweep_trades.csv``. The trainer:

* fits one ``HistGradientBoostingClassifier`` per symbol that has at
  least ``--min-per-symbol`` trades (default 200),
* always fits a global fallback on the union,
* persists everything under ``data/money_printer/ensemble/``.

The :class:`MoneyPrinter` picks the ensemble up automatically on next
start (see ``aera/ml/registry.py``).

Example
-------

::

    python -m scripts.train_ensemble \\
        --trades-csv data/backtest/sweep_trades.csv \\
        --history-dir data/history \\
        --out-dir data/money_printer/ensemble \\
        --min-per-symbol 250
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
from rich.console import Console

from aera.data import CandleStore
from aera.logging import get_logger
from aera.ml import label_trades, train_ensemble
from aera.backtest.replay import TradeRecord

log = get_logger("scripts.train_ensemble")


def _trades_from_csv(path: Path) -> list[TradeRecord]:
    df = pd.read_csv(path)
    needed = {"strategy", "symbol", "side", "entry_ts", "exit_ts",
              "entry_price", "exit_price", "size_usd", "leverage",
              "pnl_usd", "fees_usd", "hold_seconds"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"sweep_trades.csv missing columns: {missing}")
    out: list[TradeRecord] = []
    for _, r in df.iterrows():
        out.append(TradeRecord(
            strategy=str(r["strategy"]),
            symbol=str(r["symbol"]),
            side=str(r["side"]),
            entry_ts=int(r["entry_ts"]),
            exit_ts=int(r["exit_ts"]),
            entry_price=float(r["entry_price"]),
            exit_price=float(r["exit_price"]),
            size_usd=float(r["size_usd"]),
            leverage=float(r["leverage"]),
            pnl_usd=float(r["pnl_usd"]),
            fees_usd=float(r["fees_usd"]),
            hold_seconds=int(r["hold_seconds"]),
            reason_open=str(r.get("reason_open", "")),
            reason_close=str(r.get("reason_close", "")),
        ))
    return out


def _candles_for_symbols(history_dir: Path, symbols: list[str], resolution: str) -> Dict[str, pd.DataFrame]:
    store = CandleStore(root=history_dir)
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = store.load(sym, resolution)
        if not df.empty:
            out[sym] = df
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the per-symbol ensemble.")
    ap.add_argument("--trades-csv", default="data/backtest/sweep_trades.csv")
    ap.add_argument("--history-dir", default="data/history")
    ap.add_argument("--resolution", default="5m",
                    help="OHLCV resolution to look up features for; should "
                         "match what the sweep used to generate the trades.")
    ap.add_argument("--out-dir", default="data/money_printer/ensemble")
    ap.add_argument("--min-per-symbol", type=int, default=200)
    ap.add_argument("--report", default="data/money_printer/ensemble_report.json")
    args = ap.parse_args()

    console = Console()
    trades_csv = Path(args.trades_csv)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)

    if not trades_csv.exists():
        console.print(f"[red]Trades CSV not found: {trades_csv}[/]")
        console.print("[yellow]Run `python -m scripts.sweep_backtest` first.[/]")
        return

    console.rule("[bold cyan]ensemble trainer[/]")
    console.print(f"[dim]Loading trades from {trades_csv}[/]")
    trades = _trades_from_csv(trades_csv)
    symbols = sorted({t.symbol for t in trades})
    console.print(f"[dim]{len(trades)} trades across {len(symbols)} symbols[/]")

    candles_by_symbol = _candles_for_symbols(Path(args.history_dir), symbols, args.resolution)
    if not candles_by_symbol:
        console.print("[red]No cached candles found — run `python -m scripts.fetch_history` first.[/]")
        return

    labelled = label_trades(trades, candles_by_symbol)
    if labelled.empty:
        console.print("[red]No labelled rows produced (every trade was filtered out).[/]")
        return
    console.print(f"[dim]Labelled rows: {len(labelled)}[/]")

    ensemble, report = train_ensemble(labelled, min_per_symbol=args.min_per_symbol)
    out_path = ensemble.save(out_dir)
    console.print(f"[green]Saved ensemble → {out_path}[/]")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    console.print(f"[green]Wrote training report → {report_path}[/]")

    console.print(
        f"\n[bold]Fallback acc[/]: {report.fallback.accuracy:.3f}  "
        f"ROC-AUC: {report.fallback.roc_auc:.3f}  "
        f"n_train: {report.fallback.n_train}"
    )
    for sym, rep in report.per_symbol.items():
        console.print(
            f"  [cyan]{sym}[/]  acc={rep.accuracy:.3f}  roc={rep.roc_auc:.3f}  n={rep.n_train}"
        )
    for sym, n in report.skipped_symbols.items():
        console.print(f"  [yellow]{sym} skipped[/] (only {n} rows < {args.min_per_symbol})")


if __name__ == "__main__":
    main()
