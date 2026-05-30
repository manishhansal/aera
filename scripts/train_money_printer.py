"""Train the MoneyPrinter's profitability classifier.

Consumes the trade list written by ``sweep_backtest.py``, extracts
features from the corresponding candles, and fits a
:class:`~aera.ml.model.ProfitabilityClassifier` on the
(features → win/loss) target. The fitted model is dumped to
``data/money_printer/model.joblib`` and consumed by the
:class:`~aera.strategies.money_printer.MoneyPrinter` strategy at
live-trade time.

Usage::

    # 1. fetch history
    python -m scripts.fetch_history --symbols BTCUSD,ETHUSD,SOLUSD --days 30

    # 2. run sweep (writes data/backtest/sweep_trades.csv + hour_maps.json)
    python -m scripts.sweep_backtest --symbols BTCUSD,ETHUSD,SOLUSD --days 30

    # 3. train the classifier
    python -m scripts.train_money_printer

By default we train ONE classifier across all symbols / strategies
(more data, more stable). Pass ``--per-symbol`` to fit a separate
model per symbol (when there's enough data).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from aera.backtest.replay import TradeRecord
from aera.data import CandleStore, fetch_history
from aera.logging import get_logger
from aera.ml import label_trades, train_model

log = get_logger("scripts.train_money_printer")


def _load_trades_from_csv(path: Path) -> list[TradeRecord]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    out: list[TradeRecord] = []
    for _, row in df.iterrows():
        out.append(TradeRecord(
            strategy=str(row.strategy),
            symbol=str(row.symbol),
            side=str(row.side),
            entry_ts=int(row.entry_ts),
            exit_ts=int(row.exit_ts),
            entry_price=float(row.entry_price),
            exit_price=float(row.exit_price),
            size_usd=float(row.size_usd),
            leverage=float(row.leverage),
            pnl_usd=float(row.pnl_usd),
            fees_usd=float(row.fees_usd),
            hold_seconds=int(row.hold_seconds),
            reason_open=str(row.get("reason_open", "")),
            reason_close=str(row.get("reason_close", "")),
        ))
    return out


async def _main() -> None:
    ap = argparse.ArgumentParser(description="Train the money-printer profitability classifier.")
    ap.add_argument("--trades-csv", default="data/backtest/sweep_trades.csv")
    ap.add_argument("--out-dir", default="data/money_printer")
    ap.add_argument("--resolution", default="1m",
                    help="Resolution of cached candles to use for feature extraction.")
    ap.add_argument("--days", type=int, default=30,
                    help="How many days of candle history to fetch / use for features.")
    ap.add_argument("--test-frac", type=float, default=0.20,
                    help="Hold-out fraction for walk-forward evaluation.")
    ap.add_argument("--per-symbol", action="store_true",
                    help="Train one model per symbol instead of a single pooled model.")
    args = ap.parse_args()

    console = Console()
    console.rule("[bold cyan]train money_printer[/]")

    trades_path = Path(args.trades_csv)
    trades = _load_trades_from_csv(trades_path)
    if not trades:
        console.print(f"[red]No trades found at {trades_path} — run sweep_backtest first.[/]")
        return
    console.print(f"[dim]Loaded {len(trades):,} trades from {trades_path}[/]")

    symbols = sorted({t.symbol for t in trades})
    console.print(f"[dim]Distinct symbols: {symbols}[/]")

    candles_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = await fetch_history(sym, args.resolution, days=args.days)
        if df.empty:
            console.print(f"[yellow]Skipping {sym}: no candles cached.[/]")
            continue
        candles_by_symbol[sym] = df

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.per_symbol:
        for sym in candles_by_symbol:
            sym_trades = [t for t in trades if t.symbol == sym]
            labelled = label_trades(sym_trades, {sym: candles_by_symbol[sym]})
            if labelled.empty:
                console.print(f"[yellow]{sym}: no labellable trades, skipping.[/]")
                continue
            console.print(f"[dim]{sym}: {len(labelled)} labelled samples → fitting...[/]")
            model, report = train_model(labelled, test_frac=args.test_frac)
            model.save(out_dir / f"model_{sym}.joblib")
            (out_dir / f"report_{sym}.json").write_text(json.dumps(report.to_dict(), indent=2))
            _print_report(console, f"Model for {sym}", report)
    else:
        labelled = label_trades(trades, candles_by_symbol)
        if labelled.empty:
            console.print("[red]No labellable trades after joining with candles. Aborting.[/]")
            return
        console.print(f"[dim]Pooled training set: {len(labelled):,} samples[/]")
        model, report = train_model(labelled, test_frac=args.test_frac)
        model.save(out_dir / "model.joblib")
        (out_dir / "train_report.json").write_text(json.dumps(report.to_dict(), indent=2))
        _print_report(console, "Pooled model", report)

    console.print(f"[green]Done. Artefacts in {out_dir}/[/]")
    console.print(
        "[dim]Enable the strategy by setting "
        "[bold]strategies.money_printer.enabled: true[/] in config.yaml "
        "and restarting the bot.[/]"
    )


def _print_report(console: Console, title: str, report) -> None:
    table = Table(title=title)
    table.add_column("metric")
    table.add_column("value", justify="right")
    d = report.to_dict()
    for k in ("n_train", "n_test", "n_features", "accuracy", "precision",
              "recall", "f1", "roc_auc"):
        v = d.get(k)
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    table.add_row("class_balance", json.dumps(d.get("class_balance", {})))
    console.print(table)
    fi = d.get("feature_importance", {})
    if fi:
        top = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:10]
        fi_table = Table(title="Top 10 feature importances")
        fi_table.add_column("feature")
        fi_table.add_column("importance", justify="right")
        for name, score in top:
            fi_table.add_row(name, f"{score:.4f}")
        console.print(fi_table)


if __name__ == "__main__":
    asyncio.run(_main())
