"""Train the transformer-encoder sequence model.

Requires ``torch`` (``pip install torch``). The model consumes the
last ``--seq-len`` bars of OHLCV-derived per-bar features
(``aera.ml.sequence.SEQUENCE_FEATURES``) and predicts ``P(win)``
for the current setup.

Reads the same trade dataset as ``scripts.train_ensemble``
(``data/backtest/sweep_trades.csv``) and the OHLCV cache
(``data/history/``); writes ``data/money_printer/sequence_model.pt``
+ metadata so the registry can auto-load it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console

from aera.data import CandleStore
from aera.logging import get_logger
from aera.ml.sequence import (
    SequenceConfig,
    SequenceScorer,
    build_sequences_from_trades,
    train_sequence_model,
)

log = get_logger("scripts.train_sequence_model")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the sequence model.")
    ap.add_argument("--trades-csv", default="data/backtest/sweep_trades.csv")
    ap.add_argument("--history-dir", default="data/history")
    ap.add_argument("--resolution", default="5m")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--out", default="data/money_printer/sequence_model.pt")
    ap.add_argument("--report", default="data/money_printer/sequence_report.json")
    args = ap.parse_args()

    console = Console()
    if not SequenceScorer.available():
        console.print(f"[red]{SequenceScorer.reason_unavailable()}[/]")
        return

    trades_csv = Path(args.trades_csv)
    if not trades_csv.exists():
        console.print(f"[red]Trades CSV not found: {trades_csv}[/]")
        console.print("[yellow]Run `python -m scripts.sweep_backtest` first.[/]")
        return

    df = pd.read_csv(trades_csv)
    if df.empty:
        console.print("[red]Trades CSV is empty — nothing to train on.[/]")
        return
    symbols = sorted(df["symbol"].astype(str).str.upper().unique().tolist())
    console.print(f"[dim]{len(df)} trades across {len(symbols)} symbols[/]")

    store = CandleStore(root=Path(args.history_dir))
    candles_by_symbol = {sym: store.load(sym, args.resolution) for sym in symbols}
    candles_by_symbol = {k: v for k, v in candles_by_symbol.items() if not v.empty}
    if not candles_by_symbol:
        console.print("[red]No cached candles found — run `python -m scripts.fetch_history` first.[/]")
        return

    X, y, syms = build_sequences_from_trades(
        df, candles_by_symbol, seq_len=args.seq_len,
    )
    console.print(f"[dim]Built {len(X)} sequences  pos={int(y.sum())}  neg={int((1 - y).sum())}[/]")
    if len(X) < 32:
        console.print("[red]Too few sequences (need ≥ 32) — wait for more trades.[/]")
        return

    cfg = SequenceConfig(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    console.print(f"[dim]Training transformer ({args.epochs} epochs)…[/]")
    scorer, report = train_sequence_model(
        X, y, config=cfg, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
    )
    out = Path(args.out)
    scorer.save(out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report.to_dict(), indent=2))

    console.print(f"[green]Saved sequence model → {out}[/]")
    console.print(f"[green]Wrote report → {args.report}[/]")
    console.print(
        f"\n[bold]Eval[/]  train_loss={report.train_loss:.4f}  "
        f"test_loss={report.test_loss:.4f}  test_acc={report.test_accuracy:.3f}  "
        f"test_roc_auc={report.test_roc_auc:.3f}  "
        f"n_train={report.n_train}  n_test={report.n_test}"
    )


if __name__ == "__main__":
    main()
