"""Post-sweep analysis: hour-of-day heatmaps, profitability tables.

Consumes a :class:`SweepResult`'s trade list and produces:

* :class:`HourMap` — per (strategy, symbol) profit by hour-of-day
  bucket. Used by ``MoneyPrinter`` at live-trade time to gate fires.
* :func:`summarise_results` — flat per-config summary suitable for
  saving as CSV and eyeballing the leader-board.
* :func:`write_summary_csv` — writes the above to disk.

The hour map is intentionally simple — UTC hour of the *entry* (the
data point we have full information about at decision time). The
ML model in ``aera.ml`` consumes the trade DataFrame directly and
learns richer features on top.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from aera.logging import get_logger

from .replay import TradeRecord

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# HourMap: per-(strategy, symbol) profitability by UTC hour
# ---------------------------------------------------------------------------


@dataclass
class HourMap:
    """24-bucket profitability map for one (strategy, symbol) pair.

    ``pnl[hour]`` = sum of realised PnL on trades ENTERED in that
    UTC hour. ``win_rate[hour]`` = fraction won. ``count[hour]`` =
    number of trades sampled.

    A bucket is considered "profitable" iff its
    expectancy (``pnl[hour] / count[hour]``) is >= ``min_expectancy``
    AND its win-rate is >= ``min_win_rate`` AND it has at least
    ``min_count`` sample trades. Default thresholds are tuned so the
    map only marks an hour "ON" when the historical edge is robust
    (not a single lucky trade).
    """

    strategy: str
    symbol: str
    pnl: List[float] = field(default_factory=lambda: [0.0] * 24)
    count: List[int] = field(default_factory=lambda: [0] * 24)
    wins: List[int] = field(default_factory=lambda: [0] * 24)

    def add(self, trade: TradeRecord) -> None:
        h = trade.hour_of_day
        self.pnl[h] += trade.pnl_usd
        self.count[h] += 1
        if trade.is_win:
            self.wins[h] += 1

    def win_rate(self, hour: int) -> float:
        n = self.count[hour]
        return (self.wins[hour] / n) if n > 0 else 0.0

    def expectancy(self, hour: int) -> float:
        n = self.count[hour]
        return (self.pnl[hour] / n) if n > 0 else 0.0

    def hours_allowed(
        self,
        *,
        min_count: int = 5,
        min_win_rate: float = 0.50,
        min_expectancy: float = 0.0,
    ) -> List[int]:
        out: List[int] = []
        for h in range(24):
            if self.count[h] < min_count:
                continue
            if self.win_rate(h) < min_win_rate:
                continue
            if self.expectancy(h) < min_expectancy:
                continue
            out.append(h)
        return out

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "pnl": self.pnl,
            "count": self.count,
            "wins": self.wins,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HourMap":
        m = cls(strategy=data["strategy"], symbol=data["symbol"])
        m.pnl = list(data["pnl"])
        m.count = list(data["count"])
        m.wins = list(data["wins"])
        return m


def build_hour_map(
    trades: Iterable[TradeRecord], *, strategy: str, symbol: str,
) -> HourMap:
    m = HourMap(strategy=strategy, symbol=symbol)
    for t in trades:
        if t.strategy == strategy and t.symbol == symbol:
            m.add(t)
    return m


def build_all_hour_maps(trades: Iterable[TradeRecord]) -> Dict[tuple[str, str], HourMap]:
    """Bucket trades by (strategy, symbol) and produce a HourMap for each."""
    trade_list = list(trades)
    pairs = {(t.strategy, t.symbol) for t in trade_list}
    out: Dict[tuple[str, str], HourMap] = {}
    for strat, sym in pairs:
        out[(strat, sym)] = build_hour_map(trade_list, strategy=strat, symbol=sym)
    return out


def write_hour_maps(
    maps: Dict[tuple[str, str], HourMap], path: str | Path,
) -> None:
    """Persist hour maps as a JSON file consumable by the money printer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"{k[0]}|{k[1]}": v.to_dict() for k, v in maps.items()
    }
    path.write_text(json.dumps(payload, indent=2))


def load_hour_maps(path: str | Path) -> Dict[tuple[str, str], HourMap]:
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out: Dict[tuple[str, str], HourMap] = {}
    for key, body in payload.items():
        strat, _, sym = key.partition("|")
        out[(strat, sym)] = HourMap.from_dict(body)
    return out


# ---------------------------------------------------------------------------
# leader-board / summary
# ---------------------------------------------------------------------------


def summarise_results(rows_df: pd.DataFrame) -> pd.DataFrame:
    """Rank configurations by total PnL, attaching the headline metrics
    every operator inspection wants.
    """
    if rows_df.empty:
        return rows_df
    out = rows_df.copy()
    out = out.sort_values(["total_pnl", "sharpe"], ascending=[False, False])
    cols = [
        "strategy", "symbol", "resolution", "leverage",
        "num_trades", "win_rate", "expectancy", "total_pnl",
        "profit_factor", "max_drawdown", "sharpe",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def write_summary_csv(rows_df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarise_results(rows_df)
    summary.to_csv(path, index=False)
    return path
