"""Grid sweep over (strategy × symbol × resolution × leverage × hour-of-day).

The sweep runs each combination through :class:`BarReplay` and stores
one :class:`SweepRow` per configuration. The CLI driver
(``scripts/sweep_backtest.py``) calls :func:`run_sweep` and writes the
flat ``SweepRow`` table to CSV for downstream analysis.

Parallelism: each (strategy, symbol, resolution, leverage) replay is
independent, so we fan them out across a thread pool. The hour-of-day
slice is applied as a POST-FILTER on the trade list to avoid running
24 replays for a single config — we replay the full window once and
attribute trades to hours based on ``entry_ts``.
"""
from __future__ import annotations

import concurrent.futures
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from aera.logging import get_logger

from .replay import BacktestResult, BarReplay, TradeRecord

log = get_logger(__name__)


@dataclass
class SweepConfig:
    """One axis of the sweep — strategies × symbols × resolutions × leverages."""

    strategies: Dict[str, Callable]   # name → factory(portfolio) -> Strategy
    symbols: Sequence[str]
    resolutions: Sequence[str] = ("1m",)
    leverages: Sequence[float] = (25.0,)
    starting_bankroll: float = 1000.0
    spread_bps: float = 5.0
    slippage_bps: float = 2.0
    taker_fee_bps: float = 5.0
    ticks_per_bar: int = 4
    max_workers: int = 4


@dataclass
class SweepRow:
    """One row of the flat sweep output. Hour-bucketed views are
    derived from this by ``analysis.build_hour_map``."""

    strategy: str
    symbol: str
    resolution: str
    leverage: float
    bars_processed: int
    starting_bankroll: float
    ending_bankroll: float
    total_pnl: float
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: Optional[float]
    expectancy: float
    max_drawdown: float
    sharpe: float
    duration_seconds: float

    @classmethod
    def from_result(cls, res: BacktestResult, duration_seconds: float) -> "SweepRow":
        pf = res.profit_factor
        return cls(
            strategy=res.strategy,
            symbol=res.symbol,
            resolution=res.resolution,
            leverage=res.leverage,
            bars_processed=res.bars_processed,
            starting_bankroll=res.starting_bankroll,
            ending_bankroll=res.ending_bankroll,
            total_pnl=res.total_pnl,
            num_trades=res.num_trades,
            num_wins=res.num_wins,
            num_losses=res.num_losses,
            win_rate=res.win_rate,
            avg_win=res.avg_win,
            avg_loss=res.avg_loss,
            profit_factor=(None if pf == float("inf") else pf),
            expectancy=res.expectancy,
            max_drawdown=res.max_drawdown,
            sharpe=res.sharpe,
            duration_seconds=duration_seconds,
        )

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "resolution": self.resolution,
            "leverage": self.leverage,
            "bars_processed": self.bars_processed,
            "starting_bankroll": self.starting_bankroll,
            "ending_bankroll": self.ending_bankroll,
            "total_pnl": self.total_pnl,
            "num_trades": self.num_trades,
            "num_wins": self.num_wins,
            "num_losses": self.num_losses,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class SweepResult:
    """Aggregate output of a sweep: rows + the per-config trade list."""

    rows: List[SweepRow] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.rows])

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "strategy": t.strategy,
            "symbol": t.symbol,
            "side": t.side,
            "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size_usd": t.size_usd,
            "leverage": t.leverage,
            "pnl_usd": t.pnl_usd,
            "fees_usd": t.fees_usd,
            "hold_seconds": t.hold_seconds,
            "hour_of_day": t.hour_of_day,
            "day_of_week": t.day_of_week,
            "is_win": t.is_win,
            "reason_open": t.reason_open,
            "reason_close": t.reason_close,
        } for t in self.trades])

    def top(self, n: int = 10, by: str = "total_pnl") -> pd.DataFrame:
        df = self.to_df()
        if df.empty:
            return df
        return df.sort_values(by, ascending=False).head(n)


# ---------------------------------------------------------------------------
# core sweep loop
# ---------------------------------------------------------------------------


def _run_single(
    *, strategy_name, factory, symbol, resolution, leverage,
    candles, cfg: SweepConfig,
) -> tuple[BacktestResult, SweepRow, list[TradeRecord]]:
    t0 = time.perf_counter()
    replay = BarReplay(
        factory,
        starting_bankroll=cfg.starting_bankroll,
        symbol=symbol,
        resolution=resolution,
        leverage=leverage,
        spread_bps=cfg.spread_bps,
        slippage_bps=cfg.slippage_bps,
        taker_fee_bps=cfg.taker_fee_bps,
        ticks_per_bar=cfg.ticks_per_bar,
    )
    result = replay.run(candles)
    dt = time.perf_counter() - t0
    row = SweepRow.from_result(result, dt)
    return result, row, list(result.trades)


def run_sweep(
    *,
    cfg: SweepConfig,
    candles_for: Callable[[str, str], pd.DataFrame],
    progress_cb: Optional[Callable[[int, int, SweepRow], None]] = None,
) -> SweepResult:
    """Run every (strategy × symbol × resolution × leverage) tuple
    through the replay engine.

    ``candles_for(symbol, resolution) -> DataFrame`` is the caller-
    supplied data accessor — typically a closure around
    ``aera.data.fetch_history`` so the network round-trips happen
    once per (symbol, resolution) and the per-leverage replays reuse
    the same in-memory frame.
    """
    combos = list(itertools.product(
        cfg.strategies.items(),
        cfg.symbols,
        cfg.resolutions,
        cfg.leverages,
    ))
    log.info("sweep: %d total configurations to backtest", len(combos))

    out = SweepResult()
    done = 0
    total = len(combos)

    # Cache candles per (symbol, resolution) so we don't pull them
    # once per leverage value (leverage doesn't affect bar data).
    candle_cache: Dict[tuple[str, str], pd.DataFrame] = {}

    def _get_candles(symbol: str, resolution: str) -> pd.DataFrame:
        key = (symbol, resolution)
        if key not in candle_cache:
            candle_cache[key] = candles_for(symbol, resolution)
        return candle_cache[key]

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = {}
        for (strategy_name, factory), symbol, resolution, leverage in combos:
            try:
                candles = _get_candles(symbol, resolution)
            except Exception as exc:
                log.error("sweep: skipping %s/%s (%s): no candles available — %s",
                          symbol, resolution, strategy_name, exc)
                continue
            if candles is None or (hasattr(candles, "empty") and candles.empty):
                log.warning(
                    "sweep: empty candle set for %s/%s — skipping",
                    symbol, resolution,
                )
                continue
            fut = ex.submit(
                _run_single,
                strategy_name=strategy_name, factory=factory,
                symbol=symbol, resolution=resolution, leverage=leverage,
                candles=candles, cfg=cfg,
            )
            futures[fut] = (strategy_name, symbol, resolution, leverage)

        for fut in concurrent.futures.as_completed(futures):
            sn, sym, res, lev = futures[fut]
            try:
                _, row, trades = fut.result()
                out.rows.append(row)
                out.trades.extend(trades)
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, row)
                log.info(
                    "sweep [%d/%d] %s/%s/%s lev=%.1f → PnL=$%+.2f WR=%.1f%% "
                    "trades=%d Sharpe=%.2f",
                    done, total, sn, sym, res, lev,
                    row.total_pnl, row.win_rate * 100, row.num_trades, row.sharpe,
                )
            except Exception as exc:
                log.exception("sweep: %s/%s/%s lev=%.1f crashed: %s",
                              sn, sym, res, lev, exc)
    return out
