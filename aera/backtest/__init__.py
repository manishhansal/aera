"""Bar-replay backtesting engine + parameter sweep + analysis."""
from .replay import (
    BacktestResult,
    BarReplay,
    TradeRecord,
    candles_to_market_stream,
)
from .sweep import (
    SweepConfig,
    SweepResult,
    SweepRow,
    run_sweep,
)
from .analysis import (
    HourMap,
    build_hour_map,
    summarise_results,
    write_summary_csv,
)

__all__ = [
    "BacktestResult",
    "BarReplay",
    "TradeRecord",
    "candles_to_market_stream",
    "SweepConfig",
    "SweepResult",
    "SweepRow",
    "run_sweep",
    "HourMap",
    "build_hour_map",
    "summarise_results",
    "write_summary_csv",
]
