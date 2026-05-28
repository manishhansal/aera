"""aera — Autonomous Edge & Risk Arbitrage.

A production-grade Delta Exchange perpetual-futures scalping bot with a
live web dashboard, multi-strategy execution, leverage-aware sizing, hard
risk caps, and absolute-USD take-profit / stop-loss controls.

The package is split into:
    core/        portfolio, risk, greedy autopilot, compounding math
    markets/     Delta exchange REST + websocket client
    signals/     microstructure features (z-score, OFI, tape, VWAP)
    strategies/  Delta perpetual scalping strategies
    execution/   order routing, slippage, Delta paper/live exchanges
    dashboard/   FastAPI + WebSocket live UI
"""

__version__ = "0.1.0"
__title__ = "aera"
__tagline__ = "Autonomous Edge & Risk Arbitrage"
