"""Live web dashboard for the aera trading bot.

Wraps the running `DeltaEngine` with a FastAPI app that exposes JSON state
snapshots, a WebSocket push stream, and a static single-page UI with charts
and live trading tables.

Entry point: ``python -m scripts.run_dashboard``.
"""
from .state import DashboardState, FillEvent, SignalEvent, EquityPoint
from .server import create_app, run_dashboard

__all__ = [
    "DashboardState",
    "FillEvent",
    "SignalEvent",
    "EquityPoint",
    "create_app",
    "run_dashboard",
]
