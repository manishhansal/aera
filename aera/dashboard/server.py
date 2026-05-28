"""FastAPI app exposing the dashboard state.

Endpoints
---------

GET  /                       single-page UI (HTML + JS + CSS)
GET  /api/state              full snapshot (engine, portfolio, strategies)
GET  /api/fills?limit=50     recent fills (most recent first)
GET  /api/trades?limit=50    completed round-trip trades (most recent first)
GET  /api/signals?limit=50   recent signals (most recent first)
GET  /api/positions          open positions
GET  /api/equity?limit=500   equity-curve points
GET  /api/markets            top markets currently in the universe
POST /api/control/pause      pause the engine (no signals fire while paused)
POST /api/control/resume     un-pause the engine
WS   /ws                     pushes a snapshot every ``push_interval_ms``

The dashboard is designed to be safe to run alongside the bot in production:
all routes are read-only except for the explicit /api/control/* endpoints,
which only flip an in-memory pause flag.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aera.logging import get_logger

from .state import DashboardState


log = get_logger(__name__)


STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(state: DashboardState, *, push_interval_ms: int = 1000) -> FastAPI:
    """Build the FastAPI app bound to ``state``.

    The app is a thin shell around the state container — every endpoint reads
    from `state` and returns JSON, except `/` which serves the SPA shell.
    """
    # ------------------------------------------------------------------
    # equity-curve sampler — pushes one point per second so the chart has
    # data even when no trades happen
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sampler_task = asyncio.create_task(
            _equity_sampler(state), name="dashboard-equity-sampler"
        )
        try:
            yield
        finally:
            sampler_task.cancel()
            try:
                await sampler_task
            except (asyncio.CancelledError, Exception):
                pass

    app = FastAPI(
        title="aera dashboard",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # static SPA
    # ------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=STATIC_DIR),
            name="static",
        )

    # ------------------------------------------------------------------
    # JSON state
    # ------------------------------------------------------------------

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.get("/api/fills")
    async def get_fills(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
        return JSONResponse({"fills": state.recent_fills(limit)})

    @app.get("/api/trades")
    async def get_trades(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
        return JSONResponse({"trades": state.recent_trades(limit)})

    @app.get("/api/signals")
    async def get_signals(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
        return JSONResponse({"signals": state.recent_signals(limit)})

    @app.get("/api/positions")
    async def get_positions() -> JSONResponse:
        return JSONResponse({"positions": state.open_positions()})

    @app.get("/api/equity")
    async def get_equity(limit: int = Query(500, ge=10, le=5000)) -> JSONResponse:
        return JSONResponse({"points": state.equity_history(limit)})

    @app.get("/api/markets")
    async def get_markets() -> JSONResponse:
        return JSONResponse({"markets": state.top_markets})

    # ------------------------------------------------------------------
    # control endpoints
    # ------------------------------------------------------------------

    @app.post("/api/control/pause")
    async def pause_engine() -> JSONResponse:
        if state.engine is None:
            return JSONResponse({"ok": False, "reason": "engine not running"}, status_code=409)
        state.engine.pause()
        return JSONResponse({"ok": True, "paused": True})

    @app.post("/api/control/resume")
    async def resume_engine() -> JSONResponse:
        if state.engine is None:
            return JSONResponse({"ok": False, "reason": "engine not running"}, status_code=409)
        state.engine.resume()
        return JSONResponse({"ok": True, "paused": False})

    # ------------------------------------------------------------------
    # websocket push
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        interval = max(0.1, push_interval_ms / 1000.0)
        try:
            while True:
                payload = {
                    "type": "tick",
                    "state": state.snapshot(),
                    "fills": state.recent_fills(15),
                    "trades": state.recent_trades(15),
                    "signals": state.recent_signals(15),
                    "positions": state.open_positions(),
                    "equity": state.equity_history(300),
                }
                await websocket.send_text(json.dumps(payload, default=_json_default))
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            return
        except Exception as exc:  # pragma: no cover - dashboard is best-effort
            log.debug("ws closed: %s", exc)

    return app


async def _equity_sampler(state: DashboardState, *, period_seconds: float = 1.0) -> None:
    """Background task that snapshots the bankroll once a second."""
    try:
        while True:
            state.record_equity_sample()
            await asyncio.sleep(period_seconds)
    except asyncio.CancelledError:
        return


def _json_default(obj):  # pragma: no cover - tiny helper
    try:
        return float(obj)
    except Exception:
        return str(obj)


async def run_dashboard(
    state: DashboardState,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    push_interval_ms: int = 1000,
    log_level: str = "warning",
) -> None:
    """Run the dashboard server in the current event loop.

    Designed to be launched via ``asyncio.create_task`` so it co-exists with
    the trading engine in a single process.
    """
    import uvicorn

    app = create_app(state, push_interval_ms=push_interval_ms)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()
