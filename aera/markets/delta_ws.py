"""Delta Exchange websocket client.

Subscribes to ``wss://socket.delta.exchange/`` and maintains an in-memory
`OrderBook` per symbol from the ``l2_orderbook`` channel.

Delta's websocket protocol (https://docs.delta.exchange/#websocket-feed):

  * Client sends a ``subscribe`` frame listing channels + symbols.
  * Server replies with one snapshot per (channel, symbol), then streams
    incremental updates.
  * The ``l2_orderbook`` channel delivers full L2 snapshots; the
    ``l2_updates`` channel delivers diffs. We default to ``l2_orderbook``
    because it is robust to dropped messages — every frame is a complete
    book, so out-of-order delivery cannot corrupt state.

The class uses the standard websocket pattern: auto-reconnect with exponential
back-off, fan-out via an `asyncio.Queue`, and a `book(symbol)` lookup so
strategies can read the latest book without consuming the stream.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Dict, Iterable, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

from aera.logging import get_logger

from .delta_signing import ws_auth_payload
from .orderbook import OrderBook


log = get_logger(__name__)


DEFAULT_DELTA_WS_URL = "wss://socket.delta.exchange"


@dataclass
class DeltaBookUpdate:
    """Emitted on every book mutation."""

    symbol: str
    book: OrderBook
    received_at: float = field(default_factory=time.time)
    source: str = "l2_orderbook"


class DeltaWebsocket:
    """Async websocket client for Delta Exchange L2 books.

    Usage::

        async with DeltaWebsocket(symbols=["BTCUSD"]) as ws:
            async for upd in ws.updates():
                book = upd.book
                ...
    """

    def __init__(
        self,
        symbols: Iterable[str],
        url: str = DEFAULT_DELTA_WS_URL,
        *,
        channels: Iterable[str] = ("l2_orderbook",),
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        max_queue_size: int = 5000,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        ping_interval: float = 20.0,
    ) -> None:
        self.url = url
        self._symbols: Set[str] = set(symbols)
        self._channels: List[str] = list(channels)
        self._api_key = api_key
        self._api_secret = api_secret
        self._books: Dict[str, OrderBook] = {s: OrderBook() for s in self._symbols}
        self._queue: asyncio.Queue[DeltaBookUpdate] = asyncio.Queue(maxsize=max_queue_size)
        self._reconnect_initial_delay = reconnect_initial_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._ping_interval = ping_interval
        self._stopped = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._ws = None
        self.connected = False
        self.messages_received = 0

    # ---------------------------------------------------------------- public

    async def __aenter__(self) -> "DeltaWebsocket":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="delta-ws")
            for _ in range(20):
                if self.connected:
                    return
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        self._stopped.set()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def add_symbols(self, symbols: Iterable[str]) -> None:
        new = set(symbols) - self._symbols
        if not new:
            return
        self._symbols |= new
        for s in new:
            self._books.setdefault(s, OrderBook())
        # Delta supports incremental subscribe — but the simplest correct
        # behaviour is to reconnect with the union and let the server replay
        # snapshots.
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    def book(self, symbol: str) -> Optional[OrderBook]:
        return self._books.get(symbol)

    def books(self) -> Dict[str, OrderBook]:
        return dict(self._books)

    async def updates(self) -> AsyncIterator[DeltaBookUpdate]:
        while not self._stopped.is_set():
            try:
                upd = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield upd
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------- internal

    async def _run(self) -> None:
        delay = self._reconnect_initial_delay
        while not self._stopped.is_set():
            try:
                await self._connect_and_consume()
                delay = self._reconnect_initial_delay
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("delta-ws connection error: %s; retry in %.1fs", exc, delay)
                self.connected = False
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2.0, self._reconnect_max_delay)

    async def _connect_and_consume(self) -> None:
        log.info("delta-ws connecting to %s (%d symbols)", self.url, len(self._symbols))
        async with websockets.connect(
            self.url,
            ping_interval=self._ping_interval,
            ping_timeout=10,
            max_size=2**22,
        ) as ws:
            self._ws = ws
            self.connected = True
            if self._api_key and self._api_secret:
                auth_frame = ws_auth_payload(
                    api_key=self._api_key, api_secret=self._api_secret
                )
                await ws.send(json.dumps(auth_frame))
            await self._subscribe(ws)
            log.info("delta-ws connected")
            try:
                async for raw in ws:
                    self.messages_received += 1
                    await self._handle_message(raw)
            except ConnectionClosed as exc:
                log.info("delta-ws closed: %s", exc)
            finally:
                self.connected = False
                self._ws = None

    async def _subscribe(self, ws) -> None:
        msg = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": ch, "symbols": sorted(self._symbols)}
                    for ch in self._channels
                ],
            },
        }
        await ws.send(json.dumps(msg))

    async def _handle_message(self, raw: str | bytes) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        # Delta wraps messages as {"type": "...", ...} for control frames and
        # {"symbol": "...", "buy": [...], "sell": [...]} for book frames.
        msg_type = data.get("type")
        if msg_type in ("subscriptions", "auth", "pong", "ping"):
            return
        if msg_type == "l2_orderbook" or (
            "buy" in data and "sell" in data and data.get("symbol")
        ):
            self._apply_book(data)
            return
        if msg_type == "error":
            log.warning("delta-ws error: %s", data)
            return

    def _apply_book(self, ev: dict) -> None:
        symbol = str(ev.get("symbol") or "")
        if not symbol:
            return
        book = self._books.setdefault(symbol, OrderBook())
        bids = [
            (float(l.get("limit_price") or l.get("price") or 0),
             float(l.get("size") or l.get("qty") or 0))
            for l in (ev.get("buy") or ev.get("bids") or [])
        ]
        asks = [
            (float(l.get("limit_price") or l.get("price") or 0),
             float(l.get("size") or l.get("qty") or 0))
            for l in (ev.get("sell") or ev.get("asks") or [])
        ]
        book.replace(bids=bids, asks=asks)
        self._enqueue(DeltaBookUpdate(symbol=symbol, book=book, source="l2_orderbook"))

    def _enqueue(self, upd: DeltaBookUpdate) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(upd)
        except asyncio.QueueFull:
            pass


async def stream_delta_books(
    symbols: Iterable[str],
    on_update: Callable[[DeltaBookUpdate], None],
    *,
    url: str = DEFAULT_DELTA_WS_URL,
    duration_seconds: Optional[float] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> None:
    """One-shot streaming helper that yields Delta book updates."""
    deadline = (time.time() + duration_seconds) if duration_seconds else None
    async with DeltaWebsocket(
        symbols, url=url, api_key=api_key, api_secret=api_secret
    ) as ws:
        async for upd in ws.updates():
            on_update(upd)
            if deadline is not None and time.time() >= deadline:
                return
