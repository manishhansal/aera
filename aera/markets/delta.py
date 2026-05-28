"""Delta Exchange async REST client.

Talks to Delta's v2 REST API (https://docs.delta.exchange/#introduction).

Public surface (no credentials needed):
    list_products()           GET  /v2/products
    list_tickers()            GET  /v2/tickers
    fetch_orderbook(symbol)   GET  /v2/l2orderbook/{symbol}
    fetch_books_batch(syms)   parallel /v2/l2orderbook calls

Authenticated surface (needs api_key + api_secret):
    get_wallet_balances()     GET  /v2/wallet/balances
    get_positions()           GET  /v2/positions/margined
    place_order(...)          POST /v2/orders
    cancel_order(order_id)    DELETE /v2/orders/{id}
    get_orders()              GET  /v2/orders

The class is structured as a typical async REST client (async context
manager, ``aiolimiter`` rate-limited, tenacity-retried, defensive parsers)
so the engine layer treats both venues uniformly.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Iterable, List, Optional

import httpx
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from aera.logging import get_logger
from aera.settings import DeltaConfig

from .base import Market, Outcome
from .delta_signing import auth_headers
from .orderbook import OrderBook


log = get_logger(__name__)


# Side label used by Delta when there is exactly one tradeable outcome per
# market (a perp / future / option). This lets us map the venue onto the same
# Market/Outcome abstraction the rest of the bot expects, while preserving directionality:
#   BUY  the LONG outcome => open long
#   SELL the LONG outcome => open short / close long
DELTA_OUTCOME_LABEL = "LONG"


class DeltaError(RuntimeError):
    pass


class DeltaClient:
    """Async REST client for Delta Exchange.

    Usage::

        async with DeltaClient(cfg) as delta:
            products = await delta.list_products()
            book = await delta.fetch_orderbook("BTCUSD")
    """

    def __init__(self, cfg: DeltaConfig) -> None:
        self.cfg = cfg
        # Delta's public rate limit is generous; cap ourselves at 8 req/s to
        # stay well under it and avoid any 429s during burst scans.
        self._limiter = AsyncLimiter(max_rate=8, time_period=1.0)
        self._client: Optional[httpx.AsyncClient] = None
        self._products_cache: Optional[List[dict]] = None
        self._cache_at: float = 0.0
        self._product_id_by_symbol: Dict[str, int] = {}
        # Filled by ``list_products``; keyed by symbol. Used by the executor
        # to convert USD notional → integer Delta contracts correctly.
        # On Delta, 1 contract = ``contract_value`` units of the underlying.
        self._contract_value_by_symbol: Dict[str, float] = {}
        self._initial_margin_pct_by_symbol: Dict[str, float] = {}
        self._maintenance_margin_pct_by_symbol: Dict[str, float] = {}

    async def __aenter__(self) -> "DeltaClient":
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise DeltaError("Use DeltaClient as an async context manager")
        return self._client

    @property
    def authenticated(self) -> bool:
        return bool(self.cfg.api_key and self.cfg.api_secret)

    # ------------------------------------------------------------------
    # low-level request helpers
    # ------------------------------------------------------------------

    @retry(
        reraise=True,
        retry=retry_if_exception_type((httpx.HTTPError, DeltaError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, max=5.0),
    )
    async def _request(
        self,
        *,
        method: str,
        path: str,
        query: Optional[dict] = None,
        body: Any = None,
        auth: bool = False,
    ) -> Any:
        url = f"{self.cfg.base_url.rstrip('/')}{path}"
        headers = {"User-Agent": self.cfg.user_agent, "Accept": "application/json"}
        content: Optional[str] = None
        if auth:
            if not self.authenticated:
                raise DeltaError(
                    f"authenticated call to {path} but DELTA_API_KEY/SECRET not set"
                )
            headers, content = auth_headers(
                api_key=self.cfg.api_key or "",
                api_secret=self.cfg.api_secret or "",
                method=method,
                path=path,
                query=query,
                body=body,
                user_agent=self.cfg.user_agent,
            )
        elif body is not None:
            content = json.dumps(body, separators=(",", ":"), sort_keys=True)
            headers["Content-Type"] = "application/json"

        async with self._limiter:
            r = await self.client.request(
                method,
                url,
                params=query,
                content=content,
                headers=headers,
            )
        if r.status_code == 429:
            raise DeltaError("rate limited")
        if r.status_code >= 500:
            raise DeltaError(f"server error {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise DeltaError(f"http {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
        except json.JSONDecodeError as exc:
            raise DeltaError(f"bad json from {url}") from exc
        # Delta wraps responses in {"success": true, "result": ...}
        if isinstance(data, dict) and "success" in data:
            if not data.get("success", True):
                raise DeltaError(
                    f"delta returned success=false: {str(data.get('error') or data)[:200]}"
                )
            return data.get("result", data)
        return data

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    async def list_products(
        self,
        *,
        contract_types: Optional[List[str]] = None,
        states: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> List[dict]:
        """Return Delta product descriptors, filtered by contract type/state.

        Cached for 60s so a tight scan loop doesn't hammer the public endpoint.
        On every refresh we also rebuild the symbol → product_id /
        contract_value / margin-% caches.
        """
        now = time.time()
        if self._products_cache is None or force_refresh or (now - self._cache_at) > 60:
            data = await self._request(method="GET", path="/v2/products")
            self._products_cache = list(data) if isinstance(data, list) else []
            self._cache_at = now
            self._rebuild_caches(self._products_cache)
        out = self._products_cache or []
        if contract_types:
            ct = set(contract_types)
            out = [p for p in out if p.get("contract_type") in ct]
        if states:
            st = set(states)
            out = [p for p in out if p.get("state") in st]
        return out

    def _rebuild_caches(self, products: List[dict]) -> None:
        self._product_id_by_symbol = {}
        self._contract_value_by_symbol = {}
        self._initial_margin_pct_by_symbol = {}
        self._maintenance_margin_pct_by_symbol = {}
        for p in products:
            sym = p.get("symbol")
            pid = p.get("id")
            if not sym or pid is None:
                continue
            sym = str(sym)
            try:
                self._product_id_by_symbol[sym] = int(pid)
            except (TypeError, ValueError):
                continue
            cv = p.get("contract_value")
            if cv is not None:
                try:
                    self._contract_value_by_symbol[sym] = float(cv)
                except (TypeError, ValueError):
                    pass
            im = p.get("initial_margin")
            if im is not None:
                try:
                    # Delta returns initial_margin as a percent string, e.g. "1" -> 1%.
                    self._initial_margin_pct_by_symbol[sym] = float(im) / 100.0
                except (TypeError, ValueError):
                    pass
            mm = p.get("maintenance_margin")
            if mm is not None:
                try:
                    self._maintenance_margin_pct_by_symbol[sym] = float(mm) / 100.0
                except (TypeError, ValueError):
                    pass

    async def product_id_for(self, symbol: str) -> int:
        """Look up Delta's numeric product_id for a symbol like BTCUSD."""
        if symbol in self._product_id_by_symbol:
            return self._product_id_by_symbol[symbol]
        await self.list_products(force_refresh=True)
        if symbol not in self._product_id_by_symbol:
            raise DeltaError(f"unknown Delta symbol: {symbol}")
        return self._product_id_by_symbol[symbol]

    def contract_value_for(self, symbol: str, default: float = 1.0) -> float:
        """Units of underlying per 1 contract for ``symbol``.

        Falls back to ``default`` (1.0 = "1 contract is 1 underlying unit")
        when the product hasn't been loaded yet. Most callers should make
        sure ``list_products`` has run at least once before relying on this.
        """
        return self._contract_value_by_symbol.get(symbol, default)

    def initial_margin_pct_for(self, symbol: str, default: float = 0.01) -> float:
        """Fractional initial margin (e.g. 0.01 = 1%)."""
        return self._initial_margin_pct_by_symbol.get(symbol, default)

    def maintenance_margin_pct_for(self, symbol: str, default: float = 0.005) -> float:
        return self._maintenance_margin_pct_by_symbol.get(symbol, default)

    async def list_active_markets(
        self,
        *,
        symbols: Optional[List[str]] = None,
        contract_types: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Market]:
        """Return a list of `Market` objects for Delta products.

        Each Delta product (e.g. BTCUSD perpetual) maps to one Market with a
        single "LONG" outcome — strategies that need binary structure won't
        fit, but directional strategies (mean-reversion, momentum, basis arb)
        slot in naturally.
        """
        cts = contract_types or self.cfg.contract_types
        products = await self.list_products(contract_types=cts, states=["live"])
        wanted = set(symbols or self.cfg.symbols or [])
        if wanted:
            products = [p for p in products if p.get("symbol") in wanted]
        markets: List[Market] = []
        for raw in products[:limit]:
            try:
                markets.append(self._parse_product(raw))
            except Exception as exc:
                log.debug("skip product: %s", exc)
        return markets

    def _parse_product(self, raw: dict) -> Market:
        symbol = str(raw.get("symbol") or raw.get("id"))
        slug = symbol.lower()
        question = str(raw.get("description") or symbol)
        category = str(raw.get("contract_type") or "delta")
        end_iso = raw.get("settlement_time")
        end_time: Optional[float] = None
        if end_iso:
            try:
                from datetime import datetime

                end_time = datetime.fromisoformat(
                    str(end_iso).replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                end_time = None

        # Tick size: Delta returns a string in "tick_size".
        tick = raw.get("tick_size")
        try:
            min_tick = float(tick) if tick is not None else 0.5
        except Exception:
            min_tick = 0.5

        # contract_value, margin %s, product_id — needed downstream so the
        # paper and live exchanges can size correctly without re-hitting the
        # REST endpoint.
        def _f(name: str, default: float | None = None) -> float | None:
            v = raw.get(name)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        # Resolve a single ``leverage`` figure to attach to the Market so the
        # generic executor / risk manager can size against
        # ``bankroll × leverage`` (i.e. buying power) instead of just
        # ``bankroll`` (margin). Priority:
        #   1. account-level ``cfg.leverage`` if set (this is what the engine
        #      pushes to Delta via change-leverage at startup),
        #   2. product's ``default_leverage`` from the API,
        #   3. ``1 / initial_margin_pct`` as a conservative lower bound,
        #   4. ``1.0`` (no leverage / spot semantics).
        initial_margin_pct_val = (_f("initial_margin", 1.0) or 1.0) / 100.0
        default_lev = _f("default_leverage")
        leverage: float
        if self.cfg.leverage is not None and self.cfg.leverage > 0:
            leverage = float(self.cfg.leverage)
        elif default_lev is not None and default_lev > 0:
            leverage = float(default_lev)
        elif initial_margin_pct_val > 0:
            leverage = 1.0 / initial_margin_pct_val
        else:
            leverage = 1.0

        metadata: Dict[str, object] = {
            "delta_product_id": int(raw["id"]) if raw.get("id") is not None else None,
            "contract_value": _f("contract_value", 1.0),
            "contract_type": raw.get("contract_type"),
            "initial_margin_pct": initial_margin_pct_val,
            "maintenance_margin_pct": (_f("maintenance_margin", 0.5) or 0.5) / 100.0,
            "default_leverage": default_lev,
            "max_leverage_notional": _f("max_leverage_notional"),
            "leverage": leverage,
            "contract_unit_currency": raw.get("contract_unit_currency"),
            "notional_type": raw.get("notional_type"),
        }

        outcome = Outcome(id=symbol, label=DELTA_OUTCOME_LABEL)
        return Market(
            id=symbol,
            slug=slug,
            question=question,
            category=category,
            outcomes={symbol: outcome},
            end_time=end_time,
            venue="delta",
            tags=[raw.get("underlying_asset", {}).get("symbol", "")] if isinstance(raw.get("underlying_asset"), dict) else [],
            minimum_tick=min_tick,
            last_update=time.time(),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # market data
    # ------------------------------------------------------------------

    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        """Snapshot of the L2 order book for one Delta symbol.

        Delta returns sizes in **contracts** (integers) and prices as strings.
        We coerce both to floats; price-units are USD for USD-quoted perps.
        """
        data = await self._request(
            method="GET",
            path=f"/v2/l2orderbook/{symbol}",
            query={"depth": depth},
        )
        book = OrderBook()
        bids_raw = data.get("buy") or data.get("bids") or []
        asks_raw = data.get("sell") or data.get("asks") or []
        bids = [
            (float(l.get("price") or l.get("limit_price") or 0),
             float(l.get("size") or l.get("qty") or 0))
            for l in bids_raw
        ]
        asks = [
            (float(l.get("price") or l.get("limit_price") or 0),
             float(l.get("size") or l.get("qty") or 0))
            for l in asks_raw
        ]
        book.replace(bids=bids, asks=asks)
        return book

    async def fetch_books_batch(self, symbols: List[str]) -> Dict[str, OrderBook]:
        """Fetch books in parallel; Delta has no batch endpoint."""
        if not symbols:
            return {}

        async def one(sym: str) -> tuple[str, Optional[OrderBook]]:
            try:
                return sym, await self.fetch_orderbook(sym)
            except DeltaError as exc:
                log.debug("delta orderbook %s failed: %s", sym, exc)
                return sym, None

        results = await asyncio.gather(*(one(s) for s in symbols))
        return {sym: book for sym, book in results if book is not None}

    async def list_tickers(self) -> List[dict]:
        data = await self._request(method="GET", path="/v2/tickers")
        return list(data) if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # account / trading (authenticated)
    # ------------------------------------------------------------------

    async def get_wallet_balances(self) -> List[dict]:
        data = await self._request(
            method="GET", path="/v2/wallet/balances", auth=True
        )
        return list(data) if isinstance(data, list) else []

    async def get_positions(self) -> List[dict]:
        data = await self._request(
            method="GET", path="/v2/positions/margined", auth=True
        )
        return list(data) if isinstance(data, list) else []

    async def get_orders(self, *, state: str = "open") -> List[dict]:
        data = await self._request(
            method="GET",
            path="/v2/orders",
            query={"state": state},
            auth=True,
        )
        return list(data) if isinstance(data, list) else []

    async def place_order(
        self,
        *,
        symbol: str,
        side: str,
        size: float,
        limit_price: Optional[float] = None,
        order_type: str = "limit_order",
        time_in_force: str = "gtc",
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Place a Delta order.

        ``size`` is in **contracts** (Delta's native unit). For USD-quoted
        perpetuals, 1 contract = $1 of underlying notional, so size = USD
        notional. For inverse contracts (e.g. BTCUSD inverse), size is in
        contracts where each contract is $1 of USD value.
        """
        if not self.authenticated:
            raise DeltaError("DELTA_API_KEY/SECRET required for placing orders")

        side_norm = side.lower()
        if side_norm in ("buy", "long"):
            side_norm = "buy"
        elif side_norm in ("sell", "short"):
            side_norm = "sell"
        else:
            raise DeltaError(f"invalid side: {side!r}")

        # Delta wants the integer product_id, not the symbol alone, for the order body.
        product_id = await self.product_id_for(symbol)

        body: Dict[str, Any] = {
            "product_id": product_id,
            "product_symbol": symbol,
            "size": int(round(size)),
            "side": side_norm,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "post_only": post_only,
            "reduce_only": reduce_only,
        }
        if limit_price is not None and order_type == "limit_order":
            # Delta limit_price must be a string.
            body["limit_price"] = f"{float(limit_price):.8f}".rstrip("0").rstrip(".")
        if client_order_id:
            body["client_order_id"] = client_order_id

        return await self._request(
            method="POST", path="/v2/orders", body=body, auth=True
        )

    async def cancel_order(self, *, order_id: int, symbol: Optional[str] = None) -> dict:
        body: Dict[str, Any] = {"id": int(order_id)}
        if symbol:
            body["product_symbol"] = symbol
        return await self._request(
            method="DELETE", path="/v2/orders", body=body, auth=True
        )

    # ------------------------------------------------------------------
    # leverage
    # ------------------------------------------------------------------

    async def set_leverage(self, *, symbol: str, leverage: float) -> dict:
        """Set account leverage for one Delta product.

        POSTs to ``/v2/products/{product_id}/orders/leverage``. The leverage
        must be between 1× and the product's ``max_leverage`` (typically 100×
        for liquid majors). Idempotent — sending the same value is a no-op.
        """
        if not self.authenticated:
            raise DeltaError("DELTA_API_KEY/SECRET required to change leverage")
        if leverage <= 0:
            raise DeltaError(f"leverage must be > 0, got {leverage!r}")
        pid = await self.product_id_for(symbol)
        path = f"/v2/products/{pid}/orders/leverage"
        # Delta wants leverage as a numeric string ("10", "20").
        body = {"leverage": f"{float(leverage):g}"}
        return await self._request(method="POST", path=path, body=body, auth=True)

    async def apply_account_leverage(
        self,
        leverage: float,
        symbols: Iterable[str],
    ) -> Dict[str, dict]:
        """Convenience helper: set leverage for every configured symbol.

        Failures on individual symbols are logged but never raised — that way
        a missing-product or rate-limit hiccup never blocks the engine from
        starting. The return dict only contains successful responses.
        """
        out: Dict[str, dict] = {}
        for sym in symbols:
            try:
                out[sym] = await self.set_leverage(symbol=sym, leverage=leverage)
                log.info("delta: leverage for %s set to %sx", sym, leverage)
            except Exception as exc:
                log.warning("delta: set_leverage %s @%sx failed: %s", sym, leverage, exc)
        return out
