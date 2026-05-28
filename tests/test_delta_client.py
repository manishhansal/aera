"""Delta REST client tests using a mocked httpx transport.

We never touch the real Delta endpoint — every request is intercepted by an
``httpx.MockTransport`` and asserted against. This catches signature drift,
schema-parsing regressions, and error-handling bugs without needing API
credentials or network.
"""
from __future__ import annotations

import hmac
import hashlib
import json
from typing import Any, Dict, List

import httpx
import pytest

from aera.markets.delta import DELTA_OUTCOME_LABEL, DeltaClient, DeltaError
from aera.settings import DeltaConfig


SECRET = "secret_abc"
KEY = "key_xyz"


def _ok(result: Any) -> dict:
    return {"success": True, "result": result}


def _make_client(handler) -> DeltaClient:
    """Build a DeltaClient with its http session pointed at a MockTransport."""
    cfg = DeltaConfig(
        base_url="https://mock.delta",
        api_key=KEY,
        api_secret=SECRET,
        user_agent="aera-tests/1.0",
    )
    client = DeltaClient(cfg)
    transport = httpx.MockTransport(handler)
    client._client = httpx.AsyncClient(transport=transport, base_url="https://mock.delta")
    return client


@pytest.mark.asyncio
async def test_list_products_caches_and_filters():
    products = [
        {"id": 27, "symbol": "BTCUSD", "contract_type": "perpetual_futures", "state": "live"},
        {"id": 28, "symbol": "ETHUSD", "contract_type": "perpetual_futures", "state": "live"},
        {"id": 29, "symbol": "BTC-22FEB", "contract_type": "futures", "state": "live"},
        {"id": 30, "symbol": "DOTUSD", "contract_type": "perpetual_futures", "state": "expired"},
    ]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_ok(products))

    client = _make_client(handler)
    perps_live = await client.list_products(
        contract_types=["perpetual_futures"], states=["live"]
    )
    assert {p["symbol"] for p in perps_live} == {"BTCUSD", "ETHUSD"}
    # Cache hit — no extra HTTP call.
    perps_live2 = await client.list_products(
        contract_types=["perpetual_futures"], states=["live"]
    )
    assert {p["symbol"] for p in perps_live2} == {"BTCUSD", "ETHUSD"}
    assert calls["n"] == 1

    # Symbol -> product_id lookup uses the same cache.
    assert await client.product_id_for("BTCUSD") == 27


@pytest.mark.asyncio
async def test_list_active_markets_yields_delta_markets():
    products = [
        {"id": 27, "symbol": "BTCUSD", "contract_type": "perpetual_futures",
         "state": "live", "tick_size": "0.5", "description": "BTC perpetual"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok(products))

    client = _make_client(handler)
    markets = await client.list_active_markets(symbols=["BTCUSD"])
    assert len(markets) == 1
    m = markets[0]
    assert m.id == "BTCUSD"
    assert m.venue == "delta"
    assert m.minimum_tick == 0.5
    outcome = next(iter(m.outcomes.values()))
    assert outcome.label == DELTA_OUTCOME_LABEL


@pytest.mark.asyncio
async def test_fetch_orderbook_parses_l2():
    response = _ok({
        "symbol": "BTCUSD",
        "buy": [{"price": "100000", "size": "5"}, {"price": "99999", "size": "10"}],
        "sell": [{"price": "100001", "size": "4"}, {"price": "100002", "size": "8"}],
    })

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/v2/l2orderbook/BTCUSD" in request.url.path
        return httpx.Response(200, json=response)

    client = _make_client(handler)
    book = await client.fetch_orderbook("BTCUSD")
    assert book.best_bid_price() == 100000.0
    assert book.best_ask_price() == 100001.0


@pytest.mark.asyncio
async def test_authenticated_request_signs_headers_correctly():
    # We assert the signature header matches our recomputed expected value,
    # which catches any drift in canonical-payload assembly.
    captured: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # GET on /v2/wallet/balances, no body
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["api-key"] = request.headers.get("api-key")
        captured["signature"] = request.headers.get("signature")
        captured["timestamp"] = request.headers.get("timestamp")
        return httpx.Response(200, json=_ok([{"asset_symbol": "USDT", "balance": "100"}]))

    client = _make_client(handler)
    out = await client.get_wallet_balances()
    assert out == [{"asset_symbol": "USDT", "balance": "100"}]
    assert captured["api-key"] == KEY
    # Recompute signature from the timestamp the client used.
    ts = captured["timestamp"]
    payload = f"GET{ts}/v2/wallet/balances"
    expected = hmac.new(
        SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert captured["signature"] == expected


@pytest.mark.asyncio
async def test_place_order_serialises_body_with_string_limit_price():
    # Make sure size is integer-coerced, side normalised, limit_price stringified.
    captured: Dict[str, Any] = {}
    products = [
        {"id": 27, "symbol": "BTCUSD", "contract_type": "perpetual_futures", "state": "live"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/products":
            return httpx.Response(200, json=_ok(products))
        if request.url.path == "/v2/orders" and request.method == "POST":
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json=_ok({
                "id": 42,
                "average_fill_price": "100100",
                "filled_size": 2,
            }))
        return httpx.Response(404, json={"success": False, "error": "not found"})

    client = _make_client(handler)
    resp = await client.place_order(
        symbol="BTCUSD", side="long", size=2, limit_price=100100.0
    )
    assert resp["id"] == 42
    body = captured["body"]
    assert body["side"] == "buy"
    assert body["size"] == 2
    assert body["product_id"] == 27
    assert body["limit_price"] == "100100"
    assert body["product_symbol"] == "BTCUSD"


@pytest.mark.asyncio
async def test_error_responses_raise_delta_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"success": false, "error": "bad request"}')

    client = _make_client(handler)
    with pytest.raises(DeltaError):
        await client.fetch_orderbook("BTCUSD")


@pytest.mark.asyncio
async def test_unauthenticated_attempt_raises_clean_error():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no http call should happen")

    cfg = DeltaConfig(base_url="https://mock.delta", api_key=None, api_secret=None)
    client = DeltaClient(cfg)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://mock.delta",
    )
    with pytest.raises(DeltaError, match="DELTA_API_KEY"):
        await client.place_order(symbol="BTCUSD", side="buy", size=1)


# --------------------------------------------------------------------------
# product metadata / contract_value / leverage
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_products_caches_contract_value_and_margin():
    products = [
        {"id": 27, "symbol": "BTCUSDT", "contract_type": "perpetual_futures",
         "state": "live", "contract_value": "0.001", "initial_margin": "1",
         "maintenance_margin": "0.5"},
        {"id": 3136, "symbol": "ETHUSDT", "contract_type": "perpetual_futures",
         "state": "live", "contract_value": "0.01", "initial_margin": "1",
         "maintenance_margin": "0.5"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok(products))

    client = _make_client(handler)
    await client.list_products()
    assert client.contract_value_for("BTCUSDT") == 0.001
    assert client.contract_value_for("ETHUSDT") == 0.01
    assert client.contract_value_for("UNKNOWN", default=42.0) == 42.0
    assert client.initial_margin_pct_for("BTCUSDT") == pytest.approx(0.01)
    assert client.maintenance_margin_pct_for("BTCUSDT") == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_market_metadata_populated_from_product_spec():
    products = [
        {"id": 27, "symbol": "BTCUSDT", "contract_type": "perpetual_futures",
         "state": "live", "tick_size": "0.5", "contract_value": "0.001",
         "initial_margin": "1", "maintenance_margin": "0.5",
         "default_leverage": "100", "max_leverage_notional": "200000",
         "notional_type": "vanilla", "contract_unit_currency": "BTC"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok(products))

    client = _make_client(handler)
    markets = await client.list_active_markets(symbols=["BTCUSDT"])
    assert len(markets) == 1
    md = markets[0].metadata
    assert md["delta_product_id"] == 27
    assert md["contract_value"] == 0.001
    assert md["initial_margin_pct"] == pytest.approx(0.01)
    assert md["maintenance_margin_pct"] == pytest.approx(0.005)
    assert md["default_leverage"] == 100.0
    assert md["contract_type"] == "perpetual_futures"


@pytest.mark.asyncio
async def test_set_leverage_signs_and_targets_correct_endpoint():
    products = [
        {"id": 27, "symbol": "BTCUSDT", "contract_type": "perpetual_futures",
         "state": "live"},
    ]
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/products":
            return httpx.Response(200, json=_ok(products))
        if request.url.path == "/v2/products/27/orders/leverage":
            captured["method"] = request.method
            captured["body"] = json.loads(request.content.decode("utf-8"))
            captured["api-key"] = request.headers.get("api-key")
            captured["signature"] = request.headers.get("signature")
            return httpx.Response(200, json=_ok({"leverage": "20"}))
        return httpx.Response(404, json={"success": False, "error": "not found"})

    client = _make_client(handler)
    resp = await client.set_leverage(symbol="BTCUSDT", leverage=20)
    assert resp == {"leverage": "20"}
    assert captured["method"] == "POST"
    assert captured["body"] == {"leverage": "20"}
    assert captured["api-key"] == KEY
    # Signature should be a non-empty hex string.
    assert captured["signature"] and len(captured["signature"]) == 64


@pytest.mark.asyncio
async def test_apply_account_leverage_swallows_individual_failures():
    products = [
        {"id": 27, "symbol": "BTCUSDT", "contract_type": "perpetual_futures",
         "state": "live"},
        {"id": 3136, "symbol": "ETHUSDT", "contract_type": "perpetual_futures",
         "state": "live"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/products":
            return httpx.Response(200, json=_ok(products))
        if request.url.path == "/v2/products/27/orders/leverage":
            return httpx.Response(200, json=_ok({"leverage": "10"}))
        if request.url.path == "/v2/products/3136/orders/leverage":
            return httpx.Response(400, text='{"success": false, "error": "bad"}')
        return httpx.Response(404, json={"success": False, "error": "not found"})

    client = _make_client(handler)
    out = await client.apply_account_leverage(10, ["BTCUSDT", "ETHUSDT"])
    assert "BTCUSDT" in out
    assert "ETHUSDT" not in out
