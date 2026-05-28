"""DeltaPaperExchange + DeltaLiveExchange unit tests, including the
contract-aware sizer (`size_in_contracts`) and the new overshoot guard.

Paper-exchange tests run synchronously against a synthetic OrderBook;
live-exchange tests use a fake ``DeltaClient`` to verify USD-notional →
contract-count conversion, contract_value lookup, and Fill construction
without touching the network.
"""
from __future__ import annotations

import pytest

from aera.execution import DeltaLiveExchange, DeltaPaperExchange, OrderRejected
from aera.execution.delta_exchange import (
    _contract_value_from_market,
    size_in_contracts,
)
from aera.execution.slippage import LinearSlippageModel
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.strategies import Leg


def _make_market(
    symbol: str = "BTCUSDT",
    bid_p: float = 100000.0,
    ask_p: float = 100001.0,
    bid_sz: float = 100.0,
    ask_sz: float = 100.0,
    contract_value: float = 0.001,
) -> Market:
    book = OrderBook()
    book.replace(
        bids=[(bid_p, bid_sz), (bid_p - 1, bid_sz * 0.5)],
        asks=[(ask_p, ask_sz), (ask_p + 1, ask_sz * 0.5)],
    )
    return Market(
        id=symbol,
        slug=symbol.lower(),
        question=f"{symbol} perp",
        category="perpetual_futures",
        outcomes={symbol: Outcome(id=symbol, label=DELTA_OUTCOME_LABEL, book=book)},
        venue="delta",
        metadata={
            "contract_value": contract_value,
            "initial_margin_pct": 0.01,
            "maintenance_margin_pct": 0.005,
        },
    )


# --------------------------------------------------------------------------
# size_in_contracts
# --------------------------------------------------------------------------


def test_sizer_btc_overshoot_is_rejected_for_tiny_intent():
    # $5 of intended notional against a 1-contract size of ~$76.
    s = size_in_contracts(
        notional_usd=5.0,
        price=76000.0,
        contract_value=0.001,
        min_trade_notional_usd=1.0,
        max_notional_overshoot=1.5,
    )
    assert s.rejected, s
    assert s.contracts == 0
    assert "overshoots" in s.reason


def test_sizer_btc_accepts_when_intent_exceeds_one_contract():
    # $100 of intent → ~1.32 contracts → rounds to 1 → $76 actual.
    # That's an *under*shoot of intent (acceptable, no guard triggers).
    s = size_in_contracts(
        notional_usd=100.0,
        price=76000.0,
        contract_value=0.001,
    )
    assert not s.rejected, s.reason
    assert s.contracts == 1
    assert s.actual_notional_usd == pytest.approx(76.0)


def test_sizer_btc_rounds_to_nearest_contract():
    s = size_in_contracts(
        notional_usd=300.0,
        price=76000.0,
        contract_value=0.001,
    )
    # 300 / 76 = 3.947 -> 4 contracts -> $304 of notional
    assert s.contracts == 4
    assert s.actual_notional_usd == pytest.approx(304.0)


def test_sizer_eth_default_path():
    # ETHUSDT: contract_value=0.01, mark=$2,000 -> 1 contract = $20.
    s = size_in_contracts(notional_usd=100.0, price=2000.0, contract_value=0.01)
    assert s.contracts == 5
    assert s.actual_notional_usd == pytest.approx(100.0)


def test_sizer_skips_below_min_notional():
    s = size_in_contracts(
        notional_usd=0.5,
        price=2000.0,
        contract_value=0.01,
        min_trade_notional_usd=1.0,
    )
    assert s.rejected
    assert "below min" in s.reason


def test_sizer_rejects_zero_or_negative_inputs():
    assert size_in_contracts(
        notional_usd=10, price=0, contract_value=0.001
    ).rejected
    assert size_in_contracts(
        notional_usd=10, price=100, contract_value=0
    ).rejected


def test_sizer_legacy_unit_one_underlying_per_contract():
    # contract_value=1 reproduces the previous "1 contract = 1 underlying"
    # convention used by other venues.
    s = size_in_contracts(notional_usd=500.0, price=100.0, contract_value=1.0)
    assert s.contracts == 5
    assert s.actual_notional_usd == pytest.approx(500.0)


def test_contract_value_from_market_reads_metadata():
    m = _make_market(contract_value=0.001)
    assert _contract_value_from_market(m) == 0.001

    m2 = Market(id="X", slug="x", question="q", category="c", venue="delta")
    # No metadata → fallback default
    assert _contract_value_from_market(m2) == 1.0


# --------------------------------------------------------------------------
# DeltaPaperExchange
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_exchange_uses_actual_contract_notional_for_fill():
    market = _make_market(contract_value=0.001)  # BTCUSDT-style
    px = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=10.0,   # allow the test trade through
    )
    leg = Leg(
        market_id="BTCUSDT", outcome_id="BTCUSDT",
        side="BUY", limit_price=100001.0, size_usd=200.0, reason="test",
    )
    fill = await px.submit(leg, market)
    assert fill is not None
    # 200 / (0.001 * 100001) ≈ 2 contracts -> $200.002 of actual notional
    # 200.002 / fill_price (~100001) -> "shares" ≈ 0.002
    assert fill.size == pytest.approx(0.002, rel=1e-3)
    assert fill.side == "BUY"


@pytest.mark.asyncio
async def test_paper_exchange_rejects_overshooting_tiny_trade():
    market = _make_market(contract_value=0.001)  # 1 contract ≈ $100
    px = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=1.5,
    )
    # $5 intent against $100 minimum -> 20× overshoot, must be rejected.
    leg = Leg(
        market_id="BTCUSDT", outcome_id="BTCUSDT",
        side="BUY", limit_price=100001.0, size_usd=5.0, reason="test",
    )
    with pytest.raises(OrderRejected, match="overshoots"):
        await px.submit(leg, market)


@pytest.mark.asyncio
async def test_paper_exchange_unknown_outcome_raises():
    market = _make_market()
    px = DeltaPaperExchange()
    leg = Leg(market_id="BTCUSDT", outcome_id="NOPE", side="BUY",
              limit_price=100001.0, size_usd=100.0)
    with pytest.raises(OrderRejected, match="unknown outcome"):
        await px.submit(leg, market)


# --------------------------------------------------------------------------
# DeltaLiveExchange (fake client)
# --------------------------------------------------------------------------


class _FakeDeltaClient:
    """Stand-in for DeltaClient that records the args of ``place_order``."""

    authenticated = True

    def __init__(self, contract_value: float = 0.001) -> None:
        self.calls: list[dict] = []
        self._cv = contract_value

    def contract_value_for(self, symbol: str, default: float = 1.0) -> float:
        return self._cv

    async def place_order(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "id": 999,
            "average_fill_price": str(kwargs["limit_price"]),
            "filled_size": kwargs["size"],
            "paid_commission": "0.05",
        }


@pytest.mark.asyncio
async def test_live_exchange_uses_metadata_contract_value_first():
    market = _make_market(contract_value=0.001)
    fake = _FakeDeltaClient(contract_value=999.0)  # client value should be ignored
    px = DeltaLiveExchange(fake, max_notional_overshoot=10.0)  # type: ignore[arg-type]
    leg = Leg(
        market_id="BTCUSDT", outcome_id="BTCUSDT",
        side="BUY", limit_price=100000.0, size_usd=200.0, reason="test",
    )
    fill = await px.submit(leg, market)
    assert fill is not None
    # 200 / (0.001 * 100000) = 2 contracts
    assert fake.calls[0]["size"] == 2
    assert fake.calls[0]["symbol"] == "BTCUSDT"
    assert fake.calls[0]["side"] == "BUY"


@pytest.mark.asyncio
async def test_live_exchange_falls_back_to_client_contract_value():
    # Market built without metadata — should use the client's cache instead.
    market = Market(
        id="BTCUSDT", slug="btcusdt", question="BTC perp",
        category="perpetual_futures",
        outcomes={"BTCUSDT": Outcome(id="BTCUSDT", label=DELTA_OUTCOME_LABEL,
                                     book=OrderBook())},
        venue="delta",
    )
    fake = _FakeDeltaClient(contract_value=0.001)
    px = DeltaLiveExchange(fake, max_notional_overshoot=10.0)  # type: ignore[arg-type]
    leg = Leg(market_id="BTCUSDT", outcome_id="BTCUSDT",
              side="BUY", limit_price=100000.0, size_usd=200.0)
    fill = await px.submit(leg, market)
    assert fill is not None
    assert fake.calls[0]["size"] == 2


@pytest.mark.asyncio
async def test_live_exchange_rejects_overshoot_without_calling_api():
    market = _make_market(contract_value=0.001)
    fake = _FakeDeltaClient(contract_value=0.001)
    px = DeltaLiveExchange(fake, max_notional_overshoot=1.5)  # type: ignore[arg-type]
    # $5 against $100 minimum -> rejected before reaching place_order
    leg = Leg(market_id="BTCUSDT", outcome_id="BTCUSDT",
              side="BUY", limit_price=100000.0, size_usd=5.0)
    with pytest.raises(OrderRejected, match="overshoots"):
        await px.submit(leg, market)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_executor_surfaces_order_rejection_reason():
    """End-to-end: the dashboard's ExecutionResult.reason should contain the
    real exchange rejection text, not a generic 'all legs failed'."""
    from aera.core import Portfolio, RiskManager
    from aera.execution import Executor
    from aera.settings import RiskConfig
    from aera.strategies import Signal

    market = _make_market(contract_value=0.001)  # 1 contract ≈ $100
    pf = Portfolio(bankroll=100.0)
    # Disable executor target-sizing so this test focuses on the exchange's
    # rejection path; legacy mode keeps the strategy-emitted $5 leg unscaled
    # and lets it reach DeltaPaperExchange where the contract-overshoot
    # guard rejects it.
    risk = RiskManager(RiskConfig(trade_size_fraction=0.0), pf)
    px = DeltaPaperExchange(
        slippage=LinearSlippageModel(bps=0),
        max_notional_overshoot=1.5,
    )
    executor = Executor(pf, risk, px)
    signal = Signal(
        strategy="delta_perp_scalper",
        confidence=1.0,
        edge=0.01,
        legs=[Leg(market_id="BTCUSDT", outcome_id="BTCUSDT",
                  side="BUY", limit_price=100001.0, size_usd=5.0)],
    )
    result = await executor.execute(signal, {"BTCUSDT": market})
    assert not result.success
    assert "overshoots" in result.reason
    assert "all legs failed" not in result.reason


@pytest.mark.asyncio
async def test_live_exchange_refuses_when_not_authenticated():
    class _Unauth:
        authenticated = False
        def contract_value_for(self, *a, **k): return 1.0
        async def place_order(self, **k):  # pragma: no cover
            raise AssertionError("should not be called")

    market = _make_market()
    px = DeltaLiveExchange(_Unauth())  # type: ignore[arg-type]
    leg = Leg(market_id="BTCUSDT", outcome_id="BTCUSDT", side="BUY",
              limit_price=100000.0, size_usd=500.0)
    with pytest.raises(OrderRejected, match="DELTA_API_KEY"):
        await px.submit(leg, market)
