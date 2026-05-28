"""Delta Exchange execution venue.

Implements the `Exchange` ABC so the existing `Executor` can route trade
legs to Delta with no code changes elsewhere. Two flavours:

  * `DeltaLiveExchange` — submits real signed orders via `DeltaClient`.
  * `DeltaPaperExchange` — simulates fills against the live Delta book
    using the same slippage model used by Delta paper trades. Useful
    for paper-trading Delta without touching capital.

Contract sizing
---------------

Delta sizes orders in integer **contracts**, and 1 contract represents
``contract_value`` units of the underlying (BTCUSDT: 0.001 BTC, ETHUSDT:
0.01 ETH, etc.). So::

    notional_usd = contracts * contract_value * mark_price

The strategy expresses size in USD notional; we invert that formula to
work out the contract count, round it, and *reject* the trade if the
minimum 1-contract size would overshoot the strategy's intent by more
than ``max_notional_overshoot`` × the intended notional. That stops the
bot from silently turning a $5 leg into a $76 leg on BTCUSDT just because
1 contract is ~$76.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from aera.core import Fill
from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, DeltaClient, DeltaError, Market
from aera.strategies import Leg

from .executor import Exchange, OrderRejected
from .slippage import LinearSlippageModel, SlippageModel


log = get_logger(__name__)


@dataclass
class ContractSizing:
    contracts: int
    actual_notional_usd: float
    rejected: bool = False
    reason: str = ""


def size_in_contracts(
    *,
    notional_usd: float,
    price: float,
    contract_value: float,
    min_trade_notional_usd: float = 1.0,
    max_notional_overshoot: float = 1.5,
) -> ContractSizing:
    """USD notional → integer contract count, with explicit reject reasons.

    Args
    ----
    notional_usd:
        What the strategy wants to trade, in USD.
    price:
        Mark price (or limit price) of the contract.
    contract_value:
        Units of underlying per 1 contract (from the Delta product spec).
        For BTCUSDT this is 0.001 BTC; for "1 contract = 1 underlying" type
        venues set this to 1.0.
    min_trade_notional_usd:
        Hard floor — reject trades below this notional. Skips the rounding
        step entirely.
    max_notional_overshoot:
        Reject when ``actual_notional / intended_notional`` exceeds this
        ratio. Protects small bankrolls from being silently up-sized to the
        1-contract minimum.
    """
    if price <= 0 or contract_value <= 0:
        return ContractSizing(0, 0.0, True, "non-positive price or contract_value")
    if notional_usd < min_trade_notional_usd:
        return ContractSizing(
            0, 0.0, True,
            f"notional ${notional_usd:.4f} below min ${min_trade_notional_usd:.4f}",
        )
    one_contract_usd = contract_value * price
    raw = notional_usd / one_contract_usd
    contracts = int(round(raw))
    if contracts <= 0:
        # The intended trade is smaller than even one contract.
        # Decide: would clamping to 1 contract overshoot acceptably?
        overshoot_ratio = one_contract_usd / max(notional_usd, 1e-9)
        if overshoot_ratio > max_notional_overshoot:
            return ContractSizing(
                0, 0.0, True,
                (f"1 contract = ${one_contract_usd:.4f} overshoots "
                 f"intended ${notional_usd:.4f} by {overshoot_ratio:.2f}x "
                 f"(> {max_notional_overshoot:g}x cap)"),
            )
        contracts = 1
    return ContractSizing(
        contracts=contracts,
        actual_notional_usd=contracts * one_contract_usd,
        rejected=False,
    )


def _contract_value_from_market(market: Market, default: float = 1.0) -> float:
    """Read ``contract_value`` from market metadata, falling back to ``default``."""
    md = market.metadata or {}
    cv = md.get("contract_value", default)
    try:
        return float(cv) if cv is not None else default
    except (TypeError, ValueError):
        return default


class DeltaPaperExchange(Exchange):
    """Paper-trade Delta using the live order book and a slippage model.

    ``taker_fee_bps`` is charged against the filled notional of each fill so
    paper P&L mirrors live trading on Delta (whose taker fee is non-zero).
    The fee flows into ``Fill.fee`` and the Portfolio subtracts it from
    bankroll just like a real commission.
    """

    def __init__(
        self,
        slippage: Optional[SlippageModel] = None,
        *,
        min_trade_notional_usd: float = 1.0,
        max_notional_overshoot: float = 1.5,
        taker_fee_bps: float = 0.0,
    ) -> None:
        self.slippage = slippage or LinearSlippageModel()
        self.min_trade_notional_usd = min_trade_notional_usd
        self.max_notional_overshoot = max_notional_overshoot
        self.taker_fee_bps = max(0.0, float(taker_fee_bps))

    async def submit(self, leg: Leg, market: Market) -> Optional[Fill]:
        outcome = market.outcomes.get(leg.outcome_id)
        if outcome is None:
            raise OrderRejected(f"unknown outcome {leg.outcome_id} for market {market.id}")
        # Delta has one outcome per market — we just sanity-check the label
        # so a stray multi-outcome leg can't accidentally be routed here.
        if outcome.label != DELTA_OUTCOME_LABEL:
            log.warning(
                "delta paper submit: outcome label %r is not %s",
                outcome.label, DELTA_OUTCOME_LABEL,
            )

        # Compute correct contract size against the product's contract_value;
        # paper-trade what the bot would actually be able to place on Delta.
        sizing = size_in_contracts(
            notional_usd=leg.size_usd,
            price=leg.limit_price,
            contract_value=_contract_value_from_market(market),
            min_trade_notional_usd=self.min_trade_notional_usd,
            max_notional_overshoot=self.max_notional_overshoot,
        )
        if sizing.rejected:
            raise OrderRejected(f"{market.id}: {sizing.reason}")
        actual_notional = sizing.actual_notional_usd

        ref = leg.limit_price
        fill_price = self.slippage.fill_price(
            side=leg.side,
            notional_usd=actual_notional,
            book=outcome.book,
            reference_price=ref,
        )
        if leg.side == "BUY" and fill_price > leg.limit_price * 1.001:
            log.debug(
                "delta paper: limit not met (buy %.4f vs %.4f)",
                fill_price, leg.limit_price,
            )
            return None
        if leg.side == "SELL" and fill_price < leg.limit_price * 0.999:
            log.debug(
                "delta paper: limit not met (sell %.4f vs %.4f)",
                fill_price, leg.limit_price,
            )
            return None

        # "shares" in the bot's portfolio is dollars-of-notional, so paper P&L
        # matches what live would produce.
        shares = actual_notional / max(fill_price, 1e-6)
        fee = actual_notional * self.taker_fee_bps * 1e-4
        return Fill(
            timestamp=time.time(),
            market_id=leg.market_id,
            outcome_id=leg.outcome_id,
            side=leg.side,
            price=fill_price,
            size=shares,
            fee=fee,
            # Leverage flows through to the Portfolio so it posts margin
            # (notional / leverage) instead of treating the open as a full
            # cash spend — without this the bankroll goes deeply negative
            # on the first leveraged paper fill and the drawdown halt fires.
            leverage=float(getattr(leg, "leverage", 1.0) or 1.0),
        )


class DeltaLiveExchange(Exchange):
    """Submits real orders to Delta Exchange via `DeltaClient`."""

    def __init__(
        self,
        client: DeltaClient,
        *,
        order_type: str = "limit_order",
        time_in_force: str = "ioc",
        post_only: bool = False,
        reduce_only: bool = False,
        min_trade_notional_usd: float = 1.0,
        max_notional_overshoot: float = 1.5,
    ) -> None:
        self.client = client
        self.order_type = order_type
        self.time_in_force = time_in_force
        self.post_only = post_only
        self.reduce_only = reduce_only
        self.min_trade_notional_usd = min_trade_notional_usd
        self.max_notional_overshoot = max_notional_overshoot

    async def submit(self, leg: Leg, market: Market) -> Optional[Fill]:
        if not self.client.authenticated:
            raise OrderRejected("DELTA_API_KEY/DELTA_API_SECRET not configured")

        # leg.market_id is the Delta symbol (we set it that way in _parse_product).
        symbol = leg.market_id
        # Prefer the contract_value baked into the market metadata; fall back
        # to the client's cache so we don't fail if a Market was hand-built
        # for tests.
        contract_value = _contract_value_from_market(market)
        if contract_value == 1.0:
            contract_value = self.client.contract_value_for(symbol, default=1.0)

        sizing = size_in_contracts(
            notional_usd=leg.size_usd,
            price=leg.limit_price,
            contract_value=contract_value,
            min_trade_notional_usd=self.min_trade_notional_usd,
            max_notional_overshoot=self.max_notional_overshoot,
        )
        if sizing.rejected:
            raise OrderRejected(f"{symbol}: {sizing.reason}")

        # Per-leg flag wins: a strategy can mark TP/SL/unwind legs as
        # reduce-only so Delta refuses to open a new opposite leg if our
        # size math is ever off; the exchange-level default still applies
        # to everything else (entries).
        leg_reduce_only = bool(getattr(leg, "reduce_only", False))
        reduce_only = self.reduce_only or leg_reduce_only
        # Per-leg execution-mode overrides: maker-only strategies stamp
        # post_only=True / time_in_force="gtc" on their emitted legs to
        # post resting maker quotes, while the exchange itself stays
        # configured for IOC taker fills used by every other strategy.
        # ``None`` = inherit the exchange-level default (the common case).
        leg_tif = getattr(leg, "time_in_force", None)
        time_in_force = leg_tif if leg_tif is not None else self.time_in_force
        leg_post_only = getattr(leg, "post_only", None)
        post_only = (
            bool(leg_post_only) if leg_post_only is not None else self.post_only
        )
        try:
            resp = await self.client.place_order(
                symbol=symbol,
                side=leg.side,
                size=sizing.contracts,
                limit_price=leg.limit_price,
                order_type=self.order_type,
                time_in_force=time_in_force,
                post_only=post_only,
                reduce_only=reduce_only,
            )
        except DeltaError as exc:
            log.error("delta live order failed: %s", exc)
            return None
        except Exception as exc:  # pragma: no cover - network errors
            log.exception("delta live order crashed: %s", exc)
            return None

        # Delta's response format: average_fill_price and filled_size on fills.
        try:
            avg_price = float(
                resp.get("average_fill_price")
                or resp.get("avg_fill_price")
                or leg.limit_price
            )
            filled_contracts = float(
                resp.get("filled_size")
                or resp.get("size")
                or sizing.contracts
            )
        except (TypeError, ValueError):
            avg_price = leg.limit_price
            filled_contracts = float(sizing.contracts)

        # Convert (contracts × contract_value × price) → dollar notional, then
        # divide by avg fill price → "shares" (the bot treats 1 share = $1 of
        # notional, matching the bankroll math for all venues).
        actual_notional = filled_contracts * contract_value * avg_price
        shares = (
            actual_notional / avg_price
            if avg_price > 0
            else float(sizing.contracts)
        )
        fee_raw = resp.get("paid_commission") or resp.get("commission") or 0.0
        try:
            fee = float(fee_raw)
        except (TypeError, ValueError):
            fee = 0.0

        return Fill(
            timestamp=time.time(),
            market_id=leg.market_id,
            outcome_id=leg.outcome_id,
            side=leg.side,
            price=avg_price,
            size=shares,
            fee=fee,
            leverage=float(getattr(leg, "leverage", 1.0) or 1.0),
        )
