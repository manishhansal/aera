"""Slippage models — converts intended price into expected fill price."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from aera.markets import OrderBook


class SlippageModel(abc.ABC):
    @abc.abstractmethod
    def fill_price(
        self,
        side: str,
        notional_usd: float,
        book: Optional[OrderBook],
        reference_price: float,
    ) -> float:
        """Return the expected average fill price for a market order."""


@dataclass
class LinearSlippageModel(SlippageModel):
    """Adds `bps` of fixed slippage to the reference price."""
    bps: float = 10.0

    def fill_price(
        self,
        side: str,
        notional_usd: float,
        book: Optional[OrderBook],
        reference_price: float,
    ) -> float:
        # If we have an actual book, walk it for VWAP.
        if book is not None:
            vwap = (
                book.vwap_buy(notional_usd)
                if side.upper() == "BUY"
                else book.vwap_sell(notional_usd)
            )
            if vwap is not None:
                return vwap
        # else fall back to flat-bps model
        bump = reference_price * (self.bps / 10_000.0)
        return reference_price + bump if side.upper() == "BUY" else reference_price - bump
