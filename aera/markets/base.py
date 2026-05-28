"""Abstract market data model — every venue maps onto this."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .orderbook import OrderBook


@dataclass
class Outcome:
    """One side of a market. For Delta perps there is exactly one Outcome
    per Market (the perp itself); the abstraction is kept to make multi-leg
    routing in the executor uniform."""
    id: str
    label: str
    book: OrderBook = field(default_factory=OrderBook)
    last_price: Optional[float] = None
    volume_24h: float = 0.0

    @property
    def best_bid(self) -> Optional[float]:
        return self.book.best_bid_price()

    @property
    def best_ask(self) -> Optional[float]:
        return self.book.best_ask_price()

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb


@dataclass
class Market:
    """A single tradeable market (e.g. one Delta perpetual contract)."""
    id: str
    slug: str
    question: str
    category: str
    outcomes: Dict[str, Outcome] = field(default_factory=dict)
    end_time: Optional[float] = None          # unix seconds
    resolution_source: Optional[str] = None
    venue: str = "delta"
    tags: List[str] = field(default_factory=list)
    minimum_tick: float = 0.001
    last_update: float = 0.0
    # Venue-specific extras (Delta uses this for contract_value, product_id,
    # initial_margin %, maintenance_margin %). Strategies should read-only.
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def is_binary(self) -> bool:
        return len(self.outcomes) == 2

    @property
    def is_open(self) -> bool:
        if self.end_time is None:
            return True
        return time.time() < self.end_time

    def outcome_list(self) -> List[Outcome]:
        return list(self.outcomes.values())


@dataclass
class MarketSnapshot:
    """Immutable point-in-time view of a market."""
    timestamp: float
    market_id: str
    outcomes: Dict[str, dict]   # outcome_id -> {"bid": .., "ask": .., "bid_size": ..}
