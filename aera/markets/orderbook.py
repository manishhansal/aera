"""Sorted order book with depth tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Level:
    price: float
    size: float


@dataclass
class OrderBook:
    """Two sorted price ladders (bids descending, asks ascending)."""
    bids: Dict[float, float] = field(default_factory=dict)   # price -> size
    asks: Dict[float, float] = field(default_factory=dict)

    def replace(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> None:
        self.bids = {float(p): float(s) for p, s in bids if float(s) > 0}
        self.asks = {float(p): float(s) for p, s in asks if float(s) > 0}

    def update_level(self, side: str, price: float, size: float) -> None:
        book = self.bids if side.lower() in ("buy", "bid") else self.asks
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size

    def best_bid_price(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def best_ask_price(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def best_bid(self) -> Optional[Level]:
        p = self.best_bid_price()
        return Level(p, self.bids[p]) if p is not None else None

    def best_ask(self) -> Optional[Level]:
        p = self.best_ask_price()
        return Level(p, self.asks[p]) if p is not None else None

    def bids_sorted(self) -> List[Level]:
        return [Level(p, self.bids[p]) for p in sorted(self.bids.keys(), reverse=True)]

    def asks_sorted(self) -> List[Level]:
        return [Level(p, self.asks[p]) for p in sorted(self.asks.keys())]

    def vwap_buy(self, notional: float) -> Optional[float]:
        """Volume-weighted price to BUY `notional` USD-worth of shares from asks."""
        remaining = notional
        cost = 0.0
        shares = 0.0
        for lvl in self.asks_sorted():
            level_notional = lvl.price * lvl.size
            if level_notional >= remaining:
                fill_shares = remaining / lvl.price
                cost += fill_shares * lvl.price
                shares += fill_shares
                remaining = 0.0
                break
            cost += level_notional
            shares += lvl.size
            remaining -= level_notional
        if remaining > 0 or shares == 0:
            return None
        return cost / shares

    def vwap_sell(self, notional: float) -> Optional[float]:
        """VWAP to SELL `notional` USD-worth of shares into bids."""
        remaining = notional
        proceeds = 0.0
        shares = 0.0
        for lvl in self.bids_sorted():
            level_notional = lvl.price * lvl.size
            if level_notional >= remaining:
                fill_shares = remaining / lvl.price
                proceeds += fill_shares * lvl.price
                shares += fill_shares
                remaining = 0.0
                break
            proceeds += level_notional
            shares += lvl.size
            remaining -= level_notional
        if remaining > 0 or shares == 0:
            return None
        return proceeds / shares

    def depth_at_or_better(self, side: str, price: float) -> float:
        """Total size on `side` at prices at least as good as `price`."""
        side = side.lower()
        if side in ("buy", "bid"):
            return sum(s for p, s in self.bids.items() if p >= price)
        return sum(s for p, s in self.asks.items() if p <= price)
