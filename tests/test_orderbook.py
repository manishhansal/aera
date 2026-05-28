"""Order book and VWAP tests."""
from __future__ import annotations

import math

from aera.markets.orderbook import OrderBook


def test_best_bid_ask():
    book = OrderBook()
    book.replace(bids=[(0.40, 100), (0.39, 50)], asks=[(0.42, 75), (0.43, 50)])
    assert book.best_bid_price() == 0.40
    assert book.best_ask_price() == 0.42


def test_vwap_buy_walks_book():
    book = OrderBook()
    book.replace(bids=[], asks=[(0.40, 10), (0.50, 10)])   # $4 + $5 of inventory
    # need $6 -> all of level 1 (4$ for 10 shares) + $2 of level 2 (4 shares @ 0.50)
    vwap = book.vwap_buy(6.0)
    # 10 shares @ 0.40 + 4 shares @ 0.50 = 14 shares for $6 -> avg = 6/14
    assert math.isclose(vwap, 6.0 / 14.0, abs_tol=1e-9)


def test_vwap_buy_insufficient_depth():
    book = OrderBook()
    book.replace(bids=[], asks=[(0.40, 1)])
    assert book.vwap_buy(100.0) is None


def test_depth_at_or_better():
    book = OrderBook()
    book.replace(bids=[(0.40, 100), (0.39, 50)], asks=[])
    assert book.depth_at_or_better("buy", 0.39) == 150
    assert book.depth_at_or_better("buy", 0.40) == 100
