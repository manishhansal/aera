"""Bar-replay backtesting engine.

Drives any ``Strategy`` subclass through a stream of historical OHLCV
bars, simulating fills against a synthetic order book, and producing
per-trade records + aggregate metrics. Reuses the production
``Portfolio`` so PnL math is identical to live.

Why bars, not ticks?
====================

Historical orderbook snapshots are rare / paid; OHLCV is everywhere
and small. We synthesise a "tick stream" from each bar by emitting
four ticks per bar — open → high → low → close — through a
configurable-spread synthetic book. This is high-fidelity for
mean-reversion / trend / volatility-driven strategies (they only
need mid + spread); it's LOSSY for orderbook-microstructure
strategies (``order_book_sniper``, ``bid_ask_spread_fade``,
``flow_scalp``) which read level-2 depth and trade tape — those
strategies still backtest end-to-end but their entry triggers will
be approximate.

Direction of replay matters: high before low on red bars, low before
high on green bars, so the synthesised tick path is the most adverse
plausible reconstruction of the bar. This is the "open-high-low-close
heuristic" used by Backtrader / VectorBT.

Output
======

* :class:`BacktestResult` aggregates per-trade rows + headline stats
  (PnL, win rate, profit factor, max DD, Sharpe).
* :class:`TradeRecord` is one round-trip; the sweep / analysis
  modules group these by hour-of-day, regime, etc.

The replay engine is sync; callers that want async parallelism over
many (strategy × symbol × resolution) tuples wrap individual replays
in ``asyncio.to_thread`` (see ``aera.backtest.sweep``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd

from aera.core.portfolio import Portfolio
from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market, OrderBook, Outcome
from aera.strategies.base import Strategy

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """One closed round-trip from a backtest."""

    strategy: str
    symbol: str
    side: str               # "LONG" or "SHORT"
    entry_ts: int           # unix seconds
    exit_ts: int
    entry_price: float
    exit_price: float
    size_usd: float         # notional at entry
    leverage: float
    pnl_usd: float
    fees_usd: float
    hold_seconds: int
    reason_open: str
    reason_close: str

    @property
    def is_win(self) -> bool:
        return self.pnl_usd > 0

    @property
    def return_pct(self) -> float:
        return (self.pnl_usd / self.size_usd) if self.size_usd > 0 else 0.0

    @property
    def hour_of_day(self) -> int:
        return datetime.fromtimestamp(self.entry_ts, tz=timezone.utc).hour

    @property
    def day_of_week(self) -> int:
        # Monday = 0
        return datetime.fromtimestamp(self.entry_ts, tz=timezone.utc).weekday()


@dataclass
class BacktestResult:
    """Aggregate output of a single backtest run."""

    strategy: str
    symbol: str
    resolution: str
    leverage: float
    bars_processed: int
    trades: List[TradeRecord] = field(default_factory=list)

    starting_bankroll: float = 0.0
    ending_bankroll: float = 0.0

    # ------------------------------------------------------------------
    # cached metrics (computed once on first access)
    # ------------------------------------------------------------------

    def _pnls(self) -> List[float]:
        return [t.pnl_usd for t in self.trades]

    @property
    def total_pnl(self) -> float:
        return sum(self._pnls())

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def num_wins(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def num_losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl_usd < 0)

    @property
    def win_rate(self) -> float:
        n = self.num_trades
        return self.num_wins / n if n > 0 else 0.0

    @property
    def avg_win(self) -> float:
        wins = [p for p in self._pnls() if p > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [p for p in self._pnls() if p < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(p for p in self._pnls() if p > 0)
        gross_loss = abs(sum(p for p in self._pnls() if p < 0))
        if gross_loss <= 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def expectancy(self) -> float:
        n = self.num_trades
        return self.total_pnl / n if n > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        """Max peak-to-trough equity drawdown as a fraction of peak."""
        if not self.trades:
            return 0.0
        equity = self.starting_bankroll
        peak = equity
        max_dd = 0.0
        for t in self.trades:
            equity += t.pnl_usd
            peak = max(peak, equity)
            if peak > 0:
                dd = 1.0 - (equity / peak)
                max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe(self) -> float:
        """Per-trade Sharpe ratio (mean PnL / stdev PnL, annualised
        only if the caller knows the cadence — we keep it raw to stay
        regime-agnostic). Returns 0.0 when there's not enough data."""
        pnls = self._pnls()
        if len(pnls) < 2:
            return 0.0
        mu = sum(pnls) / len(pnls)
        var = sum((p - mu) ** 2 for p in pnls) / (len(pnls) - 1)
        sd = math.sqrt(var)
        return (mu / sd) if sd > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "resolution": self.resolution,
            "leverage": self.leverage,
            "bars_processed": self.bars_processed,
            "starting_bankroll": self.starting_bankroll,
            "ending_bankroll": self.ending_bankroll,
            "total_pnl": self.total_pnl,
            "num_trades": self.num_trades,
            "num_wins": self.num_wins,
            "num_losses": self.num_losses,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": (
                None if self.profit_factor == float("inf") else self.profit_factor
            ),
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
        }

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "strategy": t.strategy,
            "symbol": t.symbol,
            "side": t.side,
            "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size_usd": t.size_usd,
            "leverage": t.leverage,
            "pnl_usd": t.pnl_usd,
            "fees_usd": t.fees_usd,
            "hold_seconds": t.hold_seconds,
            "hour_of_day": t.hour_of_day,
            "day_of_week": t.day_of_week,
            "is_win": t.is_win,
            "reason_open": t.reason_open,
            "reason_close": t.reason_close,
        } for t in self.trades])


# ---------------------------------------------------------------------------
# bar → tick → market conversion
# ---------------------------------------------------------------------------


def _synthetic_book(
    price: float,
    spread_bps: float,
    depth_qty: float,
    *,
    bid_qty: Optional[float] = None,
    ask_qty: Optional[float] = None,
) -> OrderBook:
    """Build a 2-level synthetic order book around a single mid price.

    By default the book is symmetric (``bid_qty == ask_qty == depth_qty``)
    but the caller can override either side independently to encode an
    OFI / depth-imbalance signal. The L2 layer is sized at half the
    top-of-book of the SAME side so cumulative-depth based features
    (``measure_depth_imbalance``) see the same imbalance as raw OFI.
    """
    bq = depth_qty if bid_qty is None else max(1e-9, bid_qty)
    aq = depth_qty if ask_qty is None else max(1e-9, ask_qty)
    half = price * spread_bps / 1e4 / 2.0
    bid_price = max(price - half, 1e-9)
    ask_price = max(price + half, bid_price + 1e-9)
    book = OrderBook()
    book.replace(
        bids=[(bid_price, bq), (bid_price * 0.999, bq * 0.5)],
        asks=[(ask_price, aq), (ask_price * 1.001, aq * 0.5)],
    )
    return book


# How aggressively the synthetic book reflects intra-tick price moves
# in its top-of-book size split. The "winning" side gets multiplied by
# ``_BOOK_IMBAL_HIGH`` while the "losing" side is scaled by ``_BOOK_IMBAL_LOW``.
# Picking 2.0 / 0.5 gives a raw OFI of (2 − 0.5) / (2 + 0.5) = 0.6 on
# an up-tick, which is comfortably above the live strategies'
# ``ofi_threshold`` (0.20–0.35) so the OFI gate actually engages while
# still leaving headroom for noisier-than-tick-perfect setups.
_BOOK_IMBAL_HIGH = 2.0
_BOOK_IMBAL_LOW = 0.5


def candles_to_market_stream(
    candles: Iterable,
    symbol: str,
    *,
    spread_bps: float = 5.0,
    depth_qty: float = 1000.0,
    ticks_per_bar: int = 4,
    leverage: float = 25.0,
) -> Iterator[tuple[int, Market]]:
    """Yield ``(ts, market)`` for each synthetic tick within each bar.

    Each bar fans into ``ticks_per_bar`` mid-price prints. The default
    sequence is open → first-extreme → second-extreme → close where
    the extreme order follows the bar's direction: red bars (close <
    open) put the HIGH first (most adverse to a long bias), green
    bars put the LOW first. This is the standard "open-high-low-close
    walk" heuristic; it doesn't recover micro-structure but is the
    best honest reconstruction from OHLCV alone.

    Each tick gets its own ``ts`` evenly spaced between the bar's
    open and the next bar's open, so timestamps strictly increase.

    Synthetic order-flow encoding
    -----------------------------

    OHLCV does not contain a level-2 book or trade tape, so a naive
    synthetic book with equal sizes on both sides produces zero
    Order-Flow Imbalance (OFI) and zero depth-imbalance — which
    silently disables every microstructure strategy (``delta_perp_scalper``,
    ``order_book_sniper``, ``tick_reversal_scalp``, ...). To keep those
    strategies functional in replay we shape the synthetic book around
    the most recent intra-tick price move:

    * tick price > previous tick price → asks were eaten, bid side
      stronger → top-of-book sizes are ``(bid = 2×base, ask = 0.5×base)``
      → raw OFI = +0.6
    * tick price < previous → mirror, raw OFI = −0.6
    * unchanged → symmetric ``(bid = ask = base)`` → OFI = 0

    This is a directionally-honest reconstruction: the imbalance only
    appears when there is real price evidence for it, never as a free
    "always-buy-side-stronger" gift. Magnitudes are picked to exceed
    the live ``ofi_threshold`` config defaults so the OFI gate engages
    when z-score does, exactly as it would on a real venue.
    """
    if hasattr(candles, "iterrows"):
        rows = list(candles.itertuples(index=False))
    else:
        rows = list(candles)

    prev_price: Optional[float] = None
    for i, row in enumerate(rows):
        ts0 = int(row.ts)
        next_ts = int(rows[i + 1].ts) if i + 1 < len(rows) else ts0 + 60
        bar_dt = max(1, next_ts - ts0)
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        vol = float(getattr(row, "volume", 0.0))

        if c < o:
            seq = [o, h, l, c]
        else:
            seq = [o, l, h, c]
        # Volume-derived depth, with a floor so dead bars still get
        # filled (otherwise min_depth_contracts gates everything out).
        base_qty = max(1.0, vol / max(1, ticks_per_bar))

        for k in range(ticks_per_bar):
            tk = ts0 + int(round(bar_dt * k / max(1, ticks_per_bar - 1)))
            price = seq[k] if k < len(seq) else c

            if prev_price is None or price == prev_price:
                bid_qty = base_qty
                ask_qty = base_qty
            elif price > prev_price:
                bid_qty = base_qty * _BOOK_IMBAL_HIGH
                ask_qty = base_qty * _BOOK_IMBAL_LOW
            else:
                bid_qty = base_qty * _BOOK_IMBAL_LOW
                ask_qty = base_qty * _BOOK_IMBAL_HIGH

            book = _synthetic_book(
                price, spread_bps, base_qty,
                bid_qty=bid_qty, ask_qty=ask_qty,
            )
            outcome = Outcome(id=symbol, label=DELTA_OUTCOME_LABEL, book=book)
            market = Market(
                id=symbol,
                slug=symbol.lower(),
                question=f"{symbol} backtest",
                category="perpetual_futures",
                outcomes={symbol: outcome},
                venue="delta",
                last_update=float(tk),
                metadata={"leverage": leverage, "contract_value": 1.0},
            )
            yield tk, market

            prev_price = price


# ---------------------------------------------------------------------------
# replay engine
# ---------------------------------------------------------------------------


@dataclass
class _OpenPosition:
    """In-flight position tracked by the replay engine."""

    side: str            # "LONG" or "SHORT"
    entry_ts: int
    entry_price: float
    size_usd: float
    leverage: float
    fees_paid: float
    reason_open: str


class BarReplay:
    """Drive a strategy through a stream of candles and record trades.

    The engine fills entry signals at the **ask** (BUY) / **bid**
    (SELL) at the current tick, plus configurable slippage. Exit
    signals (reduce-only) fill at the OPPOSITE side. Round-trip fees
    are accrued from ``taker_fee_bps`` on each leg.

    The strategy's own state machine is fully respected — entry
    triggers, internal TP / SL, debouncing etc. all run unchanged.
    The replay engine only translates "open this position" /
    "close this position" signals into bankroll changes.

    Multi-leg signals: collapsed to the largest leg by ``size_usd``.
    Backtesting multi-leg arb on bars is structurally lossy; the
    strategies that actually use multi-leg are pricing-engine bets
    which run on live mid prices only.
    """

    def __init__(
        self,
        strategy_factory,
        *,
        starting_bankroll: float = 1000.0,
        symbol: str,
        resolution: str = "1m",
        leverage: float = 25.0,
        spread_bps: float = 5.0,
        slippage_bps: float = 2.0,
        taker_fee_bps: float = 5.0,
        ticks_per_bar: int = 4,
    ) -> None:
        self.strategy_factory = strategy_factory
        self.starting_bankroll = float(starting_bankroll)
        self.symbol = symbol
        self.resolution = resolution
        self.leverage = float(leverage)
        self.spread_bps = float(spread_bps)
        self.slippage_bps = float(slippage_bps)
        self.taker_fee_bps = float(taker_fee_bps)
        self.ticks_per_bar = int(ticks_per_bar)

    def run(self, candles) -> BacktestResult:
        """Replay ``candles`` and return the aggregate result."""
        if hasattr(candles, "empty") and candles.empty:
            return BacktestResult(
                strategy=self.strategy_factory.__name__ if hasattr(self.strategy_factory, "__name__") else "?",
                symbol=self.symbol,
                resolution=self.resolution,
                leverage=self.leverage,
                bars_processed=0,
                starting_bankroll=self.starting_bankroll,
                ending_bankroll=self.starting_bankroll,
            )

        portfolio = Portfolio(bankroll=self.starting_bankroll)
        strategy: Strategy = self.strategy_factory(portfolio)
        strategy_name = getattr(strategy, "name", strategy.__class__.__name__)

        result = BacktestResult(
            strategy=strategy_name,
            symbol=self.symbol,
            resolution=self.resolution,
            leverage=self.leverage,
            bars_processed=int(len(candles)),
            starting_bankroll=self.starting_bankroll,
        )

        position: Optional[_OpenPosition] = None

        stream = candles_to_market_stream(
            candles,
            symbol=self.symbol,
            spread_bps=self.spread_bps,
            ticks_per_bar=self.ticks_per_bar,
            leverage=self.leverage,
        )

        for ts, market in stream:
            outcome = next(iter(market.outcomes.values()))
            bid = outcome.best_bid
            ask = outcome.best_ask
            if bid is None or ask is None:
                continue

            try:
                signals = list(strategy.scan([market]))
            except Exception as exc:
                log.debug("strategy.scan crashed during replay: %s", exc)
                signals = []

            for sig in signals:
                if not sig.legs:
                    continue
                # Largest leg by size_usd is the representative leg.
                leg = max(sig.legs, key=lambda l: l.size_usd)
                reduce_only = bool(getattr(leg, "reduce_only", False))
                side = leg.side
                if reduce_only:
                    if position is None:
                        continue
                    self._close_position(
                        position, side, ts, bid, ask, sig.legs[0].reason,
                        strategy_name, result,
                    )
                    position = None
                else:
                    if position is not None:
                        # Flip or stack — flatten then re-open.
                        flip_side = "SELL" if position.side == "LONG" else "BUY"
                        self._close_position(
                            position, flip_side, ts, bid, ask,
                            "replay: flipped by new entry",
                            strategy_name, result,
                        )
                        position = None
                    position = self._open_position(
                        side, ts, bid, ask, leg.size_usd, leg.reason, strategy_name,
                    )

        # Mark-to-market any open position at the final close so the
        # bankroll number reflects total realised + unrealised PnL.
        if position is not None and not (hasattr(candles, "empty") and candles.empty):
            last = candles.iloc[-1] if hasattr(candles, "iloc") else candles[-1]
            last_ts = int(last.ts if hasattr(last, "ts") else last[0])
            last_price = float(last.close if hasattr(last, "close") else last[4])
            close_side = "SELL" if position.side == "LONG" else "BUY"
            # Use last close as the fill price for the synthetic flatten.
            half = last_price * self.spread_bps / 1e4 / 2.0
            bid_p = last_price - half
            ask_p = last_price + half
            self._close_position(
                position, close_side, last_ts, bid_p, ask_p,
                "replay: forced flatten at end of data",
                strategy_name, result,
            )

        result.ending_bankroll = self.starting_bankroll + result.total_pnl
        return result

    # ------------------------------------------------------------------
    # fill simulation
    # ------------------------------------------------------------------

    def _fill_price(self, side: str, bid: float, ask: float) -> float:
        """Aggressive taker fill: BUY at ask + slippage, SELL at bid - slippage."""
        slip = (ask if side == "BUY" else bid) * self.slippage_bps / 1e4
        return (ask + slip) if side == "BUY" else max(0.0, bid - slip)

    def _open_position(
        self, side: str, ts: int, bid: float, ask: float,
        size_usd: float, reason: str, strategy_name: str,
    ) -> _OpenPosition:
        fill_px = self._fill_price(side, bid, ask)
        fee = size_usd * self.taker_fee_bps / 1e4
        return _OpenPosition(
            side="LONG" if side == "BUY" else "SHORT",
            entry_ts=ts,
            entry_price=fill_px,
            size_usd=size_usd,
            leverage=self.leverage,
            fees_paid=fee,
            reason_open=reason,
        )

    def _close_position(
        self, pos: _OpenPosition, close_side: str, ts: int,
        bid: float, ask: float, reason: str, strategy_name: str,
        result: BacktestResult,
    ) -> None:
        fill_px = self._fill_price(close_side, bid, ask)
        notional = pos.size_usd
        # Signed return in dollars: long earns (close - entry) / entry,
        # short earns (entry - close) / entry. Leverage is already
        # baked into ``size_usd`` (it's the NOTIONAL the strategy
        # asked for); the gross dollar PnL is independent of leverage.
        if pos.side == "LONG":
            pnl_gross = notional * (fill_px - pos.entry_price) / pos.entry_price
        else:
            pnl_gross = notional * (pos.entry_price - fill_px) / pos.entry_price
        close_fee = notional * self.taker_fee_bps / 1e4
        total_fees = pos.fees_paid + close_fee
        pnl_net = pnl_gross - total_fees
        result.trades.append(TradeRecord(
            strategy=strategy_name,
            symbol=self.symbol,
            side=pos.side,
            entry_ts=pos.entry_ts,
            exit_ts=ts,
            entry_price=pos.entry_price,
            exit_price=fill_px,
            size_usd=notional,
            leverage=self.leverage,
            pnl_usd=pnl_net,
            fees_usd=total_fees,
            hold_seconds=max(0, ts - pos.entry_ts),
            reason_open=pos.reason_open,
            reason_close=reason,
        ))
