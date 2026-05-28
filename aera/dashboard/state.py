"""Shared in-memory state container that the dashboard reads from.

The bot's `DeltaEngine` pushes lightweight events here via its callback hooks,
and the FastAPI server reads snapshots out of it. Keeping state in a single
container (instead of letting the dashboard reach into the engine internals)
makes the engine fully decoupled from the dashboard and easy to test.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING

from aera.core import Fill, Portfolio
from aera.execution.executor import ExecutionResult
from aera.strategies import Signal

if TYPE_CHECKING:
    from aera.core import AdaptiveBrain, DeltaEngine
    from aera.markets import Market


# ---------------------------------------------------------------------------
# Event records (small, JSON-serialisable)
# ---------------------------------------------------------------------------


@dataclass
class FillEvent:
    """One executed fill, as it appears on the live trades feed."""

    timestamp: float
    strategy: str
    market_id: str
    outcome_id: str
    side: str
    price: float
    size: float
    notional: float
    fee: float
    edge: float
    bankroll_after: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SignalEvent:
    """One strategy-emitted trade idea (executed or rejected)."""

    timestamp: float
    strategy: str
    edge: float
    confidence: float
    legs: int
    notional: float
    status: str          # "executed" | "rejected" | "pending"
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EquityPoint:
    """A single sample of the bankroll / equity curve.

    ``bankroll`` is the legacy free-cash bucket; on leveraged venues it dips
    every time we post margin and rebounds on close. ``settled_wealth`` adds
    locked margin back in, so it only moves on realised P&L (and fees) —
    that's what users want for an "am I making or losing money?" chart.
    """

    timestamp: float
    bankroll: float
    locked_margin: float
    settled_wealth: float
    equity: float
    drawdown: float
    growth_multiple: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategyStats:
    """Per-strategy aggregates."""

    name: str
    signals_emitted: int = 0
    trades_executed: int = 0
    trades_rejected: int = 0
    total_edge: float = 0.0
    realised_pnl: float = 0.0
    last_signal_ts: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["avg_edge"] = (
            self.total_edge / self.signals_emitted if self.signals_emitted else 0.0
        )
        return d


@dataclass
class TradeEvent:
    """A completed round-trip trade (one open → one close).

    Pairs the entry fill(s) with the closing fill(s) that flattened (or
    reduced) the position, so the dashboard can surface what users actually
    care about: *for this individual trade, what was the entry price, the
    exit price, and the realised P&L?*

    ``pnl`` is the realised dollar P&L after fees on the closed portion:
        pnl = (close_price − open_price) × size × direction − fees
    where direction is +1 for a LONG and −1 for a SHORT.
    """

    opened_at: float
    closed_at: float
    duration_seconds: float
    strategy: str
    market_id: str
    outcome_id: str
    side: str               # "LONG" or "SHORT" (direction of the original open)
    open_price: float
    close_price: float
    size: float             # |shares| closed in this trade
    pnl: float              # USD profit/loss after fees on the closed portion
    fees: float             # total fees attributed to this round-trip
    leverage: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _OpenTrade:
    """In-flight (still open) trade book-kept while waiting for the close.

    Only used internally by ``DashboardState`` — never serialised. The
    ``open_price`` is a running weighted average over all scale-in fills,
    so a position built up in two halves still reports a sensible single
    entry price when it eventually closes.
    """

    opened_at: float
    side: str               # "LONG" or "SHORT"
    market_id: str
    outcome_id: str
    open_price: float       # weighted-avg price across opening fills
    open_size: float        # cumulative |shares| still open
    open_strategy: str
    leverage: float
    fees_paid: float        # cumulative fees on opening fills, not yet attributed


# ---------------------------------------------------------------------------
# DashboardState
# ---------------------------------------------------------------------------


class DashboardState:
    """Singleton-style container the FastAPI app reads from.

    The engine, when wired up, pushes events into ``record_*`` methods. The
    server reads consistent snapshots via ``snapshot()``, ``recent_fills()``,
    etc. Everything here is in-memory and bounded by deques, so the process
    footprint stays flat under long runs.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        *,
        max_fills: int = 500,
        max_signals: int = 500,
        max_trades: int = 500,
        max_equity_points: int = 5000,
    ) -> None:
        self.portfolio = portfolio
        self.engine: Optional["DeltaEngine"] = None
        self.brain: Optional["AdaptiveBrain"] = None
        self.started_at = time.time()
        self.last_event_at: float = 0.0
        self.paper_mode: bool = True
        self.strategies_enabled: List[str] = []

        self.fills: Deque[FillEvent] = deque(maxlen=max_fills)
        self.signals: Deque[SignalEvent] = deque(maxlen=max_signals)
        self.trades: Deque[TradeEvent] = deque(maxlen=max_trades)
        self.equity_curve: Deque[EquityPoint] = deque(maxlen=max_equity_points)
        self.strategy_stats: Dict[str, StrategyStats] = {}

        # In-flight (not yet closed) trades, keyed by f"{market_id}:{outcome_id}".
        # Populated by ``record_fill`` when a fill opens or scales into a
        # position, and consumed when a subsequent fill closes / reduces it.
        self._open_trades: Dict[str, _OpenTrade] = {}

        # number of markets currently in the universe (set on every refresh)
        self.markets_count: int = 0
        self.top_markets: List[dict] = []

        # seed the equity curve with the starting bankroll so charts have a point
        self.record_equity_sample()

    # ------------------------------------------------------------------
    # binding
    # ------------------------------------------------------------------

    def bind_engine(self, engine: "DeltaEngine") -> None:
        self.engine = engine
        self.strategies_enabled = [s.name for s in engine.strategies]
        for name in self.strategies_enabled:
            self.strategy_stats.setdefault(name, StrategyStats(name=name))

    def bind_brain(self, brain: "AdaptiveBrain") -> None:
        """Attach the adaptive brain so its live state shows on the dashboard."""
        self.brain = brain

    # ------------------------------------------------------------------
    # event ingestion (called from engine listeners)
    # ------------------------------------------------------------------

    def record_signals(self, signals: List[Signal]) -> None:
        now = time.time()
        self.last_event_at = now
        for s in signals:
            stats = self.strategy_stats.setdefault(s.strategy, StrategyStats(name=s.strategy))
            stats.signals_emitted += 1
            stats.total_edge += s.edge
            stats.last_signal_ts = now
            self.signals.append(
                SignalEvent(
                    timestamp=now,
                    strategy=s.strategy,
                    edge=s.edge,
                    confidence=s.confidence,
                    legs=len(s.legs),
                    notional=s.total_notional,
                    status="pending",
                    metadata=_safe_metadata(s.metadata),
                )
            )

    def record_execution(self, result: ExecutionResult) -> None:
        now = time.time()
        self.last_event_at = now
        sig = result.signal
        stats = self.strategy_stats.setdefault(
            sig.strategy, StrategyStats(name=sig.strategy)
        )
        status = "executed" if result.success else "rejected"
        if result.success:
            stats.trades_executed += 1
        else:
            stats.trades_rejected += 1

        # Promote the most recent matching "pending" signal of the same strategy
        # to its terminal status, falling back to appending a new row.
        # ``sig`` here is the *post-sized* signal returned by ``Executor.execute``
        # (it ran through ``_clamp_reduce_only_legs`` and ``_size_signal``), so
        # ``sig.total_notional`` reflects the actual stake the risk vet checked.
        # That's what users need to see when debugging a rejection.
        promoted = False
        for ev in reversed(self.signals):
            if ev.strategy == sig.strategy and ev.status == "pending":
                ev.status = status
                ev.reason = result.reason
                ev.notional = sig.total_notional
                ev.metadata = _safe_metadata(sig.metadata)
                promoted = True
                break
        if not promoted:
            self.signals.append(
                SignalEvent(
                    timestamp=now,
                    strategy=sig.strategy,
                    edge=sig.edge,
                    confidence=sig.confidence,
                    legs=len(sig.legs),
                    notional=sig.total_notional,
                    status=status,
                    reason=result.reason,
                    metadata=_safe_metadata(sig.metadata),
                )
            )

        for fill in result.fills:
            self.record_fill(fill, sig)

    def record_fill(self, fill: Fill, signal: Signal) -> None:
        self.fills.append(
            FillEvent(
                timestamp=fill.timestamp,
                strategy=signal.strategy,
                market_id=fill.market_id,
                outcome_id=fill.outcome_id,
                side=fill.side,
                price=fill.price,
                size=fill.size,
                notional=fill.notional,
                fee=fill.fee,
                edge=signal.edge,
                bankroll_after=self.portfolio.bankroll,
            )
        )
        self._track_round_trip(fill, signal)

    def _track_round_trip(self, fill: Fill, signal: Signal) -> None:
        """Pair this fill with its matching open / close half and emit a
        :class:`TradeEvent` whenever a round-trip completes.

        Cases handled:

        * **Open from flat** — record a new ``_OpenTrade``.
        * **Scale in** (same-direction add) — update the weighted-avg
          ``open_price`` and accumulate fees.
        * **Partial close** (opposite-direction smaller than open) — emit a
          ``TradeEvent`` for the closed portion, leave the remainder open.
        * **Full close** — emit the ``TradeEvent`` and drop the open entry.
        * **Flip** (opposite-direction larger than open) — emit a close
          ``TradeEvent`` for the original side, then open a fresh trade for
          the residual on the opposite side.

        This runs *after* ``Portfolio.apply_fill``, so ``portfolio.positions``
        already reflects the post-fill share count; the pre-fill count is
        reconstructed by subtracting the signed fill size.
        """
        key = Portfolio._key(fill.market_id, fill.outcome_id)
        pos = self.portfolio.positions.get(key)
        post_shares = pos.shares if pos is not None else 0.0
        signed = fill.size if fill.side == "BUY" else -fill.size
        pre_shares = post_shares - signed
        leverage = float(getattr(fill, "leverage", 1.0) or 1.0)
        eps = 1e-12

        open_trade = self._open_trades.get(key)

        # Case 1: opening from flat
        if abs(pre_shares) < eps:
            self._open_trades[key] = _OpenTrade(
                opened_at=fill.timestamp,
                side="LONG" if signed > 0 else "SHORT",
                market_id=fill.market_id,
                outcome_id=fill.outcome_id,
                open_price=fill.price,
                open_size=abs(signed),
                open_strategy=signal.strategy,
                leverage=leverage,
                fees_paid=fill.fee,
            )
            return

        # Case 2: same-direction add (scale in) — weighted-avg the open price
        if (pre_shares > 0 and signed > 0) or (pre_shares < 0 and signed < 0):
            if open_trade is None:
                # We came in mid-stream (e.g. dashboard attached after some
                # fills already executed). Resync from the portfolio so we
                # at least have a sensible avg price for future closes.
                avg_cost = pos.avg_cost if pos is not None else fill.price
                open_trade = _OpenTrade(
                    opened_at=fill.timestamp,
                    side="LONG" if pre_shares > 0 else "SHORT",
                    market_id=fill.market_id,
                    outcome_id=fill.outcome_id,
                    open_price=avg_cost,
                    open_size=abs(pre_shares),
                    open_strategy=signal.strategy,
                    leverage=leverage,
                    fees_paid=0.0,
                )
            new_size = open_trade.open_size + abs(signed)
            open_trade.open_price = (
                open_trade.open_price * open_trade.open_size
                + fill.price * abs(signed)
            ) / new_size
            open_trade.open_size = new_size
            open_trade.fees_paid += fill.fee
            self._open_trades[key] = open_trade
            return

        # Case 3: opposite-direction fill — close, partial close, or flip.
        closing_size = min(abs(pre_shares), abs(signed))
        if open_trade is None:
            # Defensive: closing without a recorded open. Best-effort use
            # the portfolio's avg_cost.
            avg_cost = pos.avg_cost if pos is not None else fill.price
            open_trade = _OpenTrade(
                opened_at=fill.timestamp,
                side="LONG" if pre_shares > 0 else "SHORT",
                market_id=fill.market_id,
                outcome_id=fill.outcome_id,
                open_price=avg_cost,
                open_size=abs(pre_shares),
                open_strategy=signal.strategy,
                leverage=leverage,
                fees_paid=0.0,
            )

        direction = 1 if open_trade.side == "LONG" else -1
        gross_pnl = (fill.price - open_trade.open_price) * closing_size * direction
        # Allocate a proportional slice of the accumulated open-side fees to
        # the closed portion; the rest stays attached to any remainder.
        prop = (
            closing_size / open_trade.open_size
            if open_trade.open_size > 0
            else 1.0
        )
        open_fees_allocated = open_trade.fees_paid * prop
        total_fees = open_fees_allocated + fill.fee
        pnl_after_fees = gross_pnl - total_fees

        self.trades.append(
            TradeEvent(
                opened_at=open_trade.opened_at,
                closed_at=fill.timestamp,
                duration_seconds=max(0.0, fill.timestamp - open_trade.opened_at),
                strategy=open_trade.open_strategy,
                market_id=fill.market_id,
                outcome_id=fill.outcome_id,
                side=open_trade.side,
                open_price=open_trade.open_price,
                close_price=fill.price,
                size=closing_size,
                pnl=pnl_after_fees,
                fees=total_fees,
                leverage=open_trade.leverage,
            )
        )

        # Update / drop the open record for the remainder
        remaining = open_trade.open_size - closing_size
        if remaining > eps:
            open_trade.open_size = remaining
            open_trade.fees_paid -= open_fees_allocated
            self._open_trades[key] = open_trade
        else:
            self._open_trades.pop(key, None)

        # Flip: the fill was larger than the existing position, so the
        # residual opens a fresh position on the opposite side.
        flip_size = abs(signed) - closing_size
        if flip_size > eps:
            self._open_trades[key] = _OpenTrade(
                opened_at=fill.timestamp,
                side="LONG" if signed > 0 else "SHORT",
                market_id=fill.market_id,
                outcome_id=fill.outcome_id,
                open_price=fill.price,
                open_size=flip_size,
                open_strategy=signal.strategy,
                leverage=leverage,
                fees_paid=0.0,  # this fill's fee was consumed by the close above
            )

    def record_markets(self, markets: Dict[str, "Market"]) -> None:
        self.markets_count = len(markets)
        # rank by 24h volume for the watchlist panel
        ranked = sorted(
            markets.values(),
            key=lambda m: sum(o.volume_24h for o in m.outcomes.values()),
            reverse=True,
        )[:20]
        out: List[dict] = []
        for m in ranked:
            outcomes_summary = []
            for o in m.outcome_list()[:4]:
                outcomes_summary.append(
                    {
                        "label": o.label,
                        "best_bid": o.best_bid,
                        "best_ask": o.best_ask,
                        "mid": o.mid,
                        "volume_24h": o.volume_24h,
                    }
                )
            out.append(
                {
                    "id": m.id,
                    "slug": m.slug,
                    "question": m.question,
                    "category": m.category,
                    "outcomes": outcomes_summary,
                }
            )
        self.top_markets = out

    def record_equity_sample(self) -> EquityPoint:
        s = self.portfolio.summary()
        pt = EquityPoint(
            timestamp=time.time(),
            bankroll=float(s["bankroll"]),
            locked_margin=float(s.get("locked_margin", 0.0)),
            settled_wealth=float(s.get("settled_wealth", s["bankroll"])),
            equity=float(self.portfolio.equity()),
            drawdown=float(s["drawdown"]),
            growth_multiple=float(s["growth_multiple"]),
        )
        self.equity_curve.append(pt)
        return pt

    # ------------------------------------------------------------------
    # snapshots (read by HTTP routes / websocket)
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        ps = self.portfolio.summary()
        engine_info = self._engine_info()
        latest_equity = self.equity_curve[-1] if self.equity_curve else None

        wins_pnl = [t.pnl for t in self.trades if t.pnl > 0]
        losses_pnl = [t.pnl for t in self.trades if t.pnl < 0]
        closed_pnl = sum(t.pnl for t in self.trades)
        closed_wins = len(wins_pnl)
        closed_losses = len(losses_pnl)
        n_trades = closed_wins + closed_losses
        win_rate = (closed_wins / n_trades) if n_trades else 0.0
        gross_profit = sum(wins_pnl)
        gross_loss = abs(sum(losses_pnl))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )
        avg_win = (gross_profit / closed_wins) if closed_wins else 0.0
        avg_loss = (-gross_loss / closed_losses) if closed_losses else 0.0
        expectancy = (closed_pnl / n_trades) if n_trades else 0.0
        avg_hold_seconds = (
            sum(t.duration_seconds for t in self.trades) / len(self.trades)
            if self.trades
            else 0.0
        )

        # per-strategy PnL aggregated from the closed-trade ledger
        strat_pnl: Dict[str, Dict[str, float]] = {}
        for t in self.trades:
            row = strat_pnl.setdefault(
                t.strategy, {"pnl": 0.0, "wins": 0, "losses": 0, "trades": 0}
            )
            row["pnl"] += t.pnl
            row["trades"] += 1
            if t.pnl > 0:
                row["wins"] += 1
            elif t.pnl < 0:
                row["losses"] += 1

        # exposure breakdown: notional per market (signed)
        exposure: List[dict] = []
        gross_exposure = 0.0
        net_exposure = 0.0
        for pos in self.portfolio.positions.values():
            if pos.shares == 0:
                continue
            notional = pos.notional_exposure(pos.avg_cost)
            signed_notional = notional * (1 if pos.shares > 0 else -1)
            gross_exposure += notional
            net_exposure += signed_notional
            exposure.append(
                {
                    "market_id": pos.market_id,
                    "outcome_id": pos.outcome_id,
                    "notional": float(notional),
                    "signed_notional": float(signed_notional),
                    "side": "LONG" if pos.shares > 0 else "SHORT",
                }
            )
        exposure.sort(key=lambda x: x["notional"], reverse=True)

        # Buying power = settled_wealth * leverage on the venue. We don't
        # have a single leverage on the portfolio, so we approximate using
        # the most-recent fill's leverage, or 1.0 if there have been none.
        approx_leverage = 1.0
        if self.portfolio.fills:
            approx_leverage = float(
                getattr(self.portfolio.fills[-1], "leverage", 1.0) or 1.0
            )
        buying_power = float(ps["settled_wealth"]) * approx_leverage

        # P&L histogram bins (USD). Fixed-width buckets centred at zero.
        pnl_values = [t.pnl for t in self.trades]
        pnl_distribution = _pnl_histogram(pnl_values)

        brain_info = self._brain_info()
        return {
            "ts": time.time(),
            "uptime_seconds": time.time() - self.started_at,
            "portfolio": {
                **{k: _coerce(v) for k, v in ps.items()},
                "equity": float(self.portfolio.equity()),
            },
            "engine": engine_info,
            "brain": brain_info,
            "markets_count": self.markets_count,
            "trades_closed": len(self.trades),
            "closed_trade_pnl": float(closed_pnl),
            "closed_trade_wins": closed_wins,
            "closed_trade_losses": closed_losses,
            "analytics": {
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor) if profit_factor != float("inf") else None,
                "gross_profit": float(gross_profit),
                "gross_loss": float(gross_loss),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "expectancy": float(expectancy),
                "avg_hold_seconds": float(avg_hold_seconds),
                "gross_exposure": float(gross_exposure),
                "net_exposure": float(net_exposure),
                "buying_power": float(buying_power),
                "approx_leverage": float(approx_leverage),
            },
            "exposure": exposure,
            "pnl_distribution": pnl_distribution,
            "strategy_pnl": [
                {"name": name, **vals} for name, vals in strat_pnl.items()
            ],
            "strategies": [
                self.strategy_stats[name].to_dict()
                for name in self.strategy_stats
            ],
            "latest_equity": latest_equity.to_dict() if latest_equity else None,
            "paper_mode": self.paper_mode,
            "fills_count": len(self.fills),
            "signals_count": len(self.signals),
        }

    def _brain_info(self) -> Dict[str, Any]:
        """JSON-serialisable view of the adaptive brain's live state.

        Returns ``{"enabled": False}`` when no brain is attached so the
        dashboard can gate its panel cleanly.
        """
        if self.brain is None or not self.brain.enabled:
            return {"enabled": False}
        bs = self.brain.stats
        return {
            "enabled": True,
            "signals_seen": int(bs.signals_seen),
            "signals_passed": int(bs.signals_passed),
            "signals_vetoed_regime": int(bs.signals_vetoed_regime),
            "signals_vetoed_mute": int(bs.signals_vetoed_mute),
            "signals_vetoed_daily_loss": int(bs.signals_vetoed_daily_loss),
            "signals_vetoed_correlation": int(bs.signals_vetoed_correlation),
            "signals_vetoed_post_loss": int(bs.signals_vetoed_post_loss),
            "signals_shrunk": int(bs.signals_shrunk),
            "daily_pnl": float(bs.daily_pnl),
            "daily_pnl_floor": float(bs.daily_pnl_floor),
            "daily_window_started_at": float(bs.daily_window_started_at),
            "strategy_performance": self.brain.performance(),
            "regimes": self.brain.regimes_snapshot(),
        }

    def _engine_info(self) -> Dict[str, Any]:
        if self.engine is None:
            return {
                "running": False,
                "paused": False,
                "iterations": 0,
                "signals_emitted": 0,
                "trades_executed": 0,
                "trades_rejected": 0,
                "loop_interval_ms": 0,
                "use_websocket": False,
                "strategies": [],
            }
        e = self.engine
        return {
            "running": True,
            "paused": e.paused,
            "iterations": e.stats.iterations,
            "signals_emitted": e.stats.signals_emitted,
            "trades_executed": e.stats.trades_executed,
            "trades_rejected": e.stats.trades_rejected,
            "loop_interval_ms": int(e.loop_interval * 1000),
            "use_websocket": e.use_websocket,
            "strategies": [s.name for s in e.strategies],
        }

    def recent_fills(self, limit: int = 50) -> List[dict]:
        items = list(self.fills)[-limit:]
        return [f.to_dict() for f in reversed(items)]

    def recent_signals(self, limit: int = 50) -> List[dict]:
        items = list(self.signals)[-limit:]
        return [s.to_dict() for s in reversed(items)]

    def recent_trades(self, limit: int = 50) -> List[dict]:
        """Completed round-trip trades, most recent first.

        Each entry surfaces ``open_price``, ``close_price``, and realised
        ``pnl`` for the trade (after fees). Open positions live in
        ``open_positions()``; this view only contains *closed* trades.
        """
        items = list(self.trades)[-limit:]
        return [t.to_dict() for t in reversed(items)]

    def open_positions(self) -> List[dict]:
        out = []
        for pos in self.portfolio.positions.values():
            # FP-tolerant flat check: fills can leave residual shares like
            # 1e-12 that aren't *literally* zero but represent a closed
            # position. Without this the dashboard shows ghost "open
            # positions" with $0.00 notional and shares displayed as 0.00.
            if abs(pos.shares) < 1e-9:
                continue
            out.append(
                {
                    "market_id": pos.market_id,
                    "outcome_id": pos.outcome_id,
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "realised_pnl": pos.realised_pnl,
                    "notional": pos.notional_exposure(pos.avg_cost),
                }
            )
        return out

    def equity_history(self, limit: int = 500) -> List[dict]:
        items = list(self.equity_curve)[-limit:]
        return [p.to_dict() for p in items]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coerce(v: Any) -> Any:
    """Make sure portfolio.summary() values survive JSON encoding."""
    if isinstance(v, float):
        return float(v)
    if isinstance(v, int):
        return int(v)
    return v


def _pnl_histogram(values: List[float], bins: int = 21) -> Dict[str, List[float]]:
    """Bucket a list of trade P&L values into a fixed-width histogram.

    Returns ``{"bins": [edge0, edge1, ...], "counts": [n0, n1, ...]}`` with
    ``len(counts) == len(bins) - 1``. Symmetric around zero so wins and
    losses share the same scale (easier to compare in a bar chart).
    """
    if not values:
        return {"bins": [], "counts": []}
    max_abs = max(abs(v) for v in values) or 1.0
    # Pad a touch so the extreme values land inside the last bucket
    edge = max_abs * 1.05
    step = (2 * edge) / bins
    edges = [-edge + i * step for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = int((v + edge) / step)
        if idx < 0:
            idx = 0
        elif idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return {"bins": edges, "counts": counts}


def _safe_metadata(md: dict) -> dict:
    """Drop non-JSON-friendly values from a Signal.metadata."""
    out = {}
    for k, v in (md or {}).items():
        try:
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[str(k)] = v
            else:
                out[str(k)] = str(v)
        except Exception:
            continue
    return out
