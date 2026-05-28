"""Delta Exchange scan/execute loop.

Coordinates market polling, strategy scans, and execution for Delta's
USD-quoted perpetuals. Designed to run forever as an asyncio task;
cancellation halts cleanly. Exposes ``run``, ``stop``, ``pause`` /
``resume``, ``stats``, ``markets`` and listener hooks the dashboard
subscribes to.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional, Union

from aera.execution import Executor
from aera.execution.executor import ExecutionResult
from aera.logging import get_logger
from aera.markets import DeltaClient, DeltaWebsocket, Market
from aera.strategies import Leg, Signal, Strategy

if TYPE_CHECKING:  # forward ref to avoid runtime import cycle
    from aera.core.brain import AdaptiveBrain
    from aera.core.greedy import GreedyTradeManager


log = get_logger(__name__)


SignalsListener = Callable[[List[Signal]], Union[None, Awaitable[None]]]
ExecutionListener = Callable[[ExecutionResult], Union[None, Awaitable[None]]]
MarketsListener = Callable[[Dict[str, Market]], Union[None, Awaitable[None]]]


@dataclass
class DeltaEngineStats:
    iterations: int = 0
    signals_emitted: int = 0
    signals_inverted: int = 0
    trades_executed: int = 0
    trades_rejected: int = 0
    market_refreshes: int = 0
    # Per-strategy scan counts (incremented each time a strategy.scan() returns).
    # Lets the heartbeat surface "X iterations and strategy Y has emitted 0 signals"
    # so the user can tell quickly that scans are happening but gates are blocking.
    scans_by_strategy: Dict[str, int] = field(default_factory=dict)
    signals_by_strategy: Dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def uptime(self) -> float:
        return time.time() - self.started_at


class DeltaEngine:
    """Coordinates Delta market polling, strategy scans, and execution."""

    def __init__(
        self,
        *,
        strategies: List[Strategy],
        executor: Executor,
        delta: DeltaClient,
        symbols: Optional[List[str]] = None,
        loop_interval_ms: int = 500,
        market_refresh_seconds: Optional[float] = None,
        use_websocket: bool = False,
        on_signals: Optional[SignalsListener] = None,
        on_execution: Optional[ExecutionListener] = None,
        on_markets: Optional[MarketsListener] = None,
        greedy: Optional["GreedyTradeManager"] = None,
        brain: Optional["AdaptiveBrain"] = None,
        heartbeat_seconds: float = 10.0,
        invert_signals: bool = False,
    ) -> None:
        self.strategies = [s for s in strategies if s.enabled]
        self.executor = executor
        self.delta = delta
        self.symbols = symbols
        self.loop_interval = loop_interval_ms / 1000.0
        # Default REST book-refresh cadence. With websocket on, the WS feed
        # streams tick-level book updates and this loop only needs to refresh
        # the market universe (product list + symbol metadata) occasionally,
        # so 5 s is plenty. Without websocket, this is the ONLY way fresh
        # books reach the strategies — keep it fast (2 s) so tick-level
        # strategies have a chance to accumulate rolling state. The previous
        # 30 s default starved every tape/streak/bar strategy in REST mode.
        if market_refresh_seconds is None:
            market_refresh_seconds = 5.0 if use_websocket else 2.0
        self.market_refresh_seconds = market_refresh_seconds
        self.use_websocket = use_websocket
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        # Counter-trade mode: when True, every BUY entry leg is flipped to
        # SELL and vice versa before submission. Reduce-only (closing) legs
        # are passed through unchanged so TP/SL/greedy flatten signals
        # still close the actual position rather than opening a new one
        # in the wrong direction.
        self.invert_signals = bool(invert_signals)
        self.stats = DeltaEngineStats()
        self._stop = asyncio.Event()
        self._paused = asyncio.Event()
        self._markets: Dict[str, Market] = {}
        self._ws: Optional[DeltaWebsocket] = None
        self._ws_consumer_task: Optional[asyncio.Task] = None
        self._on_signals = on_signals
        self._on_execution = on_execution
        self._on_markets = on_markets
        self._leverage_applied: bool = False
        # Seed per-strategy counters so the heartbeat always has names even
        # before the first scan completes.
        for s in self.strategies:
            self.stats.scans_by_strategy.setdefault(s.name, 0)
            self.stats.signals_by_strategy.setdefault(s.name, 0)
        # Optional greedy autopilot. When provided AND enabled, the engine
        # asks it for flatten signals at the top of every step() (so
        # greedy exits beat fresh entries on the same tick) and forwards
        # every ExecutionResult into ``greedy.on_execution`` so the
        # overlay can track open positions and update its win/loss
        # streak. The greedy manager is also expected to be wired into
        # the Executor for leverage override + compound sizing.
        self.greedy = greedy
        # Optional adaptive brain. When provided AND enabled, the engine
        # observes every market refresh through the brain's regime book,
        # filters every collected signal batch through the brain's
        # gating + sizing logic, and forwards every closed round-trip
        # PnL into the brain's per-strategy tracker. The brain owns the
        # "auto-mute losing strategies + refuse to scalp during
        # news/high-vol regimes + daily-loss circuit breaker"
        # behaviour. It never opens trades on its own.
        self.brain = brain
        # Pre-fill helper used to compute per-fill realised PnL on every
        # closing fill so the brain's strategy tracker sees actual round-
        # trip PnL (the engine is the only place we have both the
        # signal source AND the portfolio reference). Maps the
        # ``(market_id, outcome_id)`` key to the pre-fill realised_pnl
        # snapshot so we can take a diff after ``apply_fill`` runs.
        self._pre_fill_realised: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # introspection (duck-typed for the dashboard)
    # ------------------------------------------------------------------

    @property
    def markets(self) -> Dict[str, Market]:
        return self._markets

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # listener plumbing
    # ------------------------------------------------------------------

    async def _emit(self, listener, payload) -> None:
        if listener is None:
            return
        try:
            res = listener(payload)
            if asyncio.iscoroutine(res):
                await res
        except Exception as exc:  # pragma: no cover - dashboard must not break the bot
            log.debug("listener raised: %s", exc)

    # ------------------------------------------------------------------
    # market discovery + book hydration
    # ------------------------------------------------------------------

    async def _refresh_markets(self) -> None:
        markets = await self.delta.list_active_markets(symbols=self.symbols)
        by_id: Dict[str, Market] = {m.id: m for m in markets}
        if not by_id:
            log.warning("delta refresh: 0 markets discovered")
            self._markets = {}
            await self._emit(self._on_markets, self._markets)
            return

        symbols = list(by_id.keys())

        # Apply account leverage once per process if configured. We do this
        # AFTER the first product list so we have the resolved product_ids
        # cached and BEFORE any orders are placed.
        if (
            not self._leverage_applied
            and self.delta.cfg.leverage is not None
            and self.delta.authenticated
        ):
            try:
                await self.delta.apply_account_leverage(
                    leverage=float(self.delta.cfg.leverage),
                    symbols=symbols,
                )
            except Exception as exc:
                log.warning("delta: leverage application failed: %s", exc)
            finally:
                self._leverage_applied = True

        books = await self.delta.fetch_books_batch(symbols)
        for sym, m in by_id.items():
            book = books.get(sym)
            outcome = next(iter(m.outcomes.values()), None)
            if outcome is not None and book is not None:
                outcome.book = book
            m.last_update = time.time()
        self._markets = by_id

        if self.use_websocket:
            await self._sync_websocket(symbols)

        await self._emit(self._on_markets, self._markets)

    async def _sync_websocket(self, symbols: List[str]) -> None:
        if self._ws is None:
            self._ws = DeltaWebsocket(
                symbols=symbols,
                url=self.delta.cfg.ws_url,
                api_key=self.delta.cfg.api_key,
                api_secret=self.delta.cfg.api_secret,
            )
            await self._ws.start()
            self._ws_consumer_task = asyncio.create_task(
                self._consume_ws(), name="delta-ws-consumer"
            )
        else:
            await self._ws.add_symbols(symbols)

    async def _consume_ws(self) -> None:
        if self._ws is None:
            return
        async for upd in self._ws.updates():
            m = self._markets.get(upd.symbol)
            if m is None:
                continue
            outcome = next(iter(m.outcomes.values()), None)
            if outcome is None:
                continue
            outcome.book = upd.book
            m.last_update = upd.received_at

    # ------------------------------------------------------------------
    # scan & execute
    # ------------------------------------------------------------------

    async def step(self) -> int:
        market_list = list(self._markets.values())
        all_signals: List[Signal] = []

        # Brain regime observation — must run before any signal
        # filtering so the per-symbol detectors are warm for this tick.
        if self.brain is not None and self.brain.enabled:
            try:
                self.brain.observe_markets(self._markets)
            except Exception as exc:
                log.exception("brain observe_markets crashed: %s", exc)

        # Greedy overlay runs FIRST so its dynamic TP/SL/trailing exits
        # beat fresh entries on the same tick. The strategies still
        # contribute their own exit signals (per-strategy TP/SL paths)
        # which is fine — once greedy has flattened, the strategy's
        # close legs become no-ops via _clamp_reduce_only_legs.
        if self.greedy is not None and self.greedy.enabled:
            try:
                all_signals.extend(self.greedy.proposed_closes(self._markets))
            except Exception as exc:
                log.exception("greedy proposed_closes crashed: %s", exc)

        for strat in self.strategies:
            try:
                emitted = list(strat.scan(market_list))
            except Exception as exc:
                log.exception("strategy %s crashed: %s", strat.name, exc)
                continue
            self.stats.scans_by_strategy[strat.name] = (
                self.stats.scans_by_strategy.get(strat.name, 0) + 1
            )
            if emitted:
                if self.invert_signals:
                    emitted = [self._invert_signal(s) for s in emitted]
                self.stats.signals_by_strategy[strat.name] = (
                    self.stats.signals_by_strategy.get(strat.name, 0) + len(emitted)
                )
                all_signals.extend(emitted)

        # Brain signal filter — vetoes / shrinks fresh entries based on
        # regime, per-strategy live performance, daily loss cap, and
        # correlation cap. Reduce-only (closing) legs always flow.
        # The brain stamps ``brain_vetoed=True`` + ``brain_veto_reason``
        # on every signal it drops; we surface ALL signals to the
        # dashboard so the user can SEE the vetoes (vs the previous
        # behaviour where vetoed signals silently disappeared).
        kept_signals: List[Signal] = all_signals
        vetoed_signals: List[Signal] = []
        if self.brain is not None and self.brain.enabled and all_signals:
            try:
                kept_signals = self.brain.filter_signals(all_signals, self._markets)
                kept_ids = {id(s) for s in kept_signals}
                vetoed_signals = [s for s in all_signals if id(s) not in kept_ids]
                if vetoed_signals:
                    log.debug(
                        "brain: filtered %d -> %d signals (%d vetoed)",
                        len(all_signals), len(kept_signals), len(vetoed_signals),
                    )
            except Exception as exc:
                log.exception("brain filter_signals crashed: %s", exc)

        # Surface EVERYTHING the strategies emitted (kept + vetoed) to
        # the dashboard so the user can see why brain vetoes happened.
        # The kept signals go through their normal "pending -> executed
        # /rejected" lifecycle via ``record_execution``; vetoed ones
        # are emitted as a synthetic batch which the dashboard
        # records as "rejected (brain veto)" via
        # ``_record_brain_vetoes``.
        if all_signals:
            await self._emit(self._on_signals, kept_signals + vetoed_signals)
        for sig in vetoed_signals:
            await self._record_brain_veto(sig)

        all_signals = kept_signals
        if not all_signals:
            return 0

        all_signals.sort(key=lambda s: s.edge, reverse=True)
        self.stats.signals_emitted += len(all_signals)

        executed = 0
        for sig in all_signals:
            # Snapshot realised PnL per-position before the execute so
            # we can compute the delta and feed closed round-trip PnLs
            # into the brain's per-strategy tracker. We only snapshot
            # the legs we're about to touch (cheap dict lookup).
            pre_realised: Dict[str, float] = {}
            for leg in sig.legs:
                key = f"{leg.market_id}:{leg.outcome_id}"
                pos = self.executor.portfolio.positions.get(key)
                pre_realised[key] = float(pos.realised_pnl) if pos is not None else 0.0

            result = await self.executor.execute(sig, self._markets)
            # Greedy is informed of every execution result (success and
            # failure) so it can track open positions and refresh its
            # win/loss streak before the next leverage decision.
            if self.greedy is not None and self.greedy.enabled:
                try:
                    self.greedy.on_execution(result)
                except Exception as exc:
                    log.exception("greedy on_execution crashed: %s", exc)
            # Brain absorption — pipe every successful close's realised
            # PnL into its strategy tracker. We compute (post − pre)
            # per touched key, which captures the round-trip outcome
            # whether the close was full, partial, or a flip.
            if self.brain is not None and self.brain.enabled and result.success:
                try:
                    self.brain.on_execution(result)
                    for key, pre in pre_realised.items():
                        pos = self.executor.portfolio.positions.get(key)
                        if pos is None:
                            continue
                        delta = float(pos.realised_pnl) - pre
                        if abs(delta) < 1e-9:
                            continue
                        # ``key`` is "<market_id>:<outcome_id>" — the
                        # market_id is the symbol the brain keys its
                        # post-loss cool-down by.
                        symbol = key.split(":", 1)[0] if ":" in key else None
                        self.brain.on_trade_closed(
                            sig.strategy, delta, symbol=symbol,
                        )
                except Exception as exc:
                    log.exception("brain on_execution/on_trade_closed crashed: %s", exc)
            await self._emit(self._on_execution, result)
            if result.success:
                executed += 1
                self.stats.trades_executed += 1
                log.info(
                    "[bold green]EXECUTED[/] %s edge=%.4f legs=%d bankroll=%.4f",
                    sig.strategy, sig.edge, len(sig.legs),
                    self.executor.portfolio.bankroll,
                )
            else:
                self.stats.trades_rejected += 1
                log.debug("REJECTED %s: %s", sig.strategy, result.reason)
        return executed

    # ------------------------------------------------------------------
    # brain veto bookkeeping
    # ------------------------------------------------------------------

    async def _record_brain_veto(self, sig: Signal) -> None:
        """Tell listeners a brain-vetoed signal was rejected.

        Builds a synthetic ``ExecutionResult`` with ``success=False``
        and the brain's veto reason so the dashboard's existing
        pending → rejected pipeline lights up the entry. Without this,
        vetoed signals silently disappeared and the user thought the
        bot was idle.
        """
        reason = sig.metadata.get("brain_veto_reason", "brain veto") if sig.metadata else "brain veto"
        result = ExecutionResult(
            signal=sig, fills=[], success=False, reason=f"brain: {reason}",
        )
        self.stats.trades_rejected += 1
        await self._emit(self._on_execution, result)

    # ------------------------------------------------------------------
    # signal inversion (counter-trade mode)
    # ------------------------------------------------------------------

    def _invert_signal(self, sig: Signal) -> Signal:
        """Return a copy of ``sig`` with every entry leg's side flipped.

        Reduce-only legs are passed through unchanged so TP/SL exits and
        greedy flatten signals still close the actual open position
        (flipping them would open a new doubled position in the wrong
        direction). The ``limit_price`` is also re-anchored to the
        opposite-side touch where the book is available — otherwise the
        paper exchange's 0.1% limit-tolerance check would reject the
        flipped fill (a SELL at the original BUY's ask limit lies far
        below the new fill price at the bid).
        """
        new_legs: List[Leg] = []
        flipped = 0
        for leg in sig.legs:
            if leg.reduce_only:
                new_legs.append(leg)
                continue
            opposite = "SELL" if leg.side == "BUY" else "BUY"
            new_price = self._opposite_touch(leg, opposite) or leg.limit_price
            new_legs.append(
                Leg(
                    market_id=leg.market_id,
                    outcome_id=leg.outcome_id,
                    side=opposite,
                    limit_price=float(new_price),
                    size_usd=leg.size_usd,
                    reason=f"INVERTED({leg.side}->{opposite}) {leg.reason}".strip(),
                    leverage=leg.leverage,
                    reduce_only=False,
                    time_in_force=getattr(leg, "time_in_force", None),
                    post_only=getattr(leg, "post_only", None),
                )
            )
            flipped += 1

        if flipped == 0:
            return sig

        self.stats.signals_inverted += flipped
        new_meta = dict(sig.metadata)
        new_meta["inverted"] = True
        new_meta["inverted_legs"] = flipped
        return Signal(
            strategy=sig.strategy,
            confidence=sig.confidence,
            edge=sig.edge,
            legs=new_legs,
            metadata=new_meta,
        )

    def _opposite_touch(self, leg: Leg, new_side: str) -> Optional[float]:
        """Best-bid (for new SELL) or best-ask (for new BUY) from the book.

        Returns ``None`` when the market or book is unavailable, in which
        case the caller falls back to the original ``limit_price``.
        """
        market = self._markets.get(leg.market_id)
        if market is None:
            return None
        outcome = market.outcomes.get(leg.outcome_id)
        if outcome is None or outcome.book is None:
            return None
        if new_side == "BUY":
            ask = outcome.book.best_ask()
            return float(ask.price) if ask is not None else None
        bid = outcome.book.best_bid()
        return float(bid.price) if bid is not None else None

    async def run(self) -> None:
        log.info(
            "[bold cyan]delta engine starting[/] strategies=%s symbols=%s ws=%s "
            "loop=%dms refresh=%.1fs heartbeat=%.0fs",
            [s.name for s in self.strategies], self.symbols, self.use_websocket,
            int(self.loop_interval * 1000), self.market_refresh_seconds,
            self.heartbeat_seconds,
        )
        if not self.use_websocket:
            log.warning(
                "[bold yellow]REST MODE[/] — book updates only every %.1fs. "
                "Tick/tape/streak strategies (order_book_sniper, flow_scalp, "
                "tick_reversal_scalp, micro_vwap_sniper, stop_hunt_reversal, "
                "bid_ask_spread_fade) need a live tick feed to fire — re-run "
                "with [bold]--websocket[/] for sub-second order-book updates.",
                self.market_refresh_seconds,
            )
        last_refresh = 0.0
        last_heartbeat = time.time()
        try:
            while not self._stop.is_set():
                self.stats.iterations += 1
                now = time.time()
                if now - last_refresh > self.market_refresh_seconds:
                    try:
                        await self._refresh_markets()
                        self.stats.market_refreshes += 1
                    except Exception as exc:
                        log.warning("delta refresh failed: %s", exc)
                    last_refresh = now
                if not self._paused.is_set():
                    await self.step()
                if now - last_heartbeat >= self.heartbeat_seconds:
                    self._log_heartbeat()
                    last_heartbeat = now
                await asyncio.sleep(self.loop_interval)
        finally:
            await self._shutdown()
            log.info(
                "delta engine stopped after %.1fs, stats=%s",
                self.stats.uptime(), self.stats,
            )

    def _log_heartbeat(self) -> None:
        """Periodic loop-alive line with per-strategy counters.

        Lets the user verify the loop is firing, books are refreshing, and
        which strategies have actually emitted any signals (vs strategies
        whose gates are still blocking — those show 0 signals despite many
        scans, which is the clearest "warmup" indicator).
        """
        st = self.stats
        feed = "WS" if self.use_websocket else "REST"
        per_strat = " ".join(
            f"{name}={st.signals_by_strategy.get(name, 0)}/"
            f"{st.scans_by_strategy.get(name, 0)}"
            for name in (s.name for s in self.strategies)
        )
        brain_tail = ""
        if self.brain is not None and self.brain.enabled:
            bs = self.brain.stats
            muted = sum(
                1 for p in self.brain._perf.values()  # type: ignore[attr-defined]
                if p.muted_until > time.time()
            )
            brain_tail = (
                f"  brain: passed={bs.signals_passed} "
                f"regime_veto={bs.signals_vetoed_regime} "
                f"mute_veto={bs.signals_vetoed_mute} "
                f"daily_veto={bs.signals_vetoed_daily_loss} "
                f"corr_veto={bs.signals_vetoed_correlation} "
                f"postloss_veto={bs.signals_vetoed_post_loss} "
                f"shrunk={bs.signals_shrunk} muted_now={muted} "
                f"daily_pnl=${bs.daily_pnl:+.2f}"
            )
        log.info(
            "heartbeat · up=%.0fs iters=%d refresh=%d feed=%s markets=%d "
            "signals=%d exec=%d rej=%d  per-strat[signals/scans]: %s%s",
            st.uptime(), st.iterations, st.market_refreshes, feed,
            len(self._markets), st.signals_emitted, st.trades_executed,
            st.trades_rejected, per_strat or "—", brain_tail,
        )

    async def _shutdown(self) -> None:
        if self._ws_consumer_task is not None:
            self._ws_consumer_task.cancel()
            try:
                await self._ws_consumer_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ws_consumer_task = None
        if self._ws is not None:
            await self._ws.stop()
            self._ws = None
