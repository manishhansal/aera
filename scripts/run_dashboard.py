"""Run the Delta bot AND the live web dashboard in a single process.

Open ``http://127.0.0.1:8787`` in a browser once it's up.

Usage:
    python -m scripts.run_dashboard --bankroll 27
    python -m scripts.run_dashboard --websocket
    python -m scripts.run_dashboard --host 0.0.0.0 --port 8080
    python -m scripts.run_dashboard --live   # requires DELTA_API_KEY/SECRET
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aera.core import (
    AdaptiveBrain,
    DeltaEngine,
    GreedyTradeManager,
    Portfolio,
    RiskManager,
)
from aera.dashboard import DashboardState, run_dashboard
from aera.execution import (
    DeltaLiveExchange,
    DeltaPaperExchange,
    Executor,
    LinearSlippageModel,
)
from aera.markets import DeltaClient
from aera.settings import get_settings
from aera.strategies import (
    BidAskSpreadFade,
    DeltaPerpetualScalper,
    FlowScalp,
    OrderBookSniper,
    TickReversalScalp,
)


def build_delta_strategies(names: list[str], portfolio: Portfolio) -> list:
    settings = get_settings()
    scalper_cfg = settings.strategies.delta_perp_scalper
    sniper_cfg = settings.strategies.order_book_sniper
    tick_cfg = settings.strategies.tick_reversal_scalp
    mm_cfg = settings.strategies.bid_ask_spread_fade
    flow_cfg = settings.strategies.flow_scalp
    avail = {
        "delta_perp_scalper": lambda: DeltaPerpetualScalper(
            zscore_window=scalper_cfg.zscore_window,
            zscore_entry=scalper_cfg.zscore_entry,
            ofi_threshold=scalper_cfg.ofi_threshold,
            min_edge=scalper_cfg.min_edge,
            notional_usd=scalper_cfg.notional_usd,
            min_depth_contracts=scalper_cfg.min_depth_contracts,
            rearm_distance_bps=scalper_cfg.rearm_distance_bps,
            take_profit_pct=scalper_cfg.take_profit_pct,
            stop_loss_pct=scalper_cfg.stop_loss_pct,
            take_profit_usd=scalper_cfg.take_profit_usd,
            stop_loss_usd=scalper_cfg.stop_loss_usd,
            portfolio=portfolio,
            enabled=scalper_cfg.enabled,
        ),
        "order_book_sniper": lambda: OrderBookSniper(
            imbalance_ratio=sniper_cfg.imbalance_ratio,
            imbalance_band_bps=sniper_cfg.imbalance_band_bps,
            imbalance_max_levels=sniper_cfg.imbalance_max_levels,
            tape_min_count=sniper_cfg.tape_min_count,
            tape_window_seconds=sniper_cfg.tape_window_seconds,
            notional_usd=sniper_cfg.notional_usd,
            take_profit_pct=sniper_cfg.take_profit_pct,
            stop_loss_pct=sniper_cfg.stop_loss_pct,
            take_profit_usd=sniper_cfg.take_profit_usd,
            stop_loss_usd=sniper_cfg.stop_loss_usd,
            max_hold_seconds=sniper_cfg.max_hold_seconds,
            spoof_min_wall_contracts=sniper_cfg.spoof_min_wall_contracts,
            spoof_persist_seconds=sniper_cfg.spoof_persist_seconds,
            spoof_vanish_ratio=sniper_cfg.spoof_vanish_ratio,
            entry_tick_offset=sniper_cfg.entry_tick_offset,
            rearm_distance_bps=sniper_cfg.rearm_distance_bps,
            min_edge=sniper_cfg.min_edge,
            portfolio=portfolio,
            enabled=sniper_cfg.enabled,
        ),
        "tick_reversal_scalp": lambda: TickReversalScalp(
            min_streak=tick_cfg.min_streak,
            max_buffer_ticks=tick_cfg.max_buffer_ticks,
            size_decay_threshold=tick_cfg.size_decay_threshold,
            sr_band_bps=tick_cfg.sr_band_bps,
            sr_lookback_ticks=tick_cfg.sr_lookback_ticks,
            require_depth_trend=tick_cfg.require_depth_trend,
            max_spread_multiple=tick_cfg.max_spread_multiple,
            spread_ema_alpha=tick_cfg.spread_ema_alpha,
            volume_spike_multiple=tick_cfg.volume_spike_multiple,
            volume_short_window_seconds=tick_cfg.volume_short_window_seconds,
            volume_long_window_seconds=tick_cfg.volume_long_window_seconds,
            news_lookback_seconds=tick_cfg.news_lookback_seconds,
            news_max_tick_bps=tick_cfg.news_max_tick_bps,
            notional_usd=tick_cfg.notional_usd,
            entry_offset_bps=tick_cfg.entry_offset_bps,
            take_profit_pct=tick_cfg.take_profit_pct,
            stop_loss_pct=tick_cfg.stop_loss_pct,
            take_profit_usd=tick_cfg.take_profit_usd,
            stop_loss_usd=tick_cfg.stop_loss_usd,
            max_hold_seconds=tick_cfg.max_hold_seconds,
            rearm_distance_bps=tick_cfg.rearm_distance_bps,
            min_edge=tick_cfg.min_edge,
            portfolio=portfolio,
            enabled=tick_cfg.enabled,
        ),
        "bid_ask_spread_fade": lambda: BidAskSpreadFade(
            min_spread_pct=mm_cfg.min_spread_pct,
            capture_target=mm_cfg.capture_target,
            quote_size_usd=mm_cfg.quote_size_usd,
            max_inventory_usd=mm_cfg.max_inventory_usd,
            inventory_skew_threshold_usd=mm_cfg.inventory_skew_threshold_usd,
            inventory_skew_ticks=mm_cfg.inventory_skew_ticks,
            refresh_rate_ms=mm_cfg.refresh_rate_ms,
            kill_move_pct=mm_cfg.kill_move_pct,
            kill_window_seconds=mm_cfg.kill_window_seconds,
            maker_fee_bps=mm_cfg.maker_fee_bps,
            min_net_edge_bps=mm_cfg.min_net_edge_bps,
            leverage_override=mm_cfg.leverage_override,
            min_edge=mm_cfg.min_edge,
            portfolio=portfolio,
            enabled=mm_cfg.enabled,
        ),
        "flow_scalp": lambda: FlowScalp(
            whale_multiple=flow_cfg.whale_multiple,
            confirm_multiple=flow_cfg.confirm_multiple,
            confirm_count=flow_cfg.confirm_count,
            confirm_window_seconds=flow_cfg.confirm_window_seconds,
            avg_window=flow_cfg.avg_window,
            tape_max_trades=flow_cfg.tape_max_trades,
            notional_usd=flow_cfg.notional_usd,
            take_profit_pct=flow_cfg.take_profit_pct,
            stop_loss_pct=flow_cfg.stop_loss_pct,
            trailing_stop_pct=flow_cfg.trailing_stop_pct,
            take_profit_usd=flow_cfg.take_profit_usd,
            stop_loss_usd=flow_cfg.stop_loss_usd,
            max_hold_seconds=flow_cfg.max_hold_seconds,
            rearm_distance_bps=flow_cfg.rearm_distance_bps,
            leverage_override=flow_cfg.leverage_override,
            min_edge=flow_cfg.min_edge,
            auto_infer_from_book=flow_cfg.auto_infer_from_book,
            portfolio=portfolio,
            enabled=flow_cfg.enabled,
        ),
    }
    # "all" → every available strategy. The DeltaEngine will then drop
    # any whose ``enabled`` flag is false in config.yaml, so the YAML is
    # the single source of truth for strategy activation.
    if any(n == "all" for n in names):
        names = list(avail.keys())
    return [avail[n]() for n in names if n in avail]


async def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the aera trading bot with a live web dashboard.",
    )
    ap.add_argument("--bankroll", type=float, default=None)
    ap.add_argument(
        "--strategies", default=None,
        help=(
            "comma-separated list of strategies, or 'all' (default) to "
            "use every strategy whose ``enabled: true`` flag is set in "
            "config/config.yaml. Available: delta_perp_scalper, "
            "order_book_sniper, tick_reversal_scalp, bid_ask_spread_fade, "
            "flow_scalp, micro_vwap_sniper, stop_hunt_reversal."
        ),
    )
    ap.add_argument(
        "--symbols", default=None,
        help="comma-separated symbol list (e.g. BTCUSD,ETHUSD)",
    )
    ap.add_argument(
        "--live", action="store_true",
        help="route REAL orders (needs DELTA_API_KEY + DELTA_API_SECRET)",
    )
    ap.add_argument("--websocket", action="store_true", help="use the Delta websocket feed")
    ap.add_argument("--host", default="127.0.0.1", help="dashboard bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8787, help="dashboard port (default: 8787)")
    ap.add_argument(
        "--push-interval-ms", type=int, default=1000,
        help="how often the websocket pushes a snapshot (default: 1000)",
    )
    ap.add_argument(
        "--no-engine", action="store_true",
        help="serve the dashboard only (useful for UI development)",
    )
    args = ap.parse_args()

    settings = get_settings()
    bankroll = args.bankroll if args.bankroll is not None else settings.bot.bankroll
    console = Console()
    console.rule(
        f"[bold cyan]aera dashboard · delta[/]  "
        f"bankroll=${bankroll} · http://{args.host}:{args.port}",
    )

    portfolio = Portfolio(
        bankroll=bankroll,
        dust_threshold_usd=float(settings.execution.min_order_size_usd),
    )
    state = DashboardState(portfolio)
    state.paper_mode = not args.live

    server_task = asyncio.create_task(
        run_dashboard(
            state,
            host=args.host,
            port=args.port,
            push_interval_ms=args.push_interval_ms,
        ),
        name="dashboard-server",
    )

    if args.no_engine:
        console.print("[yellow]--no-engine set: serving dashboard only[/]")
        try:
            await server_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        return

    risk = RiskManager(settings.risk, portfolio)
    await _run_delta(args, settings, portfolio, risk, state, server_task, console)


async def _run_delta(args, settings, portfolio, risk, state, server_task, console):
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols
        else None
    )
    # "all" → expand to every strategy in config.yaml; the engine drops
    # any whose ``enabled`` flag is false. Same default semantic as
    # `python -m scripts.run_delta`.
    names = args.strategies or "all"
    strategies = build_delta_strategies(
        [s.strip() for s in names.split(",")],
        portfolio,
    )
    if not strategies:
        console.print("[red]no valid Delta strategies selected[/]")
        server_task.cancel()
        return

    cfg_delta = settings.markets.delta
    async with DeltaClient(cfg_delta) as delta:
        if args.live:
            if not delta.authenticated:
                console.print(
                    "[red]--live requires DELTA_API_KEY + DELTA_API_SECRET in env[/]"
                )
                server_task.cancel()
                return
            exchange = DeltaLiveExchange(
                delta,
                min_trade_notional_usd=cfg_delta.min_trade_notional_usd,
                max_notional_overshoot=cfg_delta.max_notional_overshoot,
            )
            lev = cfg_delta.leverage
            lev_str = f"{lev}x" if lev is not None else "(account default)"
            console.print(
                f"[bold red]DELTA LIVE MODE — real orders will be placed · leverage={lev_str}[/]"
            )
        else:
            exchange = DeltaPaperExchange(
                slippage=LinearSlippageModel(
                    bps=settings.execution.default_slippage_bps
                ),
                min_trade_notional_usd=cfg_delta.min_trade_notional_usd,
                max_notional_overshoot=cfg_delta.max_notional_overshoot,
                taker_fee_bps=settings.execution.taker_fee_bps,
            )
        greedy_mgr = GreedyTradeManager(
            settings.greedy,
            portfolio,
            taker_fee_bps=settings.execution.taker_fee_bps,
        )
        executor = Executor(portfolio, risk, exchange, greedy=greedy_mgr)
        if greedy_mgr.enabled:
            console.print(
                f"[bold magenta]greedy autopilot ON[/] "
                f"min_profit=${settings.greedy.min_profit_usd:.2f} "
                f"lev=[{settings.greedy.min_leverage:g}..{settings.greedy.max_leverage:g}]x "
                f"compound={settings.greedy.compound_fraction:.0%}"
            )
        brain = AdaptiveBrain(
            settings.brain, portfolio,
            taker_fee_bps=float(settings.execution.taker_fee_bps),
        )
        if brain.enabled:
            console.print(
                f"[bold cyan]adaptive brain ON[/] "
                f"min_wr={settings.brain.min_win_rate:.0%} "
                f"max_streak={settings.brain.max_strategy_loss_streak} "
                f"mute={settings.brain.mute_seconds:.0f}s "
                f"daily_loss_cap={settings.brain.daily_loss_pct:.0%} "
                f"gross_exp_cap={settings.brain.max_gross_exposure_mult:g}x"
            )
            state.bind_brain(brain)

        engine = DeltaEngine(
            strategies=strategies,
            executor=executor,
            delta=delta,
            symbols=symbols,
            loop_interval_ms=settings.bot.loop_interval_ms,
            use_websocket=args.websocket,
            on_signals=state.record_signals,
            on_execution=state.record_execution,
            on_markets=state.record_markets,
            greedy=greedy_mgr,
            brain=brain,
        )
        state.bind_engine(engine)
        mode = "LIVE" if args.live else "PAPER"
        await _supervise(
            engine, server_task, portfolio, console,
            f"delta {mode} ws={args.websocket} symbols={symbols or settings.markets.delta.symbols}",
        )


async def _supervise(engine, server_task, portfolio, console, banner: str) -> None:
    engine_task = asyncio.create_task(engine.run(), name="engine")
    console.print(
        f"[green]engine running[/] strategies={[s.name for s in engine.strategies]} {banner}"
    )
    try:
        done, _pending = await asyncio.wait(
            {engine_task, server_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            exc = t.exception()
            if exc:
                console.print(f"[red]{t.get_name()} crashed:[/] {exc}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        engine.stop()
        for t in (engine_task, server_task):
            if not t.done():
                t.cancel()
        for t in (engine_task, server_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        console.print(portfolio.summary())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
