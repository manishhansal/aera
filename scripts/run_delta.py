"""Run the bot against Delta Exchange (paper or live).

The default mode is **paper-trading**: orders never leave the process, fills
are simulated against the live order book using the configured slippage
model. Pass ``--live`` (and have DELTA_API_KEY + DELTA_API_SECRET set) to
route real orders to Delta.

Optionally bring up the live web dashboard with ``--dashboard`` so you can
watch fills, equity, and per-strategy stats in your browser.

Examples
--------
    # Paper-trade BTCUSD and ETHUSD perpetuals
    python -m scripts.run_delta --bankroll 5.0

    # Same, with the dashboard at http://127.0.0.1:8787
    python -m scripts.run_delta --bankroll 5.0 --dashboard

    # Use the live websocket book feed for sub-second reactions
    python -m scripts.run_delta --websocket --dashboard

    # Live trading (requires DELTA_API_KEY + DELTA_API_SECRET; uses small size)
    python -m scripts.run_delta --bankroll 10.0 --live --duration-seconds 300
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List

from rich.console import Console
from rich.live import Live
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aera.core import (
    AdaptiveBrain,
    DeltaEngine,
    GreedyTradeManager,
    Portfolio,
    RiskManager,
)
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
    MicroVWAPSniper,
    OrderBookSniper,
    StopHuntReversal,
    TickReversalScalp,
)


# Sentinel passed by the CLI to mean "every strategy whose
# `enabled: true` flag is set in config.yaml". Lets users curate the
# active strategy list in a single place (the YAML) instead of having
# to remember to keep CLI flags and YAML in sync.
ALL_STRATEGIES_SENTINEL = "all"


def build_strategies(names: List[str], portfolio: Portfolio) -> list:
    settings = get_settings()
    scalper_cfg = settings.strategies.delta_perp_scalper
    sniper_cfg = settings.strategies.order_book_sniper
    tick_cfg = settings.strategies.tick_reversal_scalp
    mm_cfg = settings.strategies.bid_ask_spread_fade
    flow_cfg = settings.strategies.flow_scalp
    vwap_cfg = settings.strategies.micro_vwap_sniper
    sweep_cfg = settings.strategies.stop_hunt_reversal
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
        "micro_vwap_sniper": lambda: MicroVWAPSniper(
            vwap_window_seconds=vwap_cfg.vwap_window_seconds,
            deviation_pct=vwap_cfg.deviation_pct,
            volume_short_seconds=vwap_cfg.volume_short_seconds,
            volume_long_seconds=vwap_cfg.volume_long_seconds,
            volume_ratio_max=vwap_cfg.volume_ratio_max,
            notional_usd=vwap_cfg.notional_usd,
            take_profit_pct=vwap_cfg.take_profit_pct,
            tp_extra_bps=vwap_cfg.tp_extra_bps,
            stop_loss_pct=vwap_cfg.stop_loss_pct,
            take_profit_usd=vwap_cfg.take_profit_usd,
            stop_loss_usd=vwap_cfg.stop_loss_usd,
            max_hold_seconds=vwap_cfg.max_hold_seconds,
            max_spread_pct=vwap_cfg.max_spread_pct,
            hour_skip_seconds=vwap_cfg.hour_skip_seconds,
            leverage_override=vwap_cfg.leverage_override,
            rearm_distance_bps=vwap_cfg.rearm_distance_bps,
            min_edge=vwap_cfg.min_edge,
            portfolio=portfolio,
            enabled=vwap_cfg.enabled,
        ),
        "stop_hunt_reversal": lambda: StopHuntReversal(
            bar_seconds=sweep_cfg.bar_seconds,
            max_bars=sweep_cfg.max_bars,
            swing_lookback_bars=sweep_cfg.swing_lookback_bars,
            swing_pivot_strength=sweep_cfg.swing_pivot_strength,
            swing_count=sweep_cfg.swing_count,
            wick_size_pct=sweep_cfg.wick_size_pct,
            body_ratio_max=sweep_cfg.body_ratio_max,
            recovery_seconds=sweep_cfg.recovery_seconds,
            volume_multiple=sweep_cfg.volume_multiple,
            volume_lookback_bars=sweep_cfg.volume_lookback_bars,
            delta_flip_threshold=sweep_cfg.delta_flip_threshold,
            require_delta_confirmation=sweep_cfg.require_delta_confirmation,
            take_profit_pct=sweep_cfg.take_profit_pct,
            tp1_pct=sweep_cfg.tp1_pct,
            tp1_fraction=sweep_cfg.tp1_fraction,
            stop_extra_pct=sweep_cfg.stop_extra_pct,
            stop_loss_pct=sweep_cfg.stop_loss_pct,
            take_profit_usd=sweep_cfg.take_profit_usd,
            stop_loss_usd=sweep_cfg.stop_loss_usd,
            max_hold_seconds=sweep_cfg.max_hold_seconds,
            leverage_override=sweep_cfg.leverage_override,
            notional_usd=sweep_cfg.notional_usd,
            rearm_distance_bps=sweep_cfg.rearm_distance_bps,
            min_edge=sweep_cfg.min_edge,
            portfolio=portfolio,
            enabled=sweep_cfg.enabled,
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
    }
    # "all" → expand to every available strategy. The DeltaEngine
    # constructor already drops strategies whose ``enabled`` flag is
    # False (see ``self.strategies = [s for s in strategies if s.enabled]``
    # in delta_engine.py), so passing every name here is exactly
    # equivalent to "use whatever the YAML says is enabled" — which is
    # the principle of least surprise for a config-driven bot.
    if any(n == ALL_STRATEGIES_SENTINEL for n in names):
        names = list(avail.keys())

    out = []
    for n in names:
        if n in avail:
            out.append(avail[n]())
    return out


async def status_render(portfolio: Portfolio, engine: DeltaEngine, console: Console) -> None:
    with Live(refresh_per_second=2, console=console) as live:
        while True:
            s = portfolio.summary()
            t = Table(title="delta · portfolio")
            t.add_column("metric")
            t.add_column("value")
            for k, v in s.items():
                if isinstance(v, float):
                    if "bankroll" in k or "pnl" in k:
                        t.add_row(k, f"${v:,.4f}")
                    elif "drawdown" in k:
                        t.add_row(k, f"{v*100:.2f}%")
                    elif "multiple" in k:
                        t.add_row(k, f"{v:.4f}x")
                    else:
                        t.add_row(k, f"{v:.4f}")
                else:
                    t.add_row(k, str(v))
            t.add_row("iterations", str(engine.stats.iterations))
            t.add_row("signals emitted", str(engine.stats.signals_emitted))
            t.add_row("trades executed", str(engine.stats.trades_executed))
            t.add_row("trades rejected", str(engine.stats.trades_rejected))
            live.update(t)
            await asyncio.sleep(1.0)


async def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the aera bot against Delta Exchange.",
    )
    ap.add_argument("--bankroll", type=float, default=None)
    ap.add_argument(
        "--strategies",
        default=ALL_STRATEGIES_SENTINEL,
        help=(
            "comma-separated list of Delta strategies, or 'all' "
            "(default) to use every strategy whose ``enabled: true`` "
            "flag is set in config/config.yaml. "
            "Available: delta_perp_scalper, order_book_sniper, "
            "tick_reversal_scalp, bid_ask_spread_fade, flow_scalp, "
            "micro_vwap_sniper, stop_hunt_reversal."
        ),
    )
    ap.add_argument("--symbols", default=None, help="comma-separated; e.g. BTCUSD,ETHUSD")
    ap.add_argument("--websocket", action="store_true", help="use the Delta WS book feed")
    ap.add_argument("--live", action="store_true", help="route REAL orders to Delta (needs creds)")
    ap.add_argument("--duration-seconds", type=int, default=0, help="0 = forever")
    ap.add_argument("--dashboard", action="store_true", help="start the web dashboard")
    ap.add_argument("--dashboard-host", default="127.0.0.1")
    ap.add_argument("--dashboard-port", type=int, default=8787)
    args = ap.parse_args()

    settings = get_settings()
    bankroll = args.bankroll if args.bankroll is not None else settings.bot.bankroll
    console = Console()
    mode = "LIVE" if args.live else "PAPER"
    console.rule(
        f"[bold cyan]aera · delta {mode}[/]  bankroll=${bankroll}"
    )

    portfolio = Portfolio(bankroll=bankroll)
    risk = RiskManager(settings.risk, portfolio)

    strategies = build_strategies(
        [s.strip() for s in args.strategies.split(",")],
        portfolio,
    )
    if not strategies:
        console.print("[red]no valid Delta strategies selected[/]")
        return

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols
        else None
    )

    # dashboard (optional)
    state = None
    dashboard_task = None
    if args.dashboard:
        from aera.dashboard import DashboardState, run_dashboard

        state = DashboardState(portfolio)
        state.paper_mode = not args.live
        dashboard_task = asyncio.create_task(
            run_dashboard(
                state,
                host=args.dashboard_host,
                port=args.dashboard_port,
            ),
            name="dashboard-server",
        )
        console.print(
            f"[bold green]dashboard ->[/] http://{args.dashboard_host}:{args.dashboard_port}"
        )

    async with DeltaClient(settings.markets.delta) as delta:
        cfg_delta = settings.markets.delta
        if args.live:
            if not delta.authenticated:
                console.print(
                    "[red]--live set but DELTA_API_KEY/DELTA_API_SECRET missing in env[/]"
                )
                if dashboard_task:
                    dashboard_task.cancel()
                return
            exchange = DeltaLiveExchange(
                delta,
                min_trade_notional_usd=cfg_delta.min_trade_notional_usd,
                max_notional_overshoot=cfg_delta.max_notional_overshoot,
            )
            lev = cfg_delta.leverage
            lev_str = f"{lev}x" if lev is not None else "(account default)"
            console.print(
                f"[bold red]LIVE MODE — real orders will be placed · leverage={lev_str}[/]"
            )
        else:
            exchange = DeltaPaperExchange(
                slippage=LinearSlippageModel(bps=settings.execution.default_slippage_bps),
                min_trade_notional_usd=cfg_delta.min_trade_notional_usd,
                max_notional_overshoot=cfg_delta.max_notional_overshoot,
                taker_fee_bps=settings.execution.taker_fee_bps,
            )

        # Greedy autopilot overlay: dynamic TP=fees+$1, trailing SL,
        # leverage selection, and fast compounding. Active only when
        # ``greedy.enabled`` in config (see config/config.yaml -> greedy).
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

        # Adaptive Brain overlay: per-strategy live edge tracker,
        # regime router (RANGE / TREND_* / HIGH_VOL / NEWS_SPIKE),
        # daily-loss circuit-breaker, gross-exposure cap. Mutes
        # losing strategies and shrinks size in bad regimes.
        brain = AdaptiveBrain(settings.brain, portfolio)
        if brain.enabled:
            console.print(
                f"[bold cyan]adaptive brain ON[/] "
                f"min_wr={settings.brain.min_win_rate:.0%} "
                f"max_streak={settings.brain.max_strategy_loss_streak} "
                f"mute={settings.brain.mute_seconds:.0f}s "
                f"daily_loss_cap={settings.brain.daily_loss_pct:.0%} "
                f"gross_exp_cap={settings.brain.max_gross_exposure_mult:g}x"
            )

        engine_kwargs = dict(
            strategies=strategies,
            executor=executor,
            delta=delta,
            symbols=symbols,
            loop_interval_ms=settings.bot.loop_interval_ms,
            use_websocket=args.websocket,
            greedy=greedy_mgr,
            brain=brain,
            invert_signals=settings.bot.invert_signals,
        )
        if settings.bot.invert_signals:
            console.print(
                "[bold yellow]signal inversion ON[/] — every strategy "
                "BUY -> SELL and SELL -> BUY (reduce-only closes unchanged). "
                "Disable with BOT_INVERT_SIGNALS=false or bot.invert_signals: false."
            )
        if state is not None:
            engine_kwargs.update(
                on_signals=state.record_signals,
                on_execution=state.record_execution,
                on_markets=state.record_markets,
            )

        engine = DeltaEngine(**engine_kwargs)
        if state is not None:
            state.bind_engine(engine)
            if brain.enabled:
                state.bind_brain(brain)

        console.print(
            f"[green]engine running[/] strategies={[s.name for s in strategies]} "
            f"ws={args.websocket} symbols={symbols or settings.markets.delta.symbols}"
        )

        engine_task = asyncio.create_task(engine.run(), name="delta-engine")
        status_task = (
            asyncio.create_task(status_render(portfolio, engine, console))
            if state is None
            else None
        )

        async def stopper():
            if args.duration_seconds > 0:
                await asyncio.sleep(args.duration_seconds)
                engine.stop()

        stop_task = asyncio.create_task(stopper())

        try:
            tasks = [engine_task]
            if dashboard_task is not None:
                tasks.append(dashboard_task)
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (KeyboardInterrupt, asyncio.CancelledError):
            engine.stop()
        finally:
            stop_task.cancel()
            if status_task is not None:
                status_task.cancel()
            if dashboard_task is not None:
                dashboard_task.cancel()
            for t in (engine_task,):
                if not t.done():
                    t.cancel()
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
