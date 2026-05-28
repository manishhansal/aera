"""Centralised configuration loader.

Layers, in order of precedence:
    1. Environment variables (BOT_*, DELTA_*)
    2. config/config.yaml
    3. Hard-coded defaults inside the pydantic models
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


class BotConfig(BaseModel):
    bankroll: float = 1.0
    paper: bool = True
    log_level: str = "INFO"
    loop_interval_ms: int = 250
    # Reverse every entry leg's direction before submission: BUY -> SELL,
    # SELL -> BUY. Useful for fading strategies whose live edge has flipped
    # sign without rewriting them. Only NON-reduce-only legs are flipped —
    # TP/SL closes and greedy flatten signals always retain their direction
    # so they still close the actual open position (not open a new one).
    # Default: False — the strategies in this codebase are tuned for
    # positive expectancy; inverting them produces -E - 2×fees per trade.
    invert_signals: bool = False


class RiskConfig(BaseModel):
    kelly_fraction: float = 0.25
    # Hard ceiling on a single trade's largest leg, as a fraction of bankroll.
    # `trade_size_fraction` (below) targets a size; this caps it.
    max_trade_fraction: float = 0.50
    # Target trade size as a fraction of bankroll. The executor scales every
    # signal so its largest leg ≈ ``trade_size_fraction * bankroll`` (capped
    # by ``max_trade_fraction``). Set to 0.0 to disable targeting and fall
    # back to the legacy "respect strategy/Kelly size, only scale down to fit
    # the cap" behaviour.
    trade_size_fraction: float = 0.50
    # Per-market cumulative exposure cap. MUST be >= max_trade_fraction
    # otherwise the per-market ceiling is tighter than the per-trade ceiling
    # and any allowed trade is rejected immediately. ``get_settings`` warns
    # at startup when this invariant is violated.
    max_market_exposure: float = 0.50
    max_drawdown: float = 0.50
    # Number of consecutive LOSING closed trades that triggers the
    # loss-streak cool-down. This is NOT a permanent halt — the bot
    # pauses for ``loss_streak_cooldown_seconds`` and then resumes
    # automatically. Set 0 to disable the cool-down entirely. For
    # scalping bots taking dozens of trades per hour, six losses in
    # a row is normal noise; the default of 15 lets the brain's
    # per-strategy mute do the precise work and only fires the
    # bot-wide cool-down on real bleeding.
    max_consecutive_losses: int = 15
    # How long the loss-streak halt pauses the bot before letting
    # new entries through again. Defaults to 5 min — long enough to
    # let the market move past whatever noise tagged us, short
    # enough that the bot recovers within a session.
    loss_streak_cooldown_seconds: float = 300.0


class ExecutionConfig(BaseModel):
    taker_fee_bps: float = 0.0
    default_slippage_bps: float = 10.0
    min_order_size_usd: float = 1.0
    order_timeout_seconds: int = 30


class StrategyConfig(BaseModel):
    enabled: bool = True
    min_edge: float = 0.005
    # arbitrary extra fields per strategy
    extra: dict = Field(default_factory=dict)


class DeltaPerpScalperConfig(StrategyConfig):
    """Tunables for the Delta Exchange perpetual mean-reversion scalper."""

    min_edge: float = 0.002
    zscore_window: int = 60
    zscore_entry: float = 2.0
    ofi_threshold: float = 0.2
    notional_usd: float = 5.0
    min_depth_contracts: float = 1.0
    rearm_distance_bps: float = 5.0
    # Take-profit / stop-loss as a fraction of entry mid (0 disables).
    # At 100x leverage a 1% mid move ≈ 100% of margin, so keep these tight.
    take_profit_pct: float = 0.0
    stop_loss_pct: float = 0.0
    # Take-profit / stop-loss as absolute USD unrealised P&L on the live
    # position (0 disables). When set with a live portfolio attached, these
    # take precedence over the percentage thresholds. Example: 5.0 / 3.0 →
    # close at +$5 of profit or -$3 of loss.
    take_profit_usd: float = 0.0
    stop_loss_usd: float = 0.0


class OrderBookSniperConfig(StrategyConfig):
    """Tunables for the L2 depth-imbalance "Order Book Sniper" (DOM scalp).

    A high-frequency micro-profit strategy. Front-runs visible bid/ask walls
    when the cumulative top-of-book depth on one side is N× the other within
    a tight price band, and the recent trade tape agrees with the same
    direction (inferred from book deltas to avoid needing a separate trades
    feed). Holds for at most ``max_hold_seconds`` and exits early on a
    spoofing event (the entry-side wall disappears within
    ``spoof_persist_seconds`` of entry).
    """

    enabled: bool = False                # opt-in: paper-trade alongside the mean-reversion scalper
    min_edge: float = 0.0005             # 5 bps — micro-profit target
    # Cumulative size on the favoured side must be >= imbalance_ratio × the
    # other side within ``imbalance_band_bps`` of mid.
    imbalance_ratio: float = 3.0
    imbalance_band_bps: float = 10.0     # ±10 bps from mid = 0.1% band per spec
    imbalance_max_levels: int = 10       # top-N levels considered for the band
    # Minimum tape activity in the same direction over the last
    # ``tape_window_seconds`` seconds (taker buys for a long entry, taker
    # sells for a short entry). Inferred from successive book snapshots.
    tape_min_count: int = 3
    tape_window_seconds: float = 2.0
    # Reference USD notional emitted on each fire. The executor's
    # ``trade_size_fraction`` will normally replace this, identical to the
    # mean-reversion scalper.
    notional_usd: float = 1000.0
    # Hold time / exits — TIGHT bands per spec (0.05% TP, 0.03% SL).
    take_profit_pct: float = 0.0005
    stop_loss_pct: float = 0.0003
    take_profit_usd: float = 0.0
    stop_loss_usd: float = 0.0
    max_hold_seconds: float = 10.0       # 1–10s hold window
    # Spoofing defense: if the entry-side wall shrinks by more than
    # ``spoof_vanish_ratio`` within ``spoof_persist_seconds`` of entry and
    # the wall was at least ``spoof_min_wall_contracts`` in size, market-
    # exit regardless of P&L.
    spoof_min_wall_contracts: float = 0.0
    spoof_persist_seconds: float = 1.0
    spoof_vanish_ratio: float = 0.5
    # Tick-aware limit price: order at best_bid + ``entry_tick_offset`` ticks
    # (or best_ask − N ticks for a short). Use the market's minimum_tick.
    entry_tick_offset: int = 1
    # Cheap rearm so a sustained wall doesn't fire on every tick — must move
    # ``rearm_distance_bps`` from the last firing mid before re-engaging.
    rearm_distance_bps: float = 3.0


class TickReversalScalpConfig(StrategyConfig):
    """Tunables for the Tick Reversal Scalp (3-tick exhaustion fade).

    Detects N consecutive same-direction mid ticks with shrinking per-tick
    "eaten" liquidity (momentum exhaustion), then fires a fade in the
    opposite direction. Includes safety filters for wide spreads, volume
    spikes, and news-style price jumps. Holds for at most
    ``max_hold_seconds`` (default 30 s) per the spec.

    "Tick" here is defined as any scan where the mid moves. Direction =
    sign of mid change. "Size" = the size eaten from the leading side of
    the book (inferred from successive top-of-book snapshots — no
    separate trades-channel subscription required).
    """

    enabled: bool = False                    # opt-in like the sniper
    min_edge: float = 0.0004                 # 4 bps target — spec micro-profit
    # Streak detection
    min_streak: int = 5                      # 5+ consecutive same-direction ticks
    max_buffer_ticks: int = 200              # rolling tick history depth
    # Size decay: last-tick size must be <= (1 − threshold) × first-tick
    # size of the streak. 0.20 = "20% decay across the streak". The spec
    # text "size decay > 20% per tick drop" reads ambiguously; we use the
    # overall decay across the streak as the practical interpretation.
    size_decay_threshold: float = 0.20
    # S/R proxy: current mid must be within ``sr_band_bps`` of the
    # ``sr_lookback_ticks`` extreme in the streak direction (= the local
    # low for a long entry, local high for a short entry). 0 disables.
    sr_band_bps: float = 5.0
    sr_lookback_ticks: int = 50
    # Bid depth must trend up over the streak for a long entry (mirror
    # for shorts on the ask side). Compared at the start of the streak vs
    # the most recent tick. Set ``require_depth_trend`` = false to skip.
    require_depth_trend: bool = True
    # Filters: skip an otherwise-valid entry when these tripping conditions
    # apply. Set each multiplier / threshold to 0 to disable that filter.
    max_spread_multiple: float = 3.0         # spread > N × EMA spread → skip
    spread_ema_alpha: float = 0.05           # smoothing for the spread EMA
    volume_spike_multiple: float = 5.0       # short-window rate / long-window rate
    volume_short_window_seconds: float = 5.0
    volume_long_window_seconds: float = 60.0
    news_lookback_seconds: float = 60.0      # ignored when news_max_tick_bps=0
    news_max_tick_bps: float = 50.0          # any single tick > N bps → news spike
    # Notional / sizing — the executor's trade_size_fraction typically
    # overrides this number; it's just the reference proportion.
    notional_usd: float = 1000.0
    # Limit-order entry at the mid (per spec). Set ``entry_offset_bps`` > 0
    # to bias the entry away from mid (positive bps = more passive — bids
    # below mid, asks above mid); negative biases more aggressive. The IOC
    # behavior on live exchanges drops unfilled orders inside 300 ms.
    entry_offset_bps: float = 0.0
    # Exits per spec — tight TP/SL plus a hard hold timeout.
    take_profit_pct: float = 0.0004          # +0.04%
    stop_loss_pct: float = 0.00025           # -0.025%
    take_profit_usd: float = 0.0
    stop_loss_usd: float = 0.0
    max_hold_seconds: float = 30.0           # spec: 5–30 s holds
    # Rearm debounce — don't refire on the same symbol until mid has moved
    # at least this far from the previous firing mid. Cheap.
    rearm_distance_bps: float = 3.0


class BidAskSpreadFadeConfig(StrategyConfig):
    """Tunables for the Bid-Ask Spread Fade (Market Making Lite) strategy.

    A symmetric two-sided market-making strategy: post limit orders on
    both sides of the spread, earn the spread as carry, net-flat
    inventory. Designed for transient illiquidity spikes on liquid
    Delta perpetuals (BTC/ETH-PERP) where the spread widens enough that
    a fractional capture clears two maker fees and still yields net
    positive bps per round trip.
    """

    enabled: bool = False                       # opt-in; ships with the spec defaults
    min_edge: float = 0.0004                    # 4 bps — micro-profit target
    # Spread floor: only quote when current spread/mid >= this fraction.
    # Tighter spreads can't clear two maker fees, so we sit them out.
    min_spread_pct: float = 0.0003              # 0.03% per spec
    # Fraction of the spread we attempt to capture. The quotes sit at
    # ``mid ± (spread × capture / 2)``; 0.60 captures the inner 60% and
    # leaves a 20% buffer on each side.
    capture_target: float = 0.60
    # Reference notional per side, in USD. Executor's
    # ``trade_size_fraction`` may rescale this — set the global
    # trade_size_fraction = 0 (legacy mode) for verbatim $5 quotes.
    quote_size_usd: float = 5.0
    # Hard inventory cap (absolute notional). Past this, the offending
    # side stops quoting until inventory walks back inside.
    max_inventory_usd: float = 15.0
    # Skew threshold (< cap). Past this, quotes shift toward the
    # offload side so the maker becomes biased to flatten.
    inventory_skew_threshold_usd: float = 10.0
    inventory_skew_ticks: int = 1
    # Re-quote cadence in milliseconds. The Delta engine ticks faster
    # than this, so the gate sets the effective MM rhythm.
    refresh_rate_ms: float = 500.0
    # Volatility kill switch. If mid moves > ``kill_move_pct`` (peak-to-
    # trough) inside ``kill_window_seconds``, suspend quoting until
    # the move drops out of the window.
    kill_move_pct: float = 0.0008               # 0.08% per spec
    kill_window_seconds: float = 5.0
    # Net-edge math. Delta's maker fee is ~2 bps; we require the
    # projected capture minus 2 × maker_fee to exceed
    # ``min_net_edge_bps`` before a cycle is allowed.
    maker_fee_bps: float = 2.0
    min_net_edge_bps: float = 4.0
    # If set, stamp this leverage on every emitted leg instead of
    # reading from market metadata. Spec asks for cash-style sizing
    # (1.0). ``null`` follows the venue leverage.
    leverage_override: Optional[float] = 1.0


class FlowScalpConfig(StrategyConfig):
    """Tunables for the Tape Reading Momentum (Flow Scalp) strategy.

    Detects a single taker trade ≥ ``whale_multiple`` × the rolling
    ``avg_window``-trade mean size, waits for ``confirm_count`` more
    same-direction trades ≥ ``confirm_multiple`` × avg within
    ``confirm_window_seconds``, then front-runs the continued flow.

    Trade data is sourced from book-delta inference by default (no
    separate trades-channel subscription required). When a real
    trades feed is wired in, set ``auto_infer_from_book: false`` and
    push prints via :meth:`FlowScalp.record_trade`.
    """

    enabled: bool = False                    # opt-in like the other HFT scalpers
    min_edge: float = 0.0008                 # 8 bps target — spec micro-profit
    # Whale detection
    whale_multiple: float = 5.0              # single print ≥ 5 × avg trade size
    confirm_multiple: float = 2.0            # confirmation print ≥ 2 × avg
    confirm_count: int = 1                   # one extra same-direction print
    confirm_window_seconds: float = 3.0      # window for the confirmation
    avg_window: int = 100                    # rolling trades used for avg size
    tape_max_trades: int = 500               # buffer cap (≥ 2 × avg_window)
    # Sizing — executor's trade_size_fraction typically overrides.
    notional_usd: float = 1000.0
    # Exits — tight HFT bands per spec.
    take_profit_pct: float = 0.0008          # +0.08% hard TP
    stop_loss_pct: float = 0.0004            # −0.04% hard SL
    trailing_stop_pct: float = 0.0002        # 0.02% trail from best mid (when in profit)
    take_profit_usd: float = 0.0
    stop_loss_usd: float = 0.0
    max_hold_seconds: float = 60.0           # spec: 10–60s holds
    rearm_distance_bps: float = 5.0          # debounce between fires
    # Spec asks for 5× — moderate, sized for the tight TP/SL bands.
    # Set to null to inherit the venue's account leverage instead.
    leverage_override: Optional[float] = 5.0
    # Set to false when subscribing to a real trades feed and calling
    # FlowScalp.record_trade externally. Default keeps the strategy
    # self-contained (book-delta inference) like the other scalpers.
    auto_infer_from_book: bool = True


class MicroVWAPSniperConfig(StrategyConfig):
    """Tunables for the Micro VWAP Reversion Sniper.

    Mid-frequency mean-reversion strategy that fades short-term price
    deviations from a 1-minute rolling micro-VWAP when volume drops
    off (= exhausted aggressors). Targets the VWAP value at time of
    entry as a static snapshot so the bar doesn't drift with the very
    mean we're trying to revert to.
    """

    enabled: bool = False                       # opt-in like the other HFT scalpers
    min_edge: float = 0.0005                    # 5 bps target — micro-profit
    # VWAP window. Spec: 60 s = 1-minute rolling.
    vwap_window_seconds: float = 60.0
    # Deviation trigger: |mid - vwap| / vwap must exceed this fraction.
    # Spec: 0.0012 (0.12%).
    deviation_pct: float = 0.0012
    # Volume drop-off: short-window per-second volume must be < ratio_max
    # of the long-window per-second volume baseline. Spec: 10 s vs 5 min,
    # max 0.70 (= current 10s < 70% of 5-min average).
    volume_short_seconds: float = 10.0
    volume_long_seconds: float = 300.0
    volume_ratio_max: float = 0.70
    # Reference notional emitted per fire — executor's
    # trade_size_fraction typically overrides this.
    notional_usd: float = 1000.0
    # Take-profit floor (safety, only used when VWAP target unavailable).
    # Spec primary TP is VWAP-at-entry snapshot (handled internally).
    # Set tp_extra_bps > 0 to stretch the target past VWAP by N bps in
    # the favourable direction (spec optional 3 bps secondary).
    take_profit_pct: float = 0.0007             # 0.07% safety floor
    tp_extra_bps: float = 0.0                   # 0 = exit at VWAP, >0 = stretch
    # Hard % stop relative to entry mid. Spec: 0.0005 (0.05%).
    stop_loss_pct: float = 0.0005
    # USD-PnL exits — take precedence over % when both > 0 and a
    # portfolio is wired in (matches the other strategies' contract).
    take_profit_usd: float = 0.0
    stop_loss_usd: float = 0.0
    # Hold timeout (spec: 15–90 s).
    max_hold_seconds: float = 90.0
    # Spread guard: skip when spread / mid exceeds this fraction.
    # Spec: 0.0005 (0.05%). 0 disables.
    max_spread_pct: float = 0.0005
    # Skip entries during the first N seconds of every wall-clock hour.
    # Spec: 300 s (the first 5 minutes — the VWAP buffer drains across
    # the boundary and produces false deviations). 0 disables.
    hour_skip_seconds: float = 300.0
    # Leg leverage. Spec asks for 5x. null = inherit venue leverage.
    leverage_override: Optional[float] = 5.0
    # Cheap rearm debouncer — don't refire on the same symbol until
    # mid has moved at least this many bps from the previous firing.
    rearm_distance_bps: float = 3.0


class StopHuntReversalConfig(StrategyConfig):
    """Tunables for the Stop Hunt / Liquidity Grab Reversal strategy.

    Detects engineered stop-hunt wicks — sudden 0.15%+ spikes below a
    recent swing low (bullish sweep) or above a recent swing high
    (bearish sweep) that close back inside the level on the same 1 s
    candle. Fades the sweep, riding the snap-back with a partial TP at
    +0.10% (60% of size), final TP at +0.20%, and a wick-anchored hard
    stop at the wick low − 0.08% (mirror for shorts). Spec: BTC-PERP,
    ETH-PERP; low-liquidity windows preferred.
    """

    enabled: bool = False                    # opt-in like the other HFT scalpers
    min_edge: float = 0.0015                 # 15 bps target — primary spec TP
    # ---- bar aggregation -----------------------------------------
    bar_seconds: float = 1.0                 # spec: 1 s candles
    max_bars: int = 300                      # bounded memory across many symbols
    # ---- swing pivot detection -----------------------------------
    swing_lookback_bars: int = 60            # ~1 minute on 1 s bars (spec ref chart)
    swing_pivot_strength: int = 2            # 5-bar fractal (N=2 each side)
    swing_count: int = 3                     # spec: last 3 swing highs / lows
    # ---- sweep gates ---------------------------------------------
    wick_size_pct: float = 0.0015            # spec: 0.15% min wick depth past level
    body_ratio_max: float = 0.30             # spec: body < 30% of total range
    recovery_seconds: float = 3.0            # spec: price recovers in < 3 s
    volume_multiple: float = 1.5             # spec: "volume spike on wick candle"
    volume_lookback_bars: int = 30           # baseline window for volume comparison
    # ---- delta confirmation --------------------------------------
    # For a bearish sweep the spec wants "delta flips red" (= net selling
    # on the wick close). delta_flip_threshold sets the magnitude required;
    # bullish sweeps mirror (delta must NOT be deeply red). Set
    # require_delta_confirmation = false to skip the gate entirely
    # (useful when running off a noisy inference path).
    require_delta_confirmation: bool = True
    delta_flip_threshold: float = 0.0001
    # ---- exits ---------------------------------------------------
    take_profit_pct: float = 0.0020          # spec: +0.20% final TP
    tp1_pct: float = 0.0010                  # spec: +0.10% partial TP
    tp1_fraction: float = 0.60               # spec: close 60% of size at TP1
    stop_extra_pct: float = 0.0008           # spec: wick low − 0.08% for the stop
    stop_loss_pct: float = 0.0               # safety fallback (entry-mid %); 0 disabled
    take_profit_usd: float = 0.0
    stop_loss_usd: float = 0.0
    max_hold_seconds: float = 60.0           # hard exit if neither TP/SL fires
    # ---- sizing / leg metadata -----------------------------------
    leverage_override: Optional[float] = 5.0  # spec asks for 5× max
    notional_usd: float = 1000.0
    rearm_distance_bps: float = 5.0           # debounce between fires


class StrategiesConfig(BaseModel):
    delta_perp_scalper: DeltaPerpScalperConfig = DeltaPerpScalperConfig()
    order_book_sniper: OrderBookSniperConfig = OrderBookSniperConfig()
    tick_reversal_scalp: TickReversalScalpConfig = TickReversalScalpConfig()
    bid_ask_spread_fade: BidAskSpreadFadeConfig = BidAskSpreadFadeConfig()
    flow_scalp: FlowScalpConfig = FlowScalpConfig()
    micro_vwap_sniper: MicroVWAPSniperConfig = MicroVWAPSniperConfig()
    stop_hunt_reversal: StopHuntReversalConfig = StopHuntReversalConfig()


class DeltaConfig(BaseModel):
    """Delta Exchange (https://www.delta.exchange) REST + websocket config.

    Two regional deployments:
        * Global: ``https://api.delta.exchange`` + ``wss://socket.delta.exchange``
        * India:  ``https://api.india.delta.exchange`` + ``wss://socket.india.delta.exchange``

    Set ``base_url`` / ``ws_url`` accordingly. Authentication uses HMAC-SHA256
    over (method + timestamp + path + query + body) signed with ``api_secret``.
    Credentials are only needed for live trading and authenticated reads
    (balances, positions, fills) — public market data is unauthenticated.
    """

    base_url: str = "https://api.delta.exchange"
    ws_url: str = "wss://socket.delta.exchange"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    user_agent: str = "aera-trading-bot/0.1"
    poll_interval_ms: int = 500
    websocket: bool = True
    # Default symbols to watch when no explicit list is given. Delta global
    # uses USDT-margined naming (BTCUSDT, ETHUSDT, ...); the India deployment
    # uses BTCUSD/ETHUSD inverse perpetuals — override via config or CLI.
    symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    # contract types to keep when listing products: any of perpetual_futures /
    # futures / call_options / put_options / interest_rate_swaps / spot
    contract_types: List[str] = Field(default_factory=lambda: ["perpetual_futures"])

    # ---- order sizing & leverage ----
    # If set, the engine calls Delta's `change-leverage` endpoint at startup
    # for every configured symbol, so the assumed margin math matches what
    # the account actually applies. ``None`` = leave the account setting
    # alone (Delta's default is the product's max leverage, typically 100x).
    leverage: Optional[int] = None
    # Reject a trade if its rounded-up minimum contract count would result
    # in a notional this many times larger than the strategy's intent.
    # Prevents the bot from silently 15×-ing a $5 leg into a $76 leg just
    # because 1 BTCUSDT contract is ~$76. Default 1.5 = max 50% overshoot.
    max_notional_overshoot: float = 1.5
    # Hard floor on trade notional in USD. Trades below this are skipped
    # before the contract-count step. Default $1 matches Delta's typical
    # min-order notional.
    min_trade_notional_usd: float = 1.0


class MarketsConfig(BaseModel):
    delta: DeltaConfig = DeltaConfig()


class SimulationConfig(BaseModel):
    monte_carlo_runs: int = 1000
    default_edge: float = 0.0004
    default_win_rate: float = 0.535
    default_payoff_ratio: float = 1.0


class GreedyConfig(BaseModel):
    """Greedy autopilot — dynamic TP/SL + leverage selection + fast compounding.

    When ``enabled``, an overlay sitting between the strategies and the
    executor takes ownership of three things on every trade:

    1. **Take-profit.** Computed per-position as
       ``round_trip_fees_usd × fee_pad_multiple + min_profit_usd``. With
       ``min_profit_usd = 1.0`` and a 5 bps taker fee on a $500 notional
       trade, the TP target is ``$0.50 + $1.00 = $1.50``. The target
       *rolls forward* whenever profit exceeds it — every locked-in dollar
       extends the target by ``extend_tp_step_usd`` so a strong run keeps
       running instead of capping out at the first hit.

    2. **Stop-loss.** Starts at ``-initial_sl_usd`` (a small cushion that
       absorbs round-trip fees plus a buck of slippage) and *ratchets up*
       once unrealised P&L crosses ``lock_in_trigger_ratio × tp_target``.
       From that point the SL trails the running best PnL by
       ``trailing_giveback_usd`` — so the worst-case give-back is small
       even on extended runs. SL never moves down.

    3. **Leverage.** Picked per entry from the live win streak (greedy):
       starts at ``min_leverage`` and climbs by ``leverage_step`` per
       consecutive win, capped at ``max_leverage`` (and the venue's max if
       ``respect_venue_cap``). On a loss streak the leverage is divided
       by ``(1 + consecutive_losses)`` so a bad run de-risks fast.

    Compounding is implicit: the executor always sizes against the live
    ``bankroll × leverage`` buying power, and ``compound_fraction``
    replaces the risk module's ``trade_size_fraction`` when greedy is on
    (defaulting to 0.9 so almost the entire bankroll is deployed on
    every fresh entry). Wins go back into the bankroll the instant they
    are realised, so the next trade is sized against the new wealth.

    Strategies' own TP/SL knobs are *not* disabled — they form an
    additional safety net. Set them to 0 in YAML to make greedy the
    sole exit authority.
    """

    enabled: bool = False                    # opt-in
    # ---- dynamic TP -----------------------------------------------
    # Floor profit (USD) on top of fees. Spec: "fees + $1".
    min_profit_usd: float = 1.0
    # Multiplier on the fee estimate before adding min_profit_usd.
    # 1.0 = exactly round-trip fees, 1.5 = adds 50% extra cushion.
    fee_pad_multiple: float = 1.0
    # After hitting an initial TP, extend the target by this much so
    # winners keep running. 0 disables (fixed TP, exits on first hit).
    extend_tp_step_usd: float = 1.0
    # ---- dynamic SL -----------------------------------------------
    # Initial cushion before the trailing ratchet kicks in (USD).
    initial_sl_usd: float = 1.5
    # Begin ratcheting the SL up once PnL_usd >= ratio × tp_target.
    # 0.5 = "start locking in profit at half-way to the TP target".
    lock_in_trigger_ratio: float = 0.5
    # Once ratcheting, the SL trails the running best PnL by this much.
    # 0.5 = "give back at most $0.50 from the peak".
    trailing_giveback_usd: float = 0.5
    # Hard maximum hold (sec). 0 disables; the manager only exits on
    # the TP/SL ladder.
    max_hold_seconds: float = 120.0
    # ---- greedy leverage ------------------------------------------
    min_leverage: float = 5.0
    max_leverage: float = 100.0
    # Each consecutive win bumps the leverage by this step. Set to 0
    # to lock leverage at min_leverage regardless of streak.
    leverage_step: float = 5.0
    # When True, cap the chosen leverage at the market's
    # metadata["leverage"] (the venue's account leverage). When False,
    # the overlay can ask for leverage above the account default — the
    # executor will still validate against margin / order requirements.
    respect_venue_cap: bool = True
    # ---- compounding ----------------------------------------------
    # When greedy is enabled, the executor uses this in place of
    # ``risk.trade_size_fraction`` so the bot deploys almost all of its
    # buying power on every fresh entry. Capped at 0.99 internally so a
    # rounding error doesn't blow the bankroll into negative.
    compound_fraction: float = 0.90
    # Hard ceiling on per-trade notional (USD). 0 disables. With small
    # bankrolls this is irrelevant; with large ones (e.g. 100k +) it's
    # the difference between sensible $1 scalps and trades whose fees
    # alone are $500. The greedy.min_profit_usd / initial_sl_usd
    # thresholds are absolute USD amounts — they only make sense when
    # round-trip fees on the trade are also bounded.
    max_notional_usd: float = 0.0
    # ---- fee estimation ------------------------------------------
    # If > 0, use this taker fee in bps for fee-based TP math instead
    # of execution.taker_fee_bps. Useful when the live fee differs from
    # the paper-trading config (e.g. maker rebate negotiated).
    fee_override_bps: float = 0.0


class BrainConfig(BaseModel):
    """Adaptive Brain — live edge tracker, regime router, and circuit-breakers.

    The brain sits between the strategies and the executor (after greedy)
    and applies four classes of veto / shrink rule to fresh entries:

    1. **Performance gate**: if a strategy's rolling win-rate or
       expectancy is below the floor (over ``min_trades_for_eval``
       samples), pause its entries for ``mute_seconds`` and re-engage
       at ``probation_size_mult`` until it earns its way back to 1.0.
    2. **Regime gate**: each strategy declares which regimes it likes
       (mean-reversion in RANGE, momentum in TREND_*). Entries in the
       wrong regime are dropped. NEWS_SPIKE vetoes everything.
    3. **Daily-loss circuit-breaker**: if 24h rolling realised PnL
       drops below ``-daily_loss_pct × bankroll`` no new entries
       fire for the rest of the window (closes still flow).
    4. **Correlation cap**: total gross LONG (or SHORT) notional
       across all symbols capped at
       ``max_gross_exposure_mult × settled_wealth``.

    All gates are no-op when ``enabled = false``. Reduce-only legs
    (TP/SL closes, greedy flattens) always flow regardless of state.
    """

    enabled: bool = True
    # ---- per-strategy performance gate ----------------------------
    # Number of closed round-trips a strategy needs before its
    # win-rate / expectancy is even evaluated. Below this, the
    # strategy fires at full size — we don't punish brand-new
    # strategies for a single bad trade.
    min_trades_for_eval: int = 10
    # Rolling window depth of recent PnLs kept per strategy. Older
    # trades drop off the back. Larger = slower to react to a regime
    # change but more statistically stable.
    perf_window: int = 30
    # Strategies are muted when their rolling win-rate drops below
    # this OR expectancy drops below ``min_expectancy_usd``.
    min_win_rate: float = 0.40
    min_expectancy_usd: float = 0.0
    # Hard stop on consecutive losses per strategy. Trips an
    # immediate mute even before the rolling stats are evaluated.
    # Lower = kill bleeders faster. 2 means a strategy that loses
    # two trades in a row is paused for ``mute_seconds``; the
    # post-loss cool-down (``post_loss_cooldown_seconds``) already
    # prevents same-symbol revenge trades, so this only fires when
    # the same strategy loses on TWO DIFFERENT symbols in a row.
    max_strategy_loss_streak: int = 2
    # Mute duration after a gate trips (seconds). After expiry the
    # strategy re-engages on probation.
    mute_seconds: float = 600.0
    # Per-(strategy, symbol) cool-down after a losing close. Refuses
    # new entries on the same (strategy, symbol) pair for this many
    # seconds. Stops the bot from re-entering the same losing pattern
    # right after a stop-out (the primary cause of "lost six in a
    # row in 6 minutes" runs in early sessions). 0 disables.
    post_loss_cooldown_seconds: float = 60.0
    # Probation: number of closed trades the strategy must complete
    # at ``probation_size_mult`` size before graduating back to 1.0.
    # Graduation requires non-negative expectancy across the window.
    probation_trades: int = 5
    probation_size_mult: float = 0.5
    # ---- regime detector parameters --------------------------------
    regime_short_window: int = 30
    regime_long_window: int = 300
    regime_trend_threshold: float = 0.60
    regime_high_vol_ratio: float = 2.5
    regime_news_tick_bps: float = 35.0
    # ---- regime size shrinking ------------------------------------
    # If True, wrong-regime signals (e.g. mean-reversion in TREND_UP)
    # FIRE AT REDUCED SIZE (``wrong_regime_size_mult``) instead of
    # being hard-vetoed. Lets the perf gate decide based on actual
    # PnL rather than a static "this strategy doesn't belong here"
    # prior. NEWS_SPIKE remains a hard veto regardless.
    regime_soft_veto: bool = True
    wrong_regime_size_mult: float = 0.5
    # Multiplier applied to entry size in HIGH_VOL regime for
    # vol-sensitive strategies (everyone except possibly arb).
    high_vol_size_mult: float = 0.5
    # ---- daily loss circuit-breaker -------------------------------
    # If the rolling 24h realised PnL drops below this fraction of
    # the running peak bankroll, the brain refuses new entries until
    # the window rolls off. 0 disables.
    daily_loss_pct: float = 0.10
    daily_window_seconds: float = 86400.0  # 24h
    # ---- correlation / gross exposure cap -------------------------
    # Multiplier on ``bankroll × leverage`` (= buying power) that
    # caps the total gross long (or short) exposure across all
    # symbols. With default 2.0 and a $100 bankroll on 25× perps,
    # the per-side cap is $100 × 25 × 2.0 = $5,000 of notional —
    # ≈ 13 concurrent $375 scalps before the brain says "enough".
    # 0 disables — back to the per-market exposure cap in RiskConfig.
    max_gross_exposure_mult: float = 2.0


class Settings(BaseModel):
    bot: BotConfig = BotConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    strategies: StrategiesConfig = StrategiesConfig()
    markets: MarketsConfig = MarketsConfig()
    simulation: SimulationConfig = SimulationConfig()
    greedy: GreedyConfig = GreedyConfig()
    brain: BrainConfig = BrainConfig()


def _apply_env_overrides(data: dict) -> dict:
    """Layer env vars on top of YAML."""
    env_map = {
        ("bot", "bankroll"): ("BOT_BANKROLL", float),
        ("bot", "paper"): ("BOT_PAPER", lambda v: v.lower() == "true"),
        ("bot", "log_level"): ("BOT_LOG_LEVEL", str),
        ("bot", "invert_signals"): ("BOT_INVERT_SIGNALS", lambda v: v.lower() == "true"),
        ("risk", "kelly_fraction"): ("BOT_KELLY_FRACTION", float),
        ("risk", "max_trade_fraction"): ("BOT_MAX_TRADE_FRACTION", float),
        ("risk", "trade_size_fraction"): ("BOT_TRADE_SIZE_FRACTION", float),
        ("execution", "taker_fee_bps"): ("BOT_TAKER_FEE_BPS", float),
        ("markets", "delta", "api_key"): ("DELTA_API_KEY", str),
        ("markets", "delta", "api_secret"): ("DELTA_API_SECRET", str),
        ("markets", "delta", "base_url"): ("DELTA_BASE_URL", str),
        ("markets", "delta", "ws_url"): ("DELTA_WS_URL", str),
        ("markets", "delta", "leverage"): ("DELTA_LEVERAGE", int),
        ("greedy", "enabled"): ("BOT_GREEDY", lambda v: v.lower() == "true"),
        ("greedy", "min_profit_usd"): ("BOT_GREEDY_MIN_PROFIT_USD", float),
        ("greedy", "min_leverage"): ("BOT_GREEDY_MIN_LEVERAGE", float),
        ("greedy", "max_leverage"): ("BOT_GREEDY_MAX_LEVERAGE", float),
        ("greedy", "compound_fraction"): ("BOT_GREEDY_COMPOUND_FRACTION", float),
        ("brain", "enabled"): ("BOT_BRAIN", lambda v: v.lower() == "true"),
        ("brain", "min_win_rate"): ("BOT_BRAIN_MIN_WIN_RATE", float),
        ("brain", "min_expectancy_usd"): ("BOT_BRAIN_MIN_EXPECTANCY_USD", float),
        ("brain", "daily_loss_pct"): ("BOT_BRAIN_DAILY_LOSS_PCT", float),
        ("brain", "max_gross_exposure_mult"): ("BOT_BRAIN_MAX_GROSS_EXPOSURE_MULT", float),
        ("brain", "mute_seconds"): ("BOT_BRAIN_MUTE_SECONDS", float),
    }
    for path, (env_name, cast) in env_map.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        node = data
        for key in path[:-1]:
            node = node.setdefault(key, {})
        try:
            node[path[-1]] = cast(raw)
        except Exception:
            pass
    return data


def _validate_risk_invariants(settings: "Settings") -> None:
    """Emit startup warnings for config combinations that silently break trading.

    These aren't hard errors because users sometimes intentionally pick odd
    ratios for backtests, but a single ``log.warning`` here surfaces issues
    that would otherwise present as "every signal is rejected" in the wild.
    """
    import logging

    log = logging.getLogger("aera.settings")
    risk = settings.risk
    if risk.max_market_exposure < risk.max_trade_fraction:
        log.warning(
            "risk.max_market_exposure (%.2f) < risk.max_trade_fraction (%.2f): "
            "per-market cap is tighter than per-trade cap, so single-market "
            "trades will be rejected immediately by the risk vet. "
            "Raise max_market_exposure to at least %.2f.",
            risk.max_market_exposure,
            risk.max_trade_fraction,
            risk.max_trade_fraction,
        )
    if risk.max_market_exposure < risk.trade_size_fraction:
        log.warning(
            "risk.max_market_exposure (%.2f) < risk.trade_size_fraction (%.2f): "
            "trades scaled to the configured target will exceed the per-market "
            "cap and be rejected. Raise max_market_exposure to at least %.2f.",
            risk.max_market_exposure,
            risk.trade_size_fraction,
            risk.trade_size_fraction,
        )
    if risk.trade_size_fraction > risk.max_trade_fraction:
        log.warning(
            "risk.trade_size_fraction (%.2f) > risk.max_trade_fraction (%.2f): "
            "the target will be silently clamped down to the cap. Either lower "
            "trade_size_fraction or raise max_trade_fraction.",
            risk.trade_size_fraction,
            risk.max_trade_fraction,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(override=False)
    raw: dict = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    raw = _apply_env_overrides(raw)
    settings = Settings.model_validate(raw)
    _validate_risk_invariants(settings)
    return settings
