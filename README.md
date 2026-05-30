# aera

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-397%20passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-personal--use-lightgrey.svg)](#disclaimers)

**Repository:** <https://github.com/manishhansal/aera>

> **An autonomous Delta Exchange perpetual-futures trading bot.**
> Mean-reversion + order-flow + sweep + market-making scalping on
> USD-quoted perps, with an **adaptive brain** that measures each
> strategy's live edge, mutes the losing ones, vetoes signals in the
> wrong regime, applies post-loss cool-downs to prevent revenge
> trading, and enforces a daily-loss circuit breaker — wrapped in a
> greedy autopilot that compounds the survivors aggressively. Plus
> an **offline backtest + ML training pipeline** that powers a
> backtest-trained adaptive strategy (`money_printer`) which only
> fires during historically profitable hours when a gradient-boosted
> classifier agrees. Leverage-aware sizing, hard risk caps,
> absolute-USD take-profit / stop-loss, paper-trading, and a live
> web dashboard.

```
 Delta Exchange (REST + WSS)                                 USDC / margin
        |                                                          ^
        v                                                          |
+----------------+   markets[]   +----------------+   signals[]   +-----------+   filtered    +----------+   orders
|  DeltaClient   |-------------->|   Strategies   |-------------->|  Greedy   |-------------->|  Brain   |---------+
|  + Websocket   |               |  (8 strats inc.|               |  TP/SL/   |               |  regime  |         |
+----------------+               |  money_printer)|               |  leverage |               |  + edge  |         |
                                 +----------------+               +-----------+               +----------+         |
                                          ^                                                        |               |
                  hour_maps.json +        |                                                        v               |
                  model.joblib   ────────-+                                                  +----------+           |
                  (from offline                                                              | Executor |-----------+
                  backtest+ML pipeline)                                                      +----------+
                                                                                                   |   ^
                                                                                             fills v   | bankroll
                                                                                             +-----------+
                                                                                             | Portfolio |
                                                                                             | + Risk    |
                                                                                             +-----------+
                                                                                                   ^
                                                                                                   |    paper /
                                                                                                   +-- live -----+
```

---

## Table of contents

1. [TL;DR](#tldr)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Strategy: Delta perpetual scalper](#strategy-delta-perpetual-scalper)
5. [Strategy: Order Book Sniper (DOM scalp)](#strategy-order-book-sniper-dom-scalp)
6. [Strategy: Tick Reversal Scalp](#strategy-tick-reversal-scalp)
7. [Strategy: Bid-Ask Spread Fade (Market Making Lite)](#strategy-bid-ask-spread-fade-market-making-lite)
8. [Strategy: Tape Reading Momentum (Flow Scalp)](#strategy-tape-reading-momentum-flow-scalp)
9. [Strategy: Micro VWAP Reversion Sniper](#strategy-micro-vwap-reversion-sniper)
10. [Strategy: Stop Hunt / Liquidity Grab Reversal](#strategy-stop-hunt--liquidity-grab-reversal)
11. [Strategy: Money Printer (backtest-trained adaptive)](#strategy-money-printer-backtest-trained-adaptive)
12. [Take-profit / stop-loss](#take-profit--stop-loss)
13. [Greedy autopilot](#greedy-autopilot)
14. [Adaptive brain](#adaptive-brain)
15. [Risk management](#risk-management)
16. [Offline backtest + ML pipeline](#offline-backtest--ml-pipeline)
17. [Configuration](#configuration)
18. [Installation](#installation)
19. [CLI reference](#cli-reference)
20. [Live web dashboard](#live-web-dashboard)
21. [Going live](#going-live)
22. [Testing](#testing)
23. [Disclaimers](#disclaimers)

---

## TL;DR

```bash
git clone https://github.com/manishhansal/aera.git && cd aera
python -m venv .venv && source .venv/Scripts/activate     # Windows git-bash
pip install -r requirements.txt

# 1. Read-only Delta scan (no credentials needed)
python -m scripts.scan_delta

# 2. Paper-trade Delta perps from the terminal
python -m scripts.run_delta --bankroll 27

# 3. Paper-trade Delta perps + live web dashboard
python -m scripts.run_delta --bankroll 27 --dashboard

# 4. Or run the dashboard supervisor directly
python -m scripts.run_dashboard --bankroll 27

# 5. Bankroll-growth Monte Carlo (pure math, no internet)
python -m scripts.simulate_growth

# 6. Offline backtest + ML training pipeline (powers money_printer)
python -m scripts.fetch_history   --symbols BTCUSD,ETHUSD,SOLUSD \
                                  --resolutions 1m,5m --days 90
python -m scripts.sweep_backtest  --symbols BTCUSD,ETHUSD,SOLUSD \
                                  --resolutions 1m,5m --leverages 5,10,25
python -m scripts.train_money_printer       # single GBT classifier (always)
python -m scripts.train_ensemble            # per-symbol ensemble (optional)
python -m scripts.train_rl_agent --symbol BTCUSD --resolution 5m  # DQN (optional)
# pip install -r requirements-ml-extras.txt   # only needed for the transformer
# python -m scripts.train_sequence_model      # transformer encoder (optional)
python -m scripts.run_delta --bankroll 27 --strategies money_printer --dashboard

# 7. Tests
python -m pytest -q
```

For live trading, drop your Delta credentials into `.env` (`DELTA_API_KEY` /
`DELTA_API_SECRET`) and pass `--live` to the run scripts. **Paper-trade first.**

> **India-region accounts:** if you created your API keys on
> `india.delta.exchange`, also set
> `DELTA_BASE_URL=https://api.india.delta.exchange` and
> `DELTA_WS_URL=wss://socket.india.delta.exchange` in `.env`. The global
> `api.delta.exchange` deployment is a separate platform and will return
> `invalid_api_key` for India-issued keys.

---

## Architecture

* `DeltaClient` — async REST + websocket client for [Delta Exchange](https://www.delta.exchange).
  Handles HMAC-SHA256 request signing, product discovery, order-book polling
  and websocket streaming, account leverage management, and order submission /
  cancellation / position reads.
* `DeltaEngine` — the scan/execute loop. Refreshes the market universe on a
  fixed cadence (or via websocket), feeds market state to every strategy,
  collects signals, and routes them through the `Executor`.
* `DeltaPerpetualScalper` — directional mean-reversion strategy that fires on
  z-score reversion + order-flow imbalance, and closes positions on
  configurable take-profit / stop-loss thresholds.
* `Executor` — atomic order router. Pre-vets every leg through the
  `RiskManager`, sizes against leverage-aware buying power, clamps
  `reduce_only` (closing) legs to the actual open position, submits, and
  rolls back partial fills.
* `GreedyTradeManager` — dynamic TP/SL overlay; takes ownership of
  per-trade TP (= fees + $N), trailing SL, leverage selection, and
  aggressive compounding (see *Greedy autopilot* section).
* `AdaptiveBrain` — live edge tracker + regime router. Mutes
  underperforming strategies, vetoes fresh entries during news
  spikes / wrong regimes, applies a per-(strategy, symbol) post-loss
  cool-down to suppress revenge trades, enforces a daily-loss circuit
  breaker and a gross-exposure correlation cap, and **tags every
  vetoed signal** so the dashboard can show *why* it was rejected
  (see *Adaptive brain* section).
* `RegimeBook` — per-symbol classifier that streams every mid into
  a RANGE / TREND_* / HIGH_VOL / NEWS_SPIKE detector; the brain
  consumes its snapshots to route signals.
* `MoneyPrinter` — adaptive strategy trained from the offline
  pipeline. Only fires during historically profitable hours (per-
  symbol heatmap) **and** when an `sklearn` gradient-boosted
  classifier estimates `P(win) >= min_win_probability` **and**
  ATR% is inside the configured volatility band. Exits are ATR-
  tuned. Degrades gracefully when artefacts are missing
  (see *Strategy: Money Printer*).
* `aera.data.history`, `aera.backtest.{replay,sweep,analysis}`,
  `aera.ml.{features,model}` — the **offline pipeline** that
  fetches OHLCV, replays every existing strategy across (symbol ×
  resolution × leverage), produces per-hour profitability heatmaps,
  and trains the gradient-boosted classifier that powers MoneyPrinter
  (see *Offline backtest + ML pipeline*).
* `DeltaPaperExchange` — simulates fills against the live order book using a
  slippage model. Useful for risk-free testing on real market data.
* `DeltaLiveExchange` — routes orders to the real Delta REST API.
* `Portfolio` — single source of truth for cash, locked margin, positions,
  realised + unrealised P&L, drawdown, and peak settled wealth.
* `RiskManager` — vets every trade against bankroll, per-trade and per-market
  exposure caps (all leverage-aware), drawdown, and consecutive-loss circuit
  breakers.
* `DashboardState` + FastAPI server — live in-memory state container and a
  single-page UI showing equity curve, fills, signals, open positions,
  watchlist, and engine controls.

---

## Repository layout

```
aera/
├── core/                  Portfolio, Risk, Compounding, Greedy,
│                          AdaptiveBrain, DeltaEngine
├── markets/               Market dataclass, OrderBook, Delta client + WS + signing
├── signals/               Microstructure features: z-score, OFI, L2 depth
│                          imbalance + tape inference + wall tracking +
│                          rolling tick-stream + whale-detection trade tape +
│                          rolling-window micro-VWAP stream +
│                          time-bucketed OHLC bar stream with
│                          fractal-pivot swing detection +
│                          per-symbol regime detector (RANGE / TREND /
│                          HIGH_VOL / NEWS_SPIKE)
├── strategies/            Strategy base + DeltaPerpetualScalper +
│                          OrderBookSniper + TickReversalScalp +
│                          BidAskSpreadFade + FlowScalp + MicroVWAPSniper +
│                          StopHuntReversal + MoneyPrinter
│                          (all strategies call base.sync_position_state at
│                          the top of every scan to prevent phantom positions)
├── execution/             Executor, slippage, Delta paper / live exchanges
├── data/                  History layer: Delta /v2/history/candles client +
│                          local Parquet/CSV cache (CandleStore)
├── backtest/              BarReplay engine + multi-axis sweep + per-hour
│                          profitability analysis (HourMap)
├── ml/                    Feature engineering (15 features) + sklearn
│                          gradient-boosted profitability classifier
├── dashboard/             FastAPI server + state container + static UI
├── settings.py            Pydantic config schema with env / YAML layering
└── logging.py             Rich console logger

scripts/                   Entry points:
  scan_delta.py            Read-only product/order-book scanner
  run_delta.py             Main paper / live runner
  run_dashboard.py         run_delta + dashboard wrapper
  simulate_growth.py       Pure-math Monte Carlo
  fetch_history.py         Cache N days of OHLCV
  backtest.py              Single-config backtest
  sweep_backtest.py        Grid sweep -> CSV summaries + hour_maps.json
  train_money_printer.py   Fit MoneyPrinter's ML model (model.joblib)
  train_ensemble.py        Fit per-symbol classifier ensemble (data/money_printer/ensemble/)
  train_sequence_model.py  Fit transformer encoder (needs torch; sequence_model.pt)
  train_rl_agent.py        Fit DQN trading agent (rl_policy.npz)

config/config.yaml         Single source of truth for runtime knobs
data/                      Runtime caches (gitignored / cursorignored):
  history/<SYM>/<res>.parquet     Cached OHLCV
  backtest/sweep_*.csv            Sweep summaries
  money_printer/{hour_maps.json,  MoneyPrinter artefacts (consumed live)
                 model.joblib}
tests/                     pytest suite (397 tests)
.env.example               Template for live credentials
.gitignore                 Standard Python + project-local exclusions
.cursorignore              Cursor IDE indexing exclusions (mirrors .gitignore +
                           runtime data caches + large static UI bundles)
CONTEXT.md                 Compact project map for AI assistants / new
                           contributors (architecture, conventions, gotchas)
```

---

## Strategy: Delta perpetual scalper

Delta perpetuals trade at a true mark price tied to spot, so a z-score
reversion has real mean-reversion gravity behind it (the funding-rate
mechanism enforces that the perp tracks the index). Short-horizon
mean-reversion in liquid perps is a long-known and well-documented edge.

**Entry logic** (per symbol, every scan tick):

1. Update a rolling z-score of mid-price over the last `zscore_window` ticks.
2. Update an order-flow-imbalance (OFI) EMA over top-of-book sizes.
3. If `z <= -zscore_entry` AND `ofi >= +ofi_threshold` → emit a **BUY**.
   If `z >= +zscore_entry` AND `ofi <= -ofi_threshold` → emit a **SELL**.
4. Skip if top-of-book depth is below `min_depth_contracts`.
5. Skip if mid hasn't moved at least `rearm_distance_bps` since the last fire
   on this symbol (cheap debouncer).

Two-condition gating (z-score AND OFI in the same direction) cuts the false-
positive rate from pure z-score scalping considerably.

**Sizing** is delegated to the `Executor`: the strategy emits a reference
notional and the executor scales each leg to
`trade_size_fraction × buying_power` where `buying_power = bankroll × leverage`,
clamped by `max_trade_fraction × buying_power` and the cash-availability cap.

---

## Strategy: Order Book Sniper (DOM scalp)

A high-frequency micro-profit strategy that "front-runs" visible bid/ask
walls. Off by default; flip `strategies.order_book_sniper.enabled` to
`true` in `config/config.yaml` to run it alongside the mean-reversion
scalper. Designed for liquid majors (BTC-PERP, ETH-PERP). Targets
~5 bps moves with ~3 bps stops, holds for 1–10 seconds, and bails
immediately if the structural thesis (a real resting wall providing
support / resistance) is falsified by a spoofing pull.

**Entry logic** (per symbol, every scan tick):

1. **Depth imbalance**: sum cumulative bid + ask sizes within
   `imbalance_band_bps` of mid (top `imbalance_max_levels` levels). Fire
   when one side is `imbalance_ratio`× the other or larger.
2. **Tape confirmation**: at least `tape_min_count` aggressive takers in
   the same direction over the last `tape_window_seconds`. The bot
   *infers* taker buys/sells from successive top-of-book deltas (best-ask
   size shrinking with the price unchanged → market buy ate liquidity)
   so no separate trades-channel subscription is required.
3. **Entry**: limit at `best_bid + entry_tick_offset × minimum_tick` for
   a long (mirror for shorts). The paper exchange fills immediately
   against the live book; the live exchange rides on
   `time_in_force = "ioc"` so unfilled signals don't sit stale.
4. **Wall capture**: the largest in-band resting level on the favoured
   side is recorded at entry. This becomes the *spoof reference* — if it
   shrinks past `spoof_vanish_ratio` within `spoof_persist_seconds` of
   entry, the strategy market-exits regardless of P&L.

**Exit logic** (highest priority first):

1. **Spoof exit** — entry-side wall vanished inside the persistence
   window. The structural thesis is dead, get flat.
2. **Hold-timeout** — `now − entry_time > max_hold_seconds`. The trade
   has overstayed its 1–10 second envelope.
3. **Stop-loss** — `take_profit_usd` / `stop_loss_usd` if a portfolio is
   wired and either is > 0; otherwise the `*_pct` thresholds (default
   −0.03%) computed against entry mid.
4. **Take-profit** — same precedence, default +0.05%.

**Reusable building blocks** (in `aera/signals/order_book.py`):

* `measure_depth_imbalance(book, band_bps, max_levels)` — cumulative
  bid/ask depth within a price band, returns a `DepthImbalanceSnapshot`.
* `TapeInferrer(window_seconds, max_step_fraction)` — sliding-window
  count of inferred aggressive takers from book deltas.
* `WallSnapshot.vanished(book, ratio_threshold, persist_seconds)` — the
  spoofing-detection primitive.

These are independent of the sniper itself; any future strategy that
wants depth-aware features can import them directly.

```bash
# Paper-trade the sniper alongside the mean-reversion scalper
python -m scripts.run_delta \
    --strategies delta_perp_scalper,order_book_sniper \
    --bankroll 27 --dashboard

# Sniper-only, BTC-PERP and ETH-PERP only
python -m scripts.run_delta \
    --strategies order_book_sniper --symbols BTCUSD,ETHUSD --websocket
```

---

## Strategy: Tick Reversal Scalp

A medium-high frequency micro-profit strategy that fades exhaustion
runs at the tick level. Off by default; flip
`strategies.tick_reversal_scalp.enabled` to `true` in
`config/config.yaml` to enable. Designed for liquid majors (BTC-PERP,
SOL-PERP) but works on any symbol whose tick rate is fast enough that
a 5-tick streak prints in seconds. Targets ~4 bps moves with ~2.5 bps
stops, holds for 5–30 seconds, and force-exits at 30 s regardless of
P&L (stale reversals turn into trend trades).

The core insight: when retail momentum bots over-extend at a key
level, the *per-tick* aggressive volume tends to shrink as buyers (or
sellers) run out of size. A 5+ tick same-direction run with decaying
per-tick volume that lands at a recent S/R level usually reverts a few
bps before continuing — the strategy fades that revert.

**Entry logic** (per symbol, every scan tick):

1. **Tick-stream maintenance**: each scan, the symbol's `TickStream`
   ingests the latest book. If the mid moved, a new `Tick` is recorded
   with the direction (`±1`), the inferred per-tick eaten liquidity,
   and the spread / top-of-book sizes. Flat scans do not produce ticks
   — so the tick rate adapts to market activity, not the scan cadence.
2. **Streak detection**: count trailing same-direction ticks. Fire when
   the length is ≥ `min_streak` (default 5).
3. **Size decay**: the eaten size series must decay by at least
   `size_decay_threshold` end-to-end (`1 − last/first`, default 20%).
   "Eaten" is inferred from successive top-of-book deltas — same
   technique as the sniper's `TapeInferrer`.
4. **S/R proxy** (optional, `sr_band_bps > 0`): the current mid must be
   within `±sr_band_bps` of the `sr_lookback_ticks` extreme in the
   streak direction (the local low for a long, local high for a
   short). Looser when sr_band is wide; tighter when sr_band is in the
   single bps — useful for double-bottom / double-top setups.
5. **Depth-trend** (optional, `require_depth_trend=true`): the
   favoured side's top-of-book size at the end of the streak must
   exceed the size at the start — buyers stepping in on a long fade,
   sellers on a short. Single-level books make this strict; multi-
   level data (websocket) makes it sharper.
6. **Safety filters** (any tripping veto, set the multiplier / threshold
   to 0 to disable):
   * **Spread guard** — skip when `current_spread > max_spread_multiple
     × EMA spread` (default 3×). Wide spreads = thin venue, the
     thesis assumes a normal microstructure.
   * **News-spike proxy** — skip when *any* tick in the last
     `news_lookback_seconds` (60) moved more than `news_max_tick_bps`
     (50). A 50-bp single tick is a flash event, not a normal scalp.
   * **Volume spike** — skip when the short-window (5 s) per-second
     eaten volume rate exceeds the long-window (60 s) rate by
     `volume_spike_multiple` (5×). Same shape as the news proxy from
     the trade side.
7. **Entry**: limit at mid (`+ entry_offset_bps`). The live exchange
   sends `time_in_force = "ioc"` — unfilled inside one tick = gone.
   Spec is "300 ms IOC", which is functionally identical here because
   the scan-loop cadence is < 300 ms.

**Exit logic** (highest priority first):

1. **Hold-timeout** — `now − entry_time > max_hold_seconds`. Spec:
   close after 30 s regardless of P&L.
2. **Stop-loss** — `take_profit_usd` / `stop_loss_usd` if a portfolio
   is wired and either is > 0; otherwise the `*_pct` thresholds
   (default −0.025%) computed against entry mid.
3. **Take-profit** — same precedence, default +0.04%.

**Reusable building blocks** (in `aera/signals/tick_stream.py`):

* `TickStream` — rolling tick buffer with windowed accessors:
  `current_streak`, `recent_extreme`, `current_spread_multiple`,
  `volume_in_window`, `volume_spike_ratio`, `max_tick_move_bps`,
  `depth_trend`.
* `Tick` — single observation including `prev_mid` so per-tick
  magnitudes survive even at the buffer's leading edge.

The funding-rate filter from the original spec is *not* implemented;
the bot doesn't ingest funding data today. The spread, volume, and
news-spike proxies cover the spirit of "abnormal regime → skip".

```bash
# Paper-trade the reversal scalper on its own
python -m scripts.run_delta \
    --strategies tick_reversal_scalp --symbols BTCUSD,SOLUSD --websocket

# Run all three together with the dashboard
python -m scripts.run_delta \
    --strategies delta_perp_scalper,order_book_sniper,tick_reversal_scalp \
    --bankroll 27 --dashboard
```

---

## Strategy: Bid-Ask Spread Fade (Market Making Lite)

A high-frequency market-making lite strategy that posts simultaneous
resting limit orders on both sides of the spread and earns the spread
as carry. Off by default; flip `strategies.bid_ask_spread_fade.enabled`
to `true` in `config/config.yaml` to run it alongside the other Delta
strategies. Designed for liquid majors (BTC-PERP, ETH-PERP) where the
spread occasionally widens during illiquidity spikes. Targets a 60%
capture of the spread, holds inventory net-flat with active skewing,
and bails on volatility spikes via a 5-second kill switch.

The core insight: when the spread widens past ~3 bps (Delta's typical
quiet-market spread is 1–2 bps on majors), posting a tight two-sided
quote inside the wide spread harvests the imbalance between which
side the next aggressive order takes. As long as you fill on both
sides over time, the inventory walks home and the captured spread
shows up as realised PnL net of two maker fees.

**Per-cycle behaviour** (gated to `refresh_rate_ms`, default 500 ms):

1. **Kill-switch.** Track mid prints over the last `kill_window_seconds`
   (5s). If `(max − min) / oldest_mid` exceeds `kill_move_pct` (0.08%),
   suspend quoting until the spike rolls off the window. Existing
   inventory rides through — the spec is explicit about waiting for
   normal regime to resume rather than dumping into the volatility.
2. **Spread gate.** Skip the cycle if `spread / mid < min_spread_pct`
   (default 0.03%). Tighter spreads cannot clear two maker fees plus
   the `min_net_edge_bps` floor (4 bps) and quoting them is net-
   negative.
3. **Net-edge gate.** Project net capture as
   `(spread_pct × capture_target × 10_000) − 2 × maker_fee_bps`. Only
   quote when this clears `min_net_edge_bps`. Delta's maker fee is
   0.02% so the default settings keep the strategy above its break-
   even line.
4. **Quote prices.** Bid at `mid − spread × capture_target / 2`, ask at
   `mid + spread × capture_target / 2`. Default capture = 60%, so the
   quotes sit at the inner 60% of the spread with a 20% safety buffer
   on each side.
5. **Inventory skew.** Read open-position notional from the live
   `Portfolio`. If `|inventory| > inventory_skew_threshold_usd` (default
   ±$10), shift BOTH quotes by `inventory_skew_ticks` ticks in the
   direction that biases offloading (down when long, up when short).
6. **Inventory cap.** If `|inventory| > max_inventory_usd` (default
   ±$15), suppress the side that would grow the imbalance further.
   The other side is still quoted so inventory can walk back inside.
7. **Emit.** Two **separate single-leg Signals** per cycle (one BUY,
   one SELL). They are processed independently by the Executor — no
   atomic unwinding of partial fills, which is exactly what market
   making needs: independent fills are the entire point of the
   strategy.

**Expected micro P&L per cycle** (with default config):

| Component       | bps     |
| --------------- | ------: |
| Spread captured | +6.0    |
| Maker fee (×2)  | −4.0    |
| **Net**         | **+2.0**|

The spec target of +0.04% (4 bps) net is achievable with a wider
captured spread or a higher capture ratio; the defaults here are
deliberately conservative.

**Live-mode caveat.** True market making requires resting limit orders
with `post_only=true` + `time_in_force="gtc"`. The default
`DeltaLiveExchange` uses `ioc` which expires unfilled limits inside a
tick. To deploy this strategy live, flip `post_only=True` on the
exchange and switch its `time_in_force` to `gtc` so the quotes
actually sit on the book. Paper-trading does not need any change.

```bash
# Paper-trade the spread-fade MM lite alongside the sniper
python -m scripts.run_delta \
    --strategies order_book_sniper,bid_ask_spread_fade \
    --symbols BTCUSD,ETHUSD --bankroll 27 --dashboard

# Spread-fade only, with the WS book feed for sub-second quote refresh
python -m scripts.run_delta \
    --strategies bid_ask_spread_fade --symbols BTCUSD --websocket
```

---

## Strategy: Tape Reading Momentum (Flow Scalp)

A pure order-flow HFT scalper that front-runs taker aggression. Reads
the live trade tape for *whale* prints — single market orders that slam
through multiple levels at ≥ 5× the rolling 100-trade average size —
and rides the resulting continuation. No indicators, no mean reversion;
the signal IS the flow. Off by default; flip
`strategies.flow_scalp.enabled` to `true` in `config/config.yaml` to
run it alongside the other Delta strategies. Designed for BTC-PERP and
other liquid majors during US market hours.

**Per-scan behaviour:**

1. **Detect whale.** Every scan, the strategy refreshes its per-symbol
   `TradeTape` (a rolling buffer of aggressive taker prints) and asks
   *"is there a trade in the last `confirm_window_seconds` whose size
   is ≥ `whale_multiple` × the pre-trade rolling average?"* The
   baseline is computed *as of the candidate trade* so the whale never
   inflates the threshold it has to clear. Spec defaults: 5× over 100
   trades, scanned across a 3-second window.
2. **Confirm.** Hold the whale as *pending* until ≥ `confirm_count`
   more same-direction prints land with size ≥ `confirm_multiple` ×
   the pre-baseline average. Spec: 1 more print ≥ 2× avg. Pending
   whales expire after `confirm_window_seconds` — single whales
   without follow-through are usually hedges, not continuations, and
   trading them costs you the edge.
3. **Enter.** Market entry at the touch (best ask for a long, best bid
   for a short). The signal carries `leverage_override` (spec: 5×) so
   the executor sizes against `bankroll × 5` instead of the venue's
   default leverage — moderate, sized for the tight TP/SL bands.
4. **Ride.** Three exits run in parallel, evaluated every tick:
   - **Hard TP** at `+take_profit_pct` from entry mid (spec: +0.08%).
   - **Hard SL** at `−stop_loss_pct` from entry mid (spec: −0.04%).
   - **Trailing stop** at `trailing_stop_pct` below the highest mid
     since entry, *armed only once mid prints above entry* (spec:
     0.02% trail). The effective stop on a long is therefore
     `max(hard_sl, trailing_level)` — never gives back more than 2 bps
     from the high, but always capped by the 4 bps hard floor.
5. **Time exit.** Force a market close `max_hold_seconds` (default 60s)
   after entry if neither TP, SL, nor trail tripped. Whale flow
   doesn't persist; overstaying decays the edge into noise.

**Data sourcing.** The tape ingests trades two ways:

* **Book-delta inference** (default). Successive top-of-book snapshots
  are diffed to extract per-side aggression events — an ask that
  shrinks at the same price is a taker buy of the diff; an ask that
  walks up is a full-level clear (the prior level was eaten end-to-
  end). Mirror logic for the bid. A `inference_max_step_fraction`
  guard drops single-tick collapses that look like cancellations
  rather than trades. This works against either Delta's REST poll
  loop or its websocket book feed without a separate trades-channel
  subscription.
* **External trades feed** (opt-in). Set
  `auto_infer_from_book: false` and call
  `FlowScalp.record_trade(symbol, price=..., size=..., side=...)`
  from your own websocket consumer when you want fidelity over
  inference accuracy. The strategy never mixes the two paths so the
  baseline math stays clean.

**Expected micro P&L per trade** (with default config and reasonable
fill quality):

| Component         | bps          |
| ----------------- | -----------: |
| Take-profit       | +8.0         |
| Trailing stop     | +2.0 to +6.0 |
| Stop-loss         | −4.0         |
| Taker fee (×2)    | −1.0 to −2.0 |

With a +0.08% target, a −0.04% hard stop, and a 0.02% trail riding the
peak, the strategy needs roughly a 1-in-3 win rate to be net positive
after fees. The 60s hold cap keeps Monte Carlo paths bounded — at
5–15 trades/hour during US session this is high enough turnover to
compound a $27 bankroll meaningfully on Delta's high-leverage perps.

```bash
# Paper-trade the flow scalper on BTC during US hours
python -m scripts.run_delta \
    --strategies flow_scalp --symbols BTCUSD \
    --bankroll 27 --dashboard

# Stack the flow scalper with the DOM sniper — they fire on different
# microstructure regimes (whale aggression vs. resting-wall imbalance)
# and rarely overlap in time, so combined cadence is roughly additive.
python -m scripts.run_delta \
    --strategies flow_scalp,order_book_sniper \
    --symbols BTCUSD,ETHUSD --bankroll 27 --websocket --dashboard
```

---

## Strategy: Micro VWAP Reversion Sniper

A medium-frequency mean-reversion scalper that fades short-term price
deviations from a 1-minute rolling micro-VWAP when short-window volume
drops off. Off by default; flip `strategies.micro_vwap_sniper.enabled`
to `true` in `config/config.yaml` to run it alongside the other Delta
strategies. Designed for high-liquidity perps (BTC-PERP, ETH-PERP,
SOL-PERP) during quiet intraday periods — the regime where exhausted
aggression genuinely reverts to the prevailing VWAP rather than
continuing.

The core insight: when price strays meaningfully from the rolling
micro-VWAP *and* the short-window trade rate is below the longer
baseline, the move has run without follow-on volume. That combo —
extension + drying flow — is the cleanest "exhausted aggressors"
signature available on a book-only feed, and it tends to snap back to
the VWAP value that existed *at the moment of the deviation* (the
target should not drift with the very mean we're trying to revert to).

**Per-scan behaviour:**

1. **Maintain the stream.** Every scan the symbol's `VWAPStream`
   ingests the latest book and infers per-trade volume from successive
   top-of-book deltas — same heuristic as `TradeTape`, but the stream
   keeps `(timestamp, price, size)` triples so the VWAP math has both
   numerator and denominator. The stream is buffer-capped (default
   5000 trades) so memory stays bounded across many symbols.
2. **Compute VWAP.** Rolling `Σ(price × size) / Σ(size)` over the last
   `vwap_window_seconds` (default 60 s). Returns `None` on a cold tape
   — the strategy waits for the baseline to populate before firing.
3. **Deviation gate.** Skip when `|mid − vwap| / vwap < deviation_pct`
   (default 0.0012 = 0.12%). Spec floor.
4. **Volume drop-off gate.** Skip when
   `short_rate / long_rate >= volume_ratio_max` (default 0.70). Compares
   per-second eaten volume in the trailing 10 s to per-second eaten
   volume in the trailing 5 minutes — short window must be *quieter*
   than the long baseline. Returns `None` on cold start, which the
   strategy treats as a skip (no baseline → no signal).
5. **Spread guard.** Skip when `spread / mid > max_spread_pct`
   (default 0.0005 = 0.05%). Wider spreads can't clear the round-trip
   cost. `0` disables.
6. **Hour-window skip.** Skip entries during the first
   `hour_skip_seconds` (default 300 s) of each wall-clock hour. The
   rolling-VWAP buffer drains across the hour boundary and produces
   transient false deviations — the spec is explicit about avoiding
   that window. The check uses `now % 3600` so it's deterministic
   against the injected clock for tests. `0` disables.
7. **Enter.** LONG when mid is below VWAP by more than `deviation_pct`,
   SHORT when above. Limit price = current touch (best ask for BUY,
   best bid for SELL); the executor's live path uses
   `time_in_force = "ioc"` so unfilled limits don't sit stale (matches
   the spec's "IOC with 1 s timeout").

**Exit logic** (highest priority first):

1. **Hold-timeout** — `now − entry_time > max_hold_seconds`. Default
   90 s. Reversions either happen or fade inside the window;
   overstaying decays the edge into a directional bet.
2. **USD or % stop-loss.** Fixed `stop_loss_pct` (default 0.0005 =
   0.05%) relative to entry mid. When `take_profit_usd > 0` or
   `stop_loss_usd > 0` *and* a portfolio is wired in, the USD path
   takes precedence (same contract as the other strategies).
3. **VWAP snap-back (primary TP).** Snapshot the VWAP value at entry
   time — `entry_vwap`. Close when mid returns to that level (or
   `entry_vwap + tp_extra_bps` in the favourable direction for an
   optional stretch target). The snapshot is *static* — never
   recomputed against the moving VWAP, so the target doesn't drift
   with the very mean we're reverting to.
4. **% take-profit (safety floor).** Used only if neither USD nor VWAP
   snap-back can fire (e.g. entry VWAP was zero, cold-start edge case).
   Default `take_profit_pct = 0.0007` (≈ 0.07%).

**Reusable building blocks** (in `aera/signals/vwap_stream.py`):

* `VWAPStream(window_seconds, max_trades, inference_max_step_fraction)`
  — rolling, time-windowed per-symbol VWAP. Read accessors:
  `vwap(window_seconds, now)`, `volume_in_window(seconds, now)`,
  `volume_ratio(short_seconds, long_seconds, now)`. Ingest paths:
  `update(book, now)` (book-delta inference) and `record(price, size,
  side, now)` (direct push for real trades-channel feeds).

**Expected micro P&L per trade** (with default config, after 5×
leverage on a $27 bankroll):

| Component             | bps          |
| --------------------- | -----------: |
| VWAP snap-back (full) | +12.0        |
| Snap-back (partial)   | +3.0 to +8.0 |
| Stop-loss             | −5.0         |
| Taker fee (×2)        | −1.0 to −2.0 |

A 0.12% deviation that snaps fully back to VWAP is a +12 bps move;
the strategy targets full snap-back as primary TP and falls back to
the 0.07% safety floor. With the volume-drop-off + hour-skip filters
active, expect 1–4 trades per symbol per hour on majors during quiet
US session — sparse but high-quality.

```bash
# Paper-trade the VWAP sniper on BTC + ETH + SOL during US hours
python -m scripts.run_delta \
    --strategies micro_vwap_sniper \
    --symbols BTCUSD,ETHUSD,SOLUSD --bankroll 27 --dashboard

# Stack it with the flow scalper — they fire on opposite regimes
# (VWAP fades exhausted moves; flow rides confirmed whale aggression)
# so they rarely conflict on the same tick.
python -m scripts.run_delta \
    --strategies micro_vwap_sniper,flow_scalp \
    --symbols BTCUSD,ETHUSD --bankroll 27 --websocket --dashboard
```

---

## Strategy: Stop Hunt / Liquidity Grab Reversal

A high-risk / high-reward HFT scalper that fades **engineered stop
hunts** — the classic market-maker move where a sudden 0.15%+ wick
pierces a recent swing low (or high) and immediately closes back
inside the level on the same 1-second candle. Those wicks are
algorithmic sweeps that vacuum retail stop clusters before the real
move unfolds; the strategy enters into the snap-back the instant the
wick bar closes. Off by default; flip
`strategies.stop_hunt_reversal.enabled` to `true` in
`config/config.yaml` to enable. Designed for BTC-PERP and ETH-PERP;
the spec notes that low-liquidity windows (Asian session, weekend
chop) produce the cleanest sweeps.

**Per-scan behaviour:**

1. **Aggregate 1 s OHLC bars.** Every scan, the symbol's `BarStream`
   ingests the latest book and (a) updates the in-progress bar's
   OHLC + per-side taker volumes (inferred from book deltas — same
   heuristic as `TradeTape`), and (b) rotates a fresh bar at each
   wall-clock second boundary. The closed bar from a rotation is the
   *only* candidate for sweep detection — the strategy waits for the
   wick candle to actually close before evaluating.
2. **Pre-mark key levels.** The latest `swing_count` (default 3)
   pivot highs and pivot lows are read from the trailing
   `swing_lookback_bars` (default 60 = the trailing minute). A bar
   is a swing high when its high strictly exceeds `swing_pivot_strength`
   bars on each side (5-bar fractal at the default). Levels become
   confirmed pivots `swing_pivot_strength` bars after they print —
   so a level can't qualify until the market has moved past it
   without exceeding.
3. **Test the just-closed bar against every gate** (any failure = skip):
   * **Wick depth** — bar's low must dip at least `wick_size_pct`
     (default 0.15%) below a marked swing low for a bullish sweep;
     bar's high must spike at least that far above a marked swing
     high for a bearish sweep.
   * **Recovery** — bar's close must be back inside the level.
   * **Body ratio** — `body / range < body_ratio_max` (default 0.30).
     Small body + big wick = the spec's "engineered" signature.
   * **Recovery speed** — `now − bar.start ≤ recovery_seconds`
     (default 3 s). With 1 s bars this is essentially always
     satisfied on the close-of-bar tick; the gate matters if the
     bar duration is raised.
   * **Volume confirmation** — wick bar's volume ≥
     `volume_multiple × avg_volume` over the trailing
     `volume_lookback_bars` (default 1.5× over 30 bars). The
     baseline excludes the wick bar itself so the spike doesn't
     inflate its own threshold.
   * **Delta confirmation** (`require_delta_confirmation = True`) —
     for a bearish sweep the wick bar's signed delta
     (`buy_volume − sell_volume`) must be ≤ `−delta_flip_threshold`
     (= net selling pressure on the close, matching the spec's
     "delta flips red"). Bullish sweeps mirror.
4. **Enter.** Market entry at the touch (best ask for long, best
   bid for short) the instant the wick bar closes. The live exchange
   submits with `time_in_force = "ioc"` — unfilled limits evaporate
   inside one scan, matching the spec's "do not wait for the next
   candle".

**Exit logic** (highest priority first):

1. **Hold-timeout** — `now − entry_time > max_hold_seconds` (default
   60 s). Stale snap-backs decay into trend trades; flatten.
2. **Wick-anchored stop** — exit when mid ≤ `wick_low × (1 −
   stop_extra_pct)` for a long (mirror for shorts). Spec: 0.08%
   below the wick. USD-PnL `stop_loss_usd` takes precedence when
   set with a portfolio attached.
3. **TP1 partial close** — once mid hits `entry × (1 + tp1_pct)`
   (spec: +0.10%), close `tp1_fraction` of the entry notional
   (spec: 60%). Fires exactly once per position; the remaining 40%
   rides toward TP2 with the wick-anchored stop still active.
4. **TP2 / final TP** — `take_profit_pct` (spec: +0.20%) flattens
   the remainder. USD-PnL `take_profit_usd` takes precedence when
   set; it always flattens the whole position (no partial-TP path
   in USD mode).

**Expected micro P&L per trade** (with default config and 5×
leverage on a small bankroll):

| Component         | bps                            |
| ----------------- | ------------------------------ |
| TP1 (60% closed)  | +6.0 (+10 bps × 0.6)           |
| TP2 (40% closed)  | +8.0 (+20 bps × 0.4)           |
| Full TP run       | +14.0 combined                 |
| Stop (wick + 8)   | ≈ −12 to −15 (wick + buffer)   |
| Taker fee (×2-3)  | −1.5 to −3.0                   |

The R:R sits near 1:1 on the worst-case stop and ~1:2.5 when the
wick is shallow (close to the spec floor) — matching the spec's
target ratio. Spec expects 1–4 setups per symbol per hour during
low-liquidity windows.

**Reusable building blocks** (in `aera/signals/bar_stream.py`):

* `BarStream(bar_seconds, max_bars, inference_max_step_fraction)` —
  time-bucketed OHLC bars + per-side taker volume + fractal-pivot
  swing detection. Read accessors: `closed_bars`, `current_bar`,
  `avg_volume(n)`, `swing_pivots(...)`,
  `recent_swing_highs(...)`, `recent_swing_lows(...)`.
* `Bar` — OHLC plus `buy_volume`, `sell_volume`, `volume`, `delta`,
  `range`, `body`, `body_ratio`, `upper_wick`, `lower_wick`,
  `is_bullish`.

```bash
# Paper-trade the sweep reversal on BTC + ETH during Asian session
python -m scripts.run_delta \
    --strategies stop_hunt_reversal \
    --symbols BTCUSD,ETHUSD --bankroll 27 --websocket --dashboard

# Stack it with the VWAP sniper — they're complementary fades
# (VWAP catches exhausted continuations, stop-hunt catches
# engineered single-bar wicks); pairing them roughly doubles the
# fire rate on quiet sessions.
python -m scripts.run_delta \
    --strategies stop_hunt_reversal,micro_vwap_sniper \
    --symbols BTCUSD,ETHUSD --bankroll 27 --websocket --dashboard
```

---

## Strategy: Money Printer (backtest-trained adaptive)

The eighth strategy is fundamentally different from the other
seven: it does not invent its edge from first principles. It
**reads** an edge from an offline backtest pipeline and an
ML-trained classifier, and trades only when the historical
evidence, the model, and current volatility all agree.

Off by default; flip `strategies.money_printer.enabled` to `true`
in `config/config.yaml`. **Train the artefacts first** (see the
*Offline backtest + ML pipeline* section); MoneyPrinter degrades
gracefully when artefacts are missing but adds the least value in
that mode.

**Inputs (read at construction time):**

* `data/money_printer/hour_maps.json` — per-(strategy, symbol)
  expected PnL by UTC hour-of-day. Produced by `scripts.sweep_backtest`.
* `data/money_printer/model.joblib` — a `sklearn.ensemble.HistGradientBoostingClassifier`
  predicting `P(win)` from 15 microstructure / volatility / time
  features. Produced by `scripts.train_money_printer`.

If either is missing the strategy still scans, but the
corresponding gate is disabled (logged once at INFO).

**Per-symbol scan logic (every tick):**

1. **Phantom-position sync** — every scan first reconciles the
   strategy's "I have an open position" bookkeeping with the
   `Portfolio` (same guard rail every other strategy uses).
2. **Bar aggregation** — append the latest mid to a rolling
   `_Bar` deque (default 200 bars, 60s buckets). Below
   `min_bars` (default 30) the strategy returns flat.
3. **Hour-of-day gate** — look up the current UTC hour for
   `(money_printer, <symbol>)` in the hour map. If the historical
   expectancy ≤ 0, skip.
4. **ATR band gate** — compute ATR% from the rolling bars. If it
   falls outside `[min_atr_pct, max_atr_pct]` (defaults 0.10% /
   1.50%), skip — too quiet to profit, too violent to control.
5. **ML P(win) gate** — extract a 15-feature vector and call
   `predict_proba_win`. If the probability < `min_win_probability`
   (default 0.55), skip.
6. **Side selection** — when the ML model is available, use a
   feature-derived directional bias (returns, RSI, EMA deviation,
   wick imbalance). Without a model, fall back to mean-reversion
   against the local mid.
7. **Confidence-scaled entry** — stamp `metadata["confidence"]`
   = ML score onto the leg; the executor reads it and tilts the
   final notional toward higher-confidence trades.

**Exits (highest priority first):**

1. **Max hold** — `now − entry_time > max_hold_seconds` (default
   180s). Flatten.
2. **ATR-tuned stop-loss** — `entry × (1 − sl_atr_mult × atr_pct)`
   for longs; mirror for shorts. Volatility-adaptive: a quiet
   market gets a tight stop, a noisy one gets room to breathe.
3. **ATR-tuned take-profit** — `entry × (1 + tp_atr_mult × atr_pct)`
   for longs; mirror for shorts. Same volatility scaling as the
   stop, with a wider multiplier by default (2.0× vs. 1.0×) so
   the strategy targets a 2:1 reward-to-risk per trade.

**Config knobs (`config/config.yaml -> strategies.money_printer:`)**

```yaml
strategies:
  money_printer:
    enabled: false              # opt-in; train artefacts first
    bar_seconds: 60             # rolling bar length
    max_bars: 200               # rolling window depth
    min_bars: 30                # don't trade before warm-up
    atr_window: 14
    min_atr_pct: 0.0010         # 0.10% — too quiet below this
    max_atr_pct: 0.0150         # 1.50% — too violent above
    tp_atr_mult: 2.0            # TP = entry × (1 + 2 × ATR%)
    sl_atr_mult: 1.0            # SL = entry × (1 − 1 × ATR%)
    min_win_probability: 0.55   # ML P(win) gate
    max_hold_seconds: 180.0
    hour_map_path: "data/money_printer/hour_maps.json"
    model_path:    "data/money_printer/model.joblib"
    rearm_distance_bps: 5.0
```

The strategy is regime-agnostic on purpose — its hour map and ML
score already encode regime context implicitly, so the brain's
`STRATEGY_REGIME_PREFS` map lists every regime as allowed for it.

```bash
# Once the offline pipeline has produced hour_maps.json + model.joblib:
python -m scripts.run_delta \
    --strategies money_printer \
    --symbols BTCUSD,ETHUSD,SOLUSD \
    --bankroll 27 --websocket --dashboard

# Stack it with the seven hand-built strategies; the brain
# automatically attributes PnL per-strategy and mutes losers.
python -m scripts.run_delta \
    --strategies delta_perp_scalper,money_printer \
    --bankroll 27 --dashboard
```

---

## Take-profit / stop-loss

The scalper supports two exit modes — pick one (or mix per-symbol).

### Absolute-USD thresholds (default)

```yaml
strategies:
  delta_perp_scalper:
    take_profit_usd: 5.0   # close at +$5 of unrealised P&L
    stop_loss_usd:   3.0   # close at −$3 of unrealised P&L
```

The strategy queries the live `Portfolio` position and computes
`pnl = (close-side price − avg_cost) × signed shares` on every tick.
Close-side price = bid for a long, ask for a short — i.e. what would
actually be realised on exit, not an over-optimistic mid mark. The instant
P&L crosses ±threshold, the strategy emits a reduce-only signal that the
executor clamps to the actual open notional and routes to Delta.

If `take_profit_usd > 0 OR stop_loss_usd > 0` AND a portfolio is wired into
the strategy (the standard runner does this), the USD path takes precedence
over the percentage thresholds. Stop-loss always wins on a simultaneous breach.

### Percentage thresholds (legacy)

```yaml
strategies:
  delta_perp_scalper:
    take_profit_pct: 0.01    # +1% from entry mid
    stop_loss_pct:   0.005   # −0.5% from entry mid
```

Reads the entry mid stamped at signal emission. Used when no portfolio is
wired in (e.g. hand-built test fixtures), or as a deliberate fallback by
setting both USD knobs to 0.

Disable an exit side entirely by setting its threshold to `0`. With all four
at `0`, positions only flatten when the opposite-direction reversion signal
fires.

---

## Greedy autopilot

A thin overlay (`aera.core.GreedyTradeManager`) that takes ownership
of **per-trade TP, per-trade SL, leverage selection, and compounding**
the moment you flip `greedy.enabled: true` in `config/config.yaml`. It
sits between the strategies and the executor — strategies still decide
*when* to fire; greedy decides *how big*, *at what leverage*, and *when
to flatten*.

### What it changes per trade

| Knob          | Where it normally comes from                      | What greedy does                                                                  |
| ------------- | ------------------------------------------------- | --------------------------------------------------------------------------------- |
| Take-profit   | Strategy `take_profit_usd` / `take_profit_pct`   | `round_trip_fees × fee_pad_multiple + min_profit_usd` (defaults: `$0.50 + $1`)    |
| Stop-loss     | Strategy `stop_loss_usd` / `stop_loss_pct`       | Starts at `-initial_sl_usd`, ratchets *up* with profit (trailing SL, never down)  |
| Leverage      | `markets.delta.leverage` or strategy override     | Streak-driven: starts at `min_leverage`, +`leverage_step` per win, /loss-streak   |
| Position size | `risk.trade_size_fraction × bankroll × leverage`  | `greedy.compound_fraction × bankroll × leverage` (≈90% by default)                |
| Exposure cap  | `risk.max_market_exposure × bankroll × leverage`  | Bypassed for fresh entries — concentrating into one market is the explicit intent |

The strategies' own TP/SL are *not* disabled — they remain a safety
belt. Set them to 0 in YAML to make greedy the sole exit authority.

### Per-tick lifecycle

1. **At the top of every scan**, the engine asks the greedy manager
   for any flatten signals on currently-tracked positions. Greedy
   computes live PnL = `(close_price − avg_cost) × shares`, runs the
   ratchet (raises trailing SL once `pnl ≥ lock_in_trigger_ratio ×
   tp_target`; rolls TP forward by `extend_tp_step_usd` on extension),
   and emits a reduce-only close when one of:
   * `pnl_usd ≥ tp_target_usd` → `greedy-tp`
   * `pnl_usd ≤ sl_level_usd` → `greedy-sl`
   * `now − entry_time > max_hold_seconds` → `greedy-timeout`
2. **Then strategies scan and emit entries.** For every fresh entry
   leg, the executor calls `greedy.decide_leverage(market)` and stamps
   the chosen leverage onto the leg. The executor's sizing path uses
   `compound_fraction × bankroll × leverage` as the target notional
   so almost the entire bankroll is deployed per trade. Realised PnL
   flows back into `bankroll` instantly via `Portfolio.apply_fill`,
   so the *next* entry is sized against the new wealth — that's the
   compounding loop.
3. **On every `ExecutionResult`**, the manager updates its position
   book (opens, refreshes adds, drops on closes) and bumps the live
   win/loss streak from `Portfolio.consecutive_losses`. The next
   `decide_leverage` call uses the updated streak.

### Math, by example

`$27 bankroll`, `5 bps taker fee`, `compound_fraction = 0.9`,
`min_leverage = 5x`, two consecutive wins:

* Buying power = `$27 × (5 + 2 × 5)x = $27 × 15x = $405`.
* Notional sized for entry = `$405 × 0.90 = $364.50`.
* Round-trip fees = `$364.50 × 5 bps × 2 = $0.36`.
* **TP target** = `$0.36 + $1.00 = $1.36` of unrealised PnL.
* **SL** starts at `-$1.50`. Once PnL crosses `$0.68` (50% of TP),
  it ratchets to `running_best − $0.50`. On a `$1.50` peak the SL
  sits at `$1.00` — locked-in profit even if the trade reverses.

The bot exits the moment any of those thresholds trip, takes the
`~$1.36` (or more, if extension keeps rolling the TP forward), and
restarts the loop with `$28.36` of bankroll.

### When greedy is the right mode

Use it when you want the bot to **chase the smallest possible
profit-locking exit as fast as possible**, accept that the per-trade
edge is tiny (fees + $1), and compensate by trading frequently and
compounding aggressively. It is the closest the codebase comes to a
"set it and forget it" autopilot.

It is *not* a good mode when:

* you care about per-market risk concentration (greedy bypasses
  `max_market_exposure` for entries),
* your strategies emit signals so rarely that a single losing trade
  is a meaningful fraction of the bankroll (use `min_leverage`-only
  with the streak step zeroed if so),
* you want symmetric TP/SL — greedy's SL is intentionally tighter
  in absolute USD than its TP target.

### Config knobs (`config/config.yaml -> greedy:`)

```yaml
greedy:
  enabled: true                  # opt-in
  min_profit_usd: 1.0            # the "+ $1" on top of fees
  fee_pad_multiple: 1.0          # multiplier on the fee estimate
  extend_tp_step_usd: 1.0        # roll TP forward on extension (0 disables)
  initial_sl_usd: 1.5            # cushion before trailing engages
  lock_in_trigger_ratio: 0.5     # start ratcheting at 50% of TP target
  trailing_giveback_usd: 0.5     # max give-back from running best PnL
  max_hold_seconds: 120.0        # hard exit if neither TP/SL fires
  min_leverage: 5.0              # leverage floor (first trade)
  max_leverage: 100.0
  leverage_step: 5.0             # +5x per consecutive win
  respect_venue_cap: true        # cap at market.metadata["leverage"]
  compound_fraction: 0.90        # deploy 90% of bankroll × leverage
  fee_override_bps: 0.0          # >0 overrides execution.taker_fee_bps
```

Env-var overrides: `BOT_GREEDY`, `BOT_GREEDY_MIN_PROFIT_USD`,
`BOT_GREEDY_MIN_LEVERAGE`, `BOT_GREEDY_MAX_LEVERAGE`,
`BOT_GREEDY_COMPOUND_FRACTION`.

```bash
# Paper-trade with greedy autopilot on (default config)
python -m scripts.run_delta --bankroll 27 --dashboard

# Same, but turn greedy off explicitly for an A/B compare
BOT_GREEDY=false python -m scripts.run_delta --bankroll 27 --dashboard

# Even more aggressive: 95% compounding, $2 floor profit per trade
BOT_GREEDY_COMPOUND_FRACTION=0.95 BOT_GREEDY_MIN_PROFIT_USD=2.0 \
    python -m scripts.run_delta --bankroll 27 --dashboard
```

---

## Adaptive brain

A live overlay (`aera.core.AdaptiveBrain`) that **measures the bot's own
edge in real time and refuses to trade when the edge isn't there.**
Sits between the strategies and the executor — *after* greedy's flatten
signals, *before* the risk vet — and applies five classes of veto /
shrink rule to every fresh entry:

| Gate                       | What it does                                                                     |
| -------------------------- | -------------------------------------------------------------------------------- |
| **Performance gate**       | Auto-mutes a strategy when its rolling win-rate / expectancy crashes             |
| **Regime gate**            | Mean-reversion only in RANGE; flow-scalp only in TREND_*; NEWS_SPIKE vetoes all  |
| **Post-loss cool-down**    | After ANY losing trade, the *(strategy, symbol)* pair waits N seconds before re-firing |
| **Daily-loss cap**         | Halts new entries when 24h realised PnL drops below `-daily_loss_pct × bankroll` |
| **Correlation cap**        | Caps total gross long (or short) notional across all symbols                     |

Reduce-only legs (TP / SL closes, greedy flattens) **always flow**
through the brain regardless of state — the brain only ever shrinks or
vetoes fresh entries, never blocks the bot from getting flat.

**Every vetoed signal is tagged** with `metadata["brain_veto_reason"]`
before being dropped, and surfaced to the dashboard via a synthetic
rejected `ExecutionResult` so the user can read *why* a fire was
suppressed (e.g. `brain: post-loss cool-down on BTCUSD (47s left)` or
`brain: muted (win-rate 0.18 < 0.40 over 12 trades)`).

### Per-strategy performance tracker

Every closed round-trip's PnL is fed into the brain's per-strategy
tracker (rolling deque of the last `perf_window` trades, default 30).
The tracker exposes:

* rolling win-rate, expectancy, profit factor,
* consecutive wins / losses,
* mute state (with auto-expire and probation graduation),
* live size multiplier per strategy.

Mute fires when ANY of the following is true after at least
`min_trades_for_eval` (default 10) closed trades:

* rolling win-rate < `min_win_rate` (default 0.40),
* rolling expectancy < `min_expectancy_usd` (default 0.0),
* consecutive losses ≥ `max_strategy_loss_streak` (default 2 —
  intentionally aggressive; two consecutive losses is enough
  evidence to step back).

A muted strategy stops emitting fresh entries for `mute_seconds`
(default 600s = 10 min). After the cooldown it returns *on probation*
at `probation_size_mult` (default 0.5 = half size) and stays there
until it logs `probation_trades` (default 5) closed trades with
non-negative average expectancy — then it graduates back to 1.0.

### Post-loss cool-down (per strategy × symbol)

Independently of the per-strategy mute, every losing trade arms a
short per-(strategy, symbol) cool-down (default 60s). During the
cool-down the brain refuses fresh entries from *that exact pair*
and tags the rejection `brain: post-loss cool-down on <symbol>
(<N>s left)`. Other symbols on the same strategy are unaffected,
so a single bad SOLUSD trade won't shut down BTCUSD scalping.

This is the bot's revenge-trade suppressor: a quick stop-out almost
always means the immediate microstructure thesis (mean-reversion
level, wall, exhaustion print) is invalidated, and re-firing inside
seconds is the highest-loss-probability move in the dataset.

### Regime detector

`aera.signals.RegimeBook` runs a streaming classifier per symbol that
maps the live mid stream to one of five regimes:

* **RANGE**       — chop, low drift; mean-reversion strategies work
* **TREND_UP**    — clean upward drift; momentum strategies work
* **TREND_DOWN**  — clean downward drift; momentum strategies work
* **HIGH_VOL**    — short-window ATR ≥ N × long-window ATR; shrink size
* **NEWS_SPIKE**  — single tick > `regime_news_tick_bps`; veto everything

Each strategy declares which regimes its signals are appropriate for
(see `STRATEGY_REGIME_PREFS` in `aera/core/brain.py`):

| Strategy                 | Allowed regimes                             |
| ------------------------ | ------------------------------------------- |
| `delta_perp_scalper`     | RANGE, UNKNOWN                              |
| `order_book_sniper`      | RANGE, UNKNOWN                              |
| `tick_reversal_scalp`    | RANGE, UNKNOWN                              |
| `bid_ask_spread_fade`    | RANGE, UNKNOWN                              |
| `micro_vwap_sniper`      | RANGE, UNKNOWN                              |
| `stop_hunt_reversal`     | RANGE, UNKNOWN                              |
| `flow_scalp`             | TREND_UP, TREND_DOWN, UNKNOWN               |
| `money_printer`          | ALL (regime context is encoded in ML score) |

By default the regime gate is a **soft veto**: signals outside the
allow-list are still emitted, but the brain shrinks them by
`wrong_regime_size_mult` (default 0.5) instead of dropping them
outright. Set `regime_soft_veto: false` for the old hard-veto
behaviour. NEWS_SPIKE is always a hard veto regardless of this
flag.

HIGH_VOL doesn't appear in any allow-list — it's the brain's
*shrink* signal, not a veto. The size multiplier is multiplied by
`high_vol_size_mult` (default 0.5) so a HIGH_VOL fire goes through
at half (or smaller) notional, giving the bot a way to keep harvesting
the occasional fast-market scalp at controlled risk.

### Daily-loss circuit breaker

A rolling 24h ledger of realised PnL. When it drops below
`-daily_loss_pct × bankroll` (default `-10%`), the brain refuses ALL
new entries until the window rolls off. Closes still flow so existing
positions can flatten. This is the brain's "today is not the day, stop
digging" gate; the bot's hard `max_drawdown` halt (now `25%` by default)
catches multi-day bleed past this point.

### Correlation / gross-exposure cap

`max_gross_exposure_mult × bankroll × leverage` (i.e. scaled with
**buying power**, not just settled wealth) is the brain's per-side
(long / short) gross-notional ceiling across every symbol it touches.
With the default 2.0×, a $100 bankroll, and 50× leverage, total
long notional is capped at `2 × 100 × 50 = $10,000` — so the brain
refuses to let the bot end up 100% directional crypto on a single
bad tick where BTC / ETH / SOL all fire in the same direction
simultaneously, *but* without strangling the leveraged scalper down
to one-trade-at-a-time the way a `× settled_wealth` cap would.

### Per-strategy dynamic sizing

The size multiplier the brain applies to every fresh entry is
`size_mult × loss_streak_penalty × regime_shrink`, where:

* `size_mult` starts at 1.0 and is set to `probation_size_mult` after
  a mute (graduates back to 1.0 after probation).
* `loss_streak_penalty = 1 / (1 + 0.5 × (consecutive_losses - 1))` once
  consecutive losses reach 2 — shrinks geometrically.
* `regime_shrink = high_vol_size_mult` in HIGH_VOL regime, else 1.0.

The multiplier is applied *on top of* the executor's
`trade_size_fraction` / greedy `compound_fraction`, so a probation
strategy in a HIGH_VOL regime with 4 prior losses on a $100 bankroll
fires at `0.5 × (1 / (1 + 0.5 × 3)) × 0.5 = 0.10` of the normal
notional — the brain de-risks fast and aggressively when conditions
turn.

### Config knobs (`config/config.yaml -> brain:`)

```yaml
brain:
  enabled: true
  min_trades_for_eval: 10
  perf_window: 30
  min_win_rate: 0.40
  min_expectancy_usd: 0.0
  max_strategy_loss_streak: 2          # 2 consecutive losses -> mute
  mute_seconds: 600.0
  probation_trades: 5
  probation_size_mult: 0.5
  post_loss_cooldown_seconds: 60.0     # per (strategy, symbol)
  regime_short_window: 30
  regime_long_window: 300
  regime_trend_threshold: 0.60         # loosened from 0.30 (less trigger-happy)
  regime_high_vol_ratio: 2.5           # loosened from 2.0
  regime_news_tick_bps: 35.0           # loosened from 25.0
  regime_soft_veto: true               # shrink instead of drop on wrong regime
  wrong_regime_size_mult: 0.5
  high_vol_size_mult: 0.5
  daily_loss_pct: 0.10
  daily_window_seconds: 86400.0
  max_gross_exposure_mult: 2.0         # × (bankroll × leverage)
```

Env-var overrides: `BOT_BRAIN`, `BOT_BRAIN_MIN_WIN_RATE`,
`BOT_BRAIN_MIN_EXPECTANCY_USD`, `BOT_BRAIN_DAILY_LOSS_PCT`,
`BOT_BRAIN_MAX_GROSS_EXPOSURE_MULT`, `BOT_BRAIN_MUTE_SECONDS`.

```bash
# Paper-trade with the brain on (default)
python -m scripts.run_delta --bankroll 27 --dashboard

# Same, but loosen the brain so it only intervenes on extreme losers
BOT_BRAIN_MIN_WIN_RATE=0.30 BOT_BRAIN_DAILY_LOSS_PCT=0.15 \
    python -m scripts.run_delta --bankroll 27 --dashboard

# Brain off entirely (back to greedy + raw strategies only)
BOT_BRAIN=false python -m scripts.run_delta --bankroll 27 --dashboard
```

---

## Risk management

`RiskManager.vet()` runs on every leg before it's submitted. It enforces:

| Check                          | Default cap                               |
| ------------------------------ | ----------------------------------------- |
| **Bankroll** (margin)          | `stake / leverage <= bankroll`            |
| **Per-trade fraction**         | `largest leg <= max_trade_fraction × buying_power` |
| **Per-market exposure**        | `existing + new <= max_market_exposure × buying_power` |
| **Drawdown circuit breaker**   | hard halt if `drawdown >= max_drawdown` (25%) |
| **Loss-streak cool-down**      | time-limited pause after `max_consecutive_losses` (default 15) for `loss_streak_cooldown_seconds` (default 300s) |

Where `buying_power = bankroll × leverage`. All four caps are leverage-aware,
so a $20 bankroll at 50× behaves like $1,000 of buying power for sizing
purposes, but only $20 of cash for margin coverage.

`reduce_only` (closing) legs bypass the per-market and bankroll checks because
they shrink exposure rather than open it. The executor enforces that they can
only ever reduce a position.

### `trade_size_fraction` vs `max_trade_fraction` vs `max_market_exposure`

These three knobs interact subtly:

* `trade_size_fraction`: the **target** size for the largest leg of every
  signal, as a fraction of buying power. `0.5` = scale every signal so the
  largest leg is 50% of `bankroll × leverage`.
* `max_trade_fraction`: the **ceiling** — clamps the target down. Must be
  `>= trade_size_fraction` or the target gets silently truncated.
* `max_market_exposure`: per-market cap on **existing + new** notional in
  one symbol. Must be `>= max_trade_fraction` or every fresh single-leg
  signal gets vet-rejected on its first attempt.

`get_settings()` emits a startup `log.warning` if any of these invariants are
violated, with a concrete remediation hint.

### Halt vs cool-down (changed in 2026-05)

Earlier versions of the bot treated `max_consecutive_losses` as a
**hard halt** that required a manual dashboard resume to clear.
That was the wrong default for a scalper: 6-loss streaks happen
inside a normal trading session, and the bot would spend the rest
of the day frozen.

Now the consecutive-loss trigger is a **time-limited cool-down**:

* Default `max_consecutive_losses` raised to **15** (was 6).
* Hitting it pauses *fresh entries* for
  `loss_streak_cooldown_seconds` (default **300s = 5 min**).
* The pause is automatic-clear; no dashboard interaction needed.
* Vetoed signals are tagged
  `risk: loss-streak cool-down (<N>s left)` so the dashboard
  shows the cool-down rather than a silent halt.
* `max_drawdown` (25%) is still a permanent hard halt — only
  `resume()` clears it. That ordering is intentional: per-
  session loss-streaks self-heal, but a 25% drawdown means
  something is structurally wrong and a human should look.

---

## Offline backtest + ML pipeline

The bot ships with a self-contained offline pipeline that turns
historical Delta OHLCV into:

* **`data/backtest/sweep_summary.csv`** — leaderboard of every
  (strategy × symbol × resolution × leverage) configuration ranked
  by realised PnL, Sharpe, max DD, profit factor, win rate.
* **`data/backtest/sweep_trades.csv`** — every individual closed
  round-trip the sweep produced. Used as the training dataset for
  the ML model.
* **`data/money_printer/hour_maps.json`** — per-(strategy, symbol)
  24-bucket profitability map by UTC hour-of-day. Consumed by
  `MoneyPrinter` as its time-of-day gate.
* **`data/money_printer/model.joblib`** — a `sklearn` GBT
  classifier predicting `P(win)` per trade setup. Consumed by
  `MoneyPrinter` as its ML gate.

The pipeline reuses the **exact same strategy classes** the live
runner uses; there is no parallel "research" implementation to
drift out of sync.

### Module layout

| Module                       | Role                                                              |
| ---------------------------- | ----------------------------------------------------------------- |
| `aera.data.history`          | `DeltaHistoryClient` (paginated `/v2/history/candles`) + `CandleStore` (Parquet/CSV cache w/ incremental dedupe) + `fetch_history()` convenience wrapper |
| `aera.backtest.replay`       | `candles_to_market_stream` (4-tick-per-bar synthetic OHLC stream) + `BarReplay` (drives a Strategy through a real `Portfolio`, records every trade as a `TradeRecord`, returns a `BacktestResult` with PnL/Sharpe/max-DD/profit-factor metrics) |
| `aera.backtest.sweep`        | `SweepConfig` + `run_sweep` — thread-pool grid search; candles cached per (symbol, resolution) and shared across per-leverage replays |
| `aera.backtest.analysis`     | `HourMap` (24-bucket expectancy by UTC hour) + `build_all_hour_maps`, `write_hour_maps`, `load_hour_maps`, `summarise_results`, `write_summary_csv` |
| `aera.ml.features`           | `FEATURE_COLUMNS` (15-feature vector: returns × 4 horizons, vol, ATR, RSI, EMA-deviation, wick shapes, sin/cos hour-of-day) + `extract_features` (offline) + `FeatureExtractor` (live rolling-window) + `label_trades` (join-by-timestamp) |
| `aera.ml.model`              | `ProfitabilityClassifier` wrapping `sklearn.ensemble.HistGradientBoostingClassifier` (`predict_proba_win`, `save`, `load`) + `train_model` (walk-forward 80/20 split, returns `TrainReport`) |
| `aera.ml.ensemble`           | `EnsembleClassifier` — per-symbol GBT classifiers + a global fallback; routes by symbol at scoring time, falls back when the per-symbol model is missing. `train_ensemble(min_per_symbol=200)` writes a directory layout (`fallback.joblib` + `per_symbol/*.joblib`) |
| `aera.ml.sequence`           | `SequenceScorer` + `_TransformerWinClassifier` — tiny transformer encoder (~10k params) over the last `seq_len` bars of OHLCV-derived features (`SEQUENCE_FEATURES` — return, log-vol, wicks, body, sin/cos hour). `torch` is an **optional** dependency; if missing, the scorer returns 0.5 (no opinion) and the registry skips it. `train_sequence_model` is walk-forward + per-feature scaler. |
| `aera.ml.rl`                 | `TradingEnv` (Gym-style env over an OHLCV history; HOLD/BUY/SELL; reward = unrealised-PnL delta + realised-PnL bonus on close) + `DQNAgent` (numpy-only MLP Q-network, experience replay, target net, ε-greedy) + `RLScorer` (registry adapter). No new deps required. |
| `aera.ml.registry`           | `ModelRegistry.from_dir(...)` auto-discovers whatever artefacts exist on disk (GBT, ensemble, sequence model, RL policy) and exposes `combined(ctx) -> (P(win), per-scorer breakdown)`. Weighted-average fusion with `FusionWeights`. Each scorer implements `available()` so missing-dep scorers silently degrade. |

### The four scripts

```bash
# 1. fetch_history.py
#    Cache OHLCV for one or more (symbol, resolution) pairs.
#    Incremental: subsequent runs only fetch the missing tail.
python -m scripts.fetch_history \
    --symbols BTCUSD,ETHUSD,SOLUSD \
    --resolutions 1m,5m \
    --days 90

# 2. sweep_backtest.py
#    Cartesian product over strategies × symbols × resolutions ×
#    leverages. Writes summary CSV + trades CSV + hour_maps.json.
python -m scripts.sweep_backtest \
    --strategies delta_perp_scalper,tick_reversal_scalp,micro_vwap_sniper \
    --symbols BTCUSD,ETHUSD,SOLUSD \
    --resolutions 1m,5m \
    --leverages 5,10,25 \
    --bankroll 100 \
    --workers 4

# 3. train_money_printer.py
#    Fits the ProfitabilityClassifier from the sweep's trade list.
#    Walk-forward split (last 20% held out). Writes model.joblib +
#    a training report (accuracy, precision, recall, F1, ROC-AUC,
#    top feature importances).
python -m scripts.train_money_printer \
    --trades-csv data/backtest/sweep_trades.csv \
    --min-trades 200 \
    --output data/money_printer/model.joblib

# 4. backtest.py
#    Single-config sanity backtest for one strategy / symbol /
#    resolution / leverage. Useful while iterating on a strategy.
python -m scripts.backtest \
    --strategy money_printer \
    --symbol BTCUSD --resolution 5m \
    --leverage 10 --bankroll 100
```

### What the ML model sees

The `FEATURE_COLUMNS` list — kept in `aera/ml/features.py` — is
the single source of truth for what the live `MoneyPrinter`
extracts and what the trainer feeds the GBT. Adding / removing a
feature requires retraining; the model's `joblib` payload includes
the feature schema and will refuse to score against a mismatched
vector.

Current 15 features (subject to retraining):

| Feature                  | Why                                            |
| ------------------------ | ---------------------------------------------- |
| `ret_1`, `ret_3`, `ret_5`, `ret_15` | Multi-horizon momentum / mean-reversion proxy |
| `vol_10`, `vol_30`       | Rolling realised volatility                    |
| `atr_pct`                | Average True Range / mid — vol band reference  |
| `rsi_14`                 | Classic overbought / oversold                  |
| `ema_dev_20`             | Mean-reversion distance from EMA               |
| `upper_wick_ratio`, `lower_wick_ratio` | Exhaustion / absorption shapes  |
| `body_ratio`             | Trend-bar vs. doji                             |
| `volume_z`               | Volume spike z-score                           |
| `sin_hour`, `cos_hour`   | Cyclical encoding of UTC hour-of-day           |

### Re-train cadence

* **Always retrain after a strategy logic change.** The model's
  P(win) predictions are conditional on the strategies that
  produced the training trades; changing entry logic invalidates
  the labels.
* **Weekly cadence in steady state.** Re-fetch the rolling tail
  (`fetch_history --days 90` is incremental), re-sweep, re-train.
  The whole pipeline takes ~10 minutes for 3 symbols × 2
  resolutions on a laptop.
* **Restart the bot** to pick up a new model — `MoneyPrinter`
  loads `model.joblib` once at construction.

### Graceful degradation

`MoneyPrinter` is designed to be safe to enable *before* the
artefacts exist:

* No `hour_maps.json` → no time gate; all hours allowed.
* No model artefacts at all → no ML gate; falls back to ATR
  mean-reversion + RSI directional bias (sizing is halved).
* Both missing → just an ATR-band gated mean-reversion scalper
  with ATR-tuned exits.

This lets you flip the strategy on first and let it print baseline
trades while the offline pipeline runs in another terminal.

### Multi-model fusion (the model registry)

Out of the box, MoneyPrinter doesn't depend on any one model — it
loads **whatever it finds** under `data/money_printer/` via
`aera.ml.registry.ModelRegistry.from_dir(...)`. The four supported
scorers and the files that activate them:

| Scorer       | Activates when                                  | Source module           | Extra deps |
| ------------ | ----------------------------------------------- | ----------------------- | ---------- |
| `gbt`        | `model.joblib` is present                       | `aera.ml.model`         | (already in `requirements.txt`) |
| `ensemble`   | `ensemble/` directory exists (`fallback.joblib` + `per_symbol/<SYM>.joblib`) | `aera.ml.ensemble` | none |
| `sequence`   | `sequence_model.pt` + `.meta.json` present and `torch` is importable | `aera.ml.sequence` | `pip install -r requirements-ml-extras.txt` |
| `rl`         | `rl_policy.npz` is present                      | `aera.ml.rl`            | none |

At decision time, every available scorer is asked for its
`P(win)` and the registry returns a **weighted average** (default
weights: `gbt=1.0`, `ensemble=1.5`, `sequence=0.75`, `rl=0.5`). The
strategy gates on that fused score against `win_threshold`.

Two consequences worth knowing:

1. **You can train models incrementally.** Start with the GBT
   alone, train the ensemble later, drop in an RL policy a week
   after that. MoneyPrinter picks them up on next restart with no
   code changes.
2. **Each fire's `metadata['scorers']`** carries the per-scorer
   breakdown (e.g. `{"gbt": 0.62, "ensemble": 0.71, "rl": 0.55}`),
   so the dashboard and trade logs show exactly *why* a fire was
   approved — useful when one scorer goes haywire and you want to
   yank its weight to zero in `FusionWeights`.

#### Training the extra scorers

After `sweep_backtest.py` has produced `data/backtest/sweep_trades.csv`:

```bash
# Per-symbol ensemble (sklearn — fast, no torch).
python -m scripts.train_ensemble \
    --trades-csv data/backtest/sweep_trades.csv \
    --history-dir data/history --resolution 5m \
    --out-dir data/money_printer/ensemble \
    --min-per-symbol 250

# Transformer encoder over the last 64 bars (requires torch).
pip install -r requirements-ml-extras.txt
python -m scripts.train_sequence_model \
    --trades-csv data/backtest/sweep_trades.csv \
    --history-dir data/history --resolution 5m \
    --seq-len 64 --epochs 30 \
    --out data/money_printer/sequence_model.pt

# DQN trading policy on BTC candles (numpy-only).
python -m scripts.train_rl_agent \
    --symbol BTCUSD --resolution 5m \
    --history-dir data/history \
    --episodes 30 \
    --out data/money_printer/rl_policy.npz
```

All three trainers write a `*_report.json` next to the model with
hold-out metrics. Reading them is the fastest way to spot a
broken or under-fit model before it gets fused in live.

---

## Configuration

Three layers, highest priority first:

1. **Environment variables** (`BOT_*`, `DELTA_*`).
2. **`config/config.yaml`** — the canonical project config.
3. **Pydantic defaults** in `aera/settings.py`.

Edit `config/config.yaml` for sticky changes; use `.env` (copied from
`.env.example`) for credentials and per-run overrides. The shipping defaults
target a tiny bankroll (`bankroll: 27`, `leverage: 50`, `trade_size_fraction: 0.5`)
so you can paper-trade end-to-end without thinking about ratios.

---

## Installation

```bash
git clone https://github.com/manishhansal/aera.git && cd aera
python -m venv .venv
# Windows (git-bash): source .venv/Scripts/activate
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then edit with your DELTA_API_* keys if going live
```

Requires Python 3.10+ (3.13 tested).

---

## CLI reference

### `python -m scripts.scan_delta`

Read-only scan of every active Delta perpetual. Prints a table of symbols,
bid/ask/mid, and 24h volume. No credentials needed.

```bash
python -m scripts.scan_delta
python -m scripts.scan_delta --symbols BTCUSD,ETHUSD
```

### `python -m scripts.run_delta`

The main entry point for paper or live trading on Delta. Defaults to paper.

```bash
# Paper-trade with config-file defaults
python -m scripts.run_delta

# Paper-trade with a specific bankroll + dashboard
python -m scripts.run_delta --bankroll 27 --dashboard

# Use the live websocket book feed for sub-second reactions
python -m scripts.run_delta --websocket --dashboard

# LIVE trading (real orders). Requires DELTA_API_KEY + DELTA_API_SECRET.
python -m scripts.run_delta --bankroll 10 --live --duration-seconds 300
```

Flags:

* `--bankroll FLOAT` — starting USDC (overrides config).
* `--strategies LIST` — comma-separated, default `delta_perp_scalper`.
  Available: `delta_perp_scalper`, `order_book_sniper`,
  `tick_reversal_scalp`, `bid_ask_spread_fade`, `flow_scalp`,
  `micro_vwap_sniper`, `stop_hunt_reversal`, `money_printer`.
* `--symbols LIST` — comma-separated, e.g. `BTCUSD,ETHUSD` (override config).
* `--websocket` — use Delta's WS book feed instead of REST polling.
* `--live` — route REAL orders.
* `--duration-seconds N` — auto-stop after N seconds (`0` = forever).
* `--dashboard` — also bring up the web dashboard.
* `--dashboard-host`, `--dashboard-port` — dashboard bind address.

### `python -m scripts.run_dashboard`

Same as `run_delta --dashboard` but with more dashboard-focused flags.

```bash
python -m scripts.run_dashboard --bankroll 27
python -m scripts.run_dashboard --host 0.0.0.0 --port 8080
python -m scripts.run_dashboard --no-engine   # serve UI without the bot
```

### `python -m scripts.simulate_growth`

Pure-math Monte Carlo of bankroll growth under different edge / win-rate /
Kelly assumptions. No internet, no credentials. Useful for sanity-checking
whether a configured per-trade edge is plausibly compatible with a chosen
growth target.

### `python -m scripts.fetch_history`

Fetch and cache OHLCV candles from Delta's `/v2/history/candles`
endpoint into `data/history/<SYMBOL>/<resolution>.parquet` (CSV
fallback if `pyarrow` is missing). Incremental: re-running with
the same `--days` only fetches new bars at the tail.

```bash
python -m scripts.fetch_history --symbols BTCUSD,ETHUSD --resolutions 1m,5m --days 90
python -m scripts.fetch_history --symbols SOLUSD --resolutions 5m --days 180
```

### `python -m scripts.backtest`

Single-configuration backtest. Drives one strategy through one
(symbol, resolution) with a chosen leverage; prints a
`BacktestResult` summary (trades, PnL, Sharpe, max DD, win rate,
profit factor). Useful while iterating on a strategy.

```bash
python -m scripts.backtest --strategy delta_perp_scalper \
    --symbol BTCUSD --resolution 5m --leverage 10 --bankroll 100
```

### `python -m scripts.sweep_backtest`

Cartesian-product grid backtest across strategies × symbols ×
resolutions × leverages, in a `ThreadPoolExecutor`. Outputs:

* `data/backtest/sweep_summary.csv` — ranked leaderboard,
* `data/backtest/sweep_trades.csv` — every trade ever taken,
* `data/money_printer/hour_maps.json` — per-(strategy, symbol)
  hour-of-day expectancy map consumed by `MoneyPrinter`.

```bash
python -m scripts.sweep_backtest \
    --strategies delta_perp_scalper,tick_reversal_scalp,micro_vwap_sniper \
    --symbols BTCUSD,ETHUSD,SOLUSD --resolutions 1m,5m \
    --leverages 5,10,25 --bankroll 100 --workers 4
```

### `python -m scripts.train_money_printer`

Fits the `ProfitabilityClassifier` from the sweep's
`sweep_trades.csv`. Walk-forward 80/20 split, prints a training
report (accuracy, precision, recall, F1, ROC-AUC, top feature
importances), writes `data/money_printer/model.joblib`.

```bash
python -m scripts.train_money_printer \
    --trades-csv data/backtest/sweep_trades.csv \
    --min-trades 200
```

### `python -m scripts.train_ensemble`

Fits the per-symbol ensemble — one
`HistGradientBoostingClassifier` per symbol with at least
`--min-per-symbol` trades, plus a global fallback. Writes
`data/money_printer/ensemble/` (a directory the registry
auto-discovers).

```bash
python -m scripts.train_ensemble \
    --trades-csv data/backtest/sweep_trades.csv \
    --history-dir data/history --resolution 5m \
    --min-per-symbol 250
```

### `python -m scripts.train_sequence_model`

Fits the tiny transformer encoder (`aera.ml.sequence`) over the
last `--seq-len` bars. **Requires torch** (`pip install -r
requirements-ml-extras.txt`). Walk-forward split with per-feature
scaler; writes `data/money_printer/sequence_model.pt` + meta.

```bash
python -m scripts.train_sequence_model \
    --trades-csv data/backtest/sweep_trades.csv \
    --history-dir data/history --resolution 5m \
    --seq-len 64 --epochs 30 --d-model 32 --n-heads 4
```

### `python -m scripts.train_rl_agent`

Trains the DQN trading agent (`aera.ml.rl`) on one symbol's
candle history. Numpy-only; no torch. Writes
`data/money_printer/rl_policy.npz`.

```bash
python -m scripts.train_rl_agent \
    --symbol BTCUSD --resolution 5m \
    --history-dir data/history \
    --episodes 30 --hidden 32
```

---

## Live web dashboard

Brought up by `--dashboard` on any run script, or directly via
`python -m scripts.run_dashboard`. Default URL: `http://127.0.0.1:8787`.

Shows:

* live equity curve (`bankroll`, `locked margin`, `settled wealth`),
* recent fills + signals (executed, pending, rejected),
* open positions with mark-to-market P&L,
* per-strategy stats,
* a top-of-book watchlist,
* pause / resume / halt controls.

Engine and dashboard are decoupled — the engine pushes events into a
`DashboardState` container and the FastAPI app reads snapshots from it, so the
dashboard never reaches into engine internals and a dashboard crash can't
break the bot.

---

## Going live

> **Read this section twice before your first `--live` run.**

1. **Paper-trade for at least a few hours.** Watch the dashboard. Confirm the
   bot opens and closes positions you understand, with the take-profit /
   stop-loss thresholds you've configured.
2. **Use a tiny bankroll.** `BOT_BANKROLL=10` (ten dollars) is a perfectly
   valid first live deployment. The bot compounds; small starting values are
   not a sign of weakness.
3. **Set credentials in `.env`:**
   * `DELTA_API_KEY=...`
   * `DELTA_API_SECRET=...`
   * `DELTA_BASE_URL=...` (defaults to global; use `https://api.india.delta.exchange`
     for India accounts — see comment in `.env.example`).
4. **Confirm leverage.** `DELTA_LEVERAGE=50` (or whatever you set in config)
   will be applied to every configured symbol at startup. The bot's margin
   math assumes this matches the account setting.
5. **Run with `--duration-seconds` first.** `--duration-seconds 300` auto-stops
   the bot after 5 minutes. Use it for the first live run.
6. **Read the dashboard reject reasons.** Most early rejections are config
   mis-matches (per-market exposure tighter than per-trade target, etc.).
   The reject messages tell you exactly what to change.

The bot will refuse to start `--live` if credentials are missing. The
`RiskManager`'s drawdown halt will hard-stop trading if equity falls 25% from
peak — that's a feature, not a bug. The brain's daily-loss circuit-breaker
(default 10%) trips before that for a graceful single-session halt, and a
**time-limited loss-streak cool-down** (default: pause for 5 min after 15
consecutive losses) sits between them. The brain's per-(strategy, symbol)
post-loss cool-down (default 60 s) catches single losing trades before they
escalate into a streak in the first place.

---

## Testing

```bash
python -m pytest -q
```

**397 tests, all green.** They cover portfolio bookkeeping under
leveraged + non-leveraged fills, risk vet logic (including the
reduce-only bypass), executor sizing (target mode + legacy mode,
leverage-aware), the Delta client + signing + websocket, the
mean-reversion scalper (entries, exits, USD and pct TP/SL,
precedence rules, debouncing), the Order Book Sniper (depth-
imbalance + tape confirmation + wall-vanish spoof exit + hold-
timeout + TP/SL priority), the Tick Reversal Scalp (5-tick
exhaustion detection, size decay, spread / news / volume filters,
depth-trend gate, all exit paths), the Bid-Ask Spread Fade (quote
pricing, fee + spread gates, kill switch, inventory skew + cap,
refresh-rate gate, independent BUY/SELL emission), the Micro VWAP
Reversion Sniper (VWAPStream math, volume drop-off ratio,
deviation + spread + hour-skip gates, VWAP snap-back / hard SL /
hold-timeout / USD-PnL exits), the Stop Hunt / Liquidity Grab
Reversal (BarStream rotation + inferred-volume + fractal swing
pivots, wick depth / body ratio / volume / delta gates, partial
TP1 + final TP2 + wick-anchored stop + hold-timeout exits, rearm
debounce), the Greedy autopilot (`fees + $1` TP math, trailing
ratchet, win/loss-streak leverage ramp, compounding fraction,
per-market cap bypass for fresh entries), the Adaptive Brain
(performance gate, soft + hard regime veto, leverage-aware
gross-exposure cap, post-loss cool-down, daily-loss tagging,
phantom-position reconciliation), the new Risk loss-streak
cool-down (vs. permanent halt), the historical `CandleStore`
(Parquet write + incremental dedupe), the `BarReplay` backtest
engine + synthetic 4-tick-per-bar stream, ML feature extraction
+ trade labelling, the `MoneyPrinter` strategy (hour / ATR / ML
gates, ATR-tuned exits, graceful artefact degradation), the
dashboard state container + FastAPI routes, Kelly + compounding
math, and the order book primitives.

---

## Contributing / issues

Bug reports, questions, and PRs are welcome on GitHub:

* Issues: <https://github.com/manishhansal/aera/issues>
* Pull requests: <https://github.com/manishhansal/aera/pulls>

Before opening a PR, please run the test suite locally
(`python -m pytest -q`, 397 tests should pass) and read
[`CONTEXT.md`](./CONTEXT.md) — it documents the signal-to-fill
pipeline, the offline backtest + ML pipeline, the risk-cap
invariants, and the conventions every new strategy must follow.

---

## Disclaimers

This is research / personal trading software. It is **not** financial advice.
Trading leveraged perpetuals is a fast way to lose money. The author assumes
no responsibility for losses, missed gains, exchange API quirks, network
hiccups, or anything else that happens when you push real money through it.
Paper-trade. Read the code. Use small amounts.
