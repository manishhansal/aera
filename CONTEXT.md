# CONTEXT.md

> **Compact project map for AI assistants and new contributors.**
> The README is the user-facing manual; this file is the navigation
> chart for *editing* the codebase. Read it once before making changes.

**Repository:** <https://github.com/manishhansal/aera>
**Default branch:** `master`
**Issues / PRs:** <https://github.com/manishhansal/aera/issues>

---

## 1. What this project is

**`aera`** — *Autonomous Edge & Risk Arbitrage* — is a Python 3.10+
async trading bot that scalps Delta Exchange USD-quoted **perpetual
futures** with **eight** independently-configurable strategies, a
greedy TP/SL/leverage autopilot, an adaptive performance + regime
brain, an offline **backtesting + ML training pipeline**, and a live
FastAPI web dashboard. It supports paper-trading against the real
order book and live trading via Delta's REST/WS API.

The eighth strategy — `MoneyPrinter` — is special: it consumes the
output of the offline pipeline (per-(strategy, symbol) profitable-
hour map + a gradient-boosted profitability classifier) and only
fires when the historical evidence, the live ML score, and a
volatility band all agree. See section 4.5 below.

This is **research / personal trading software**. Real money is on the
line in `--live` mode. Default to safety: prefer reversible changes,
never silently widen a risk cap, and surface assumptions explicitly.

---

## 2. Tech stack at a glance

| Layer            | Library / Tool                                            |
| ---------------- | --------------------------------------------------------- |
| Language         | Python 3.10+ (3.13 tested)                                |
| Async runtime    | `asyncio` (single event loop, all I/O is async)           |
| HTTP             | `httpx` (REST), `websockets` (Delta WS book + trades)     |
| Web / dashboard  | `fastapi`, `uvicorn` (single-page UI in `aera/dashboard/static`)  |
| Config           | `pydantic` v2 + `pydantic-settings`, `pyyaml`, `python-dotenv` |
| Numerics         | `numpy`, `pandas`, `scipy`                                |
| ML / persistence | `scikit-learn` (GBT), `joblib`, `pyarrow` (Parquet cache) |
| Logging          | `rich` console + per-strategy heartbeat logs              |
| CLI              | `typer` (limited), most entry points use stdlib `argparse`|
| Resilience       | `tenacity` retries, `aiolimiter` for rate caps            |
| Tests            | `pytest`, `pytest-asyncio` — **397 tests, all green**     |

Dependencies are pinned by lower-bound only in `requirements.txt`.
Anything new should follow that pattern (`pkg>=X.Y.Z`).

---

## 3. Directory map

```
aera/                              ← installable package
├── __init__.py                    Version + tagline
├── logging.py                     setup_logging() + get_logger()
├── settings.py                    Pydantic config: BotConfig, RiskConfig,
│                                  ExecutionConfig, per-strategy *Config,
│                                  GreedyConfig, BrainConfig, DeltaConfig,
│                                  with env (BOT_*, DELTA_*) + YAML layering
│
├── core/
│   ├── portfolio.py               Cash, locked margin, positions, P&L,
│   │                              drawdown bookkeeping (single source of
│   │                              truth for "what is my live wealth")
│   ├── risk.py                    RiskManager.vet() per-leg gate +
│   │                              kelly_fraction / fractional_kelly_bet
│   ├── compounding.py             simulate_growth Monte Carlo math
│   ├── greedy.py                  GreedyTradeManager — TP/SL/leverage/
│   │                              compounding overlay (sits between strategy
│   │                              fires and the executor)
│   ├── brain.py                   AdaptiveBrain — per-strategy edge
│   │                              tracker, regime-based veto, daily-loss
│   │                              circuit breaker, gross-exposure cap
│   └── delta_engine.py            DeltaEngine — main scan/execute loop;
│                                  exposes pause/resume/stop, listener hooks
│
├── markets/
│   ├── base.py                    Market dataclass + MarketSnapshot + Outcome
│   ├── orderbook.py               OrderBook + Level + best_bid/ask helpers
│   ├── delta.py                   DeltaClient — async REST client (HMAC
│   │                              signing, product discovery, books, orders,
│   │                              positions, change-leverage)
│   ├── delta_signing.py           HMAC-SHA256 signing primitives + WS auth
│   └── delta_ws.py                Async WS book/trade stream
│
├── signals/
│   ├── microstructure.py          RollingZScore + OrderFlowImbalance EMA
│   ├── mean_reversion.py          zscore_signal helper
│   ├── order_book.py              DepthImbalance + TapeInferrer + WallSnapshot
│   ├── tick_stream.py             TickStream — per-mid-change tick log
│   ├── trade_tape.py              TradeTape — rolling whale/avg detection
│   ├── vwap_stream.py             VWAPStream — time-windowed micro-VWAP
│   ├── bar_stream.py              BarStream — 1s OHLC + fractal pivots
│   └── regime.py                  RegimeBook — RANGE / TREND_* / HIGH_VOL /
│                                  NEWS_SPIKE classifier per symbol
│
├── strategies/                    All eight strategies emit Signals (1+ Legs):
│   ├── base.py                    Strategy ABC + Signal + Leg + the
│   │                              ``sync_position_state`` phantom-position
│   │                              reconciler every strategy calls at the
│   │                              top of each per-symbol scan
│   ├── delta_perp_scalper.py      z-score + OFI mean reversion
│   ├── order_book_sniper.py       depth imbalance + tape + wall spoof
│   ├── tick_reversal_scalp.py     5+ tick exhaustion fade
│   ├── bid_ask_spread_fade.py     market-making lite (resting maker quotes)
│   ├── flow_scalp.py              whale-print taker flow continuation
│   ├── micro_vwap_sniper.py       VWAP deviation + volume drop-off fade
│   ├── stop_hunt_reversal.py      engineered-wick / liquidity-grab fade
│   └── money_printer.py           backtest-trained adaptive: hour-of-day
│                                  gate + ML P(win) gate + ATR-band gate +
│                                  ATR-tuned TP/SL exits
│
├── execution/
│   ├── slippage.py                LinearSlippageModel
│   ├── executor.py                Executor — atomic order router
│   │                              (sizing, risk vet, reduce-only clamp,
│   │                              partial-fill rollback, ExecutionResult)
│   └── delta_exchange.py          DeltaPaperExchange (book-fill sim) +
│                                  DeltaLiveExchange (real REST submit)
│
├── data/                          ── HISTORICAL DATA LAYER (new) ──
│   └── history.py                 Candle dataclass + CandleStore
│                                  (Parquet/CSV cache with incremental
│                                  dedupe) + DeltaHistoryClient (paginated
│                                  /v2/history/candles) + fetch_history()
│
├── backtest/                      ── OFFLINE REPLAY + SWEEP (new) ──
│   ├── replay.py                  BarReplay — synthetic 4-ticks-per-bar
│   │                              stream that drives any Strategy
│   │                              through historical OHLCV; produces
│   │                              TradeRecord list + BacktestResult
│   │                              (PnL, Sharpe, max DD, profit factor)
│   ├── sweep.py                   SweepConfig + run_sweep — thread-pool
│   │                              grid search over strategy × symbol ×
│   │                              resolution × leverage. Candles cached
│   │                              per (symbol, resolution) so per-
│   │                              leverage replays share data.
│   └── analysis.py                HourMap — 24-bucket profitability map
│                                  per (strategy, symbol); JSON-persistable
│                                  for the live MoneyPrinter to consume.
│
├── ml/                            ── ML LAYER (new — multi-model fusion) ──
│   ├── features.py                15-feature vector (returns at 4
│   │                              horizons + vol + ATR + RSI + EMA-dev +
│   │                              wick shapes + sin/cos hour-of-day) +
│   │                              FeatureExtractor (live rolling-window)
│   │                              and label_trades() (offline join).
│   ├── model.py                   ProfitabilityClassifier wrapping
│   │                              sklearn HistGradientBoostingClassifier;
│   │                              train_model() does walk-forward split
│   │                              + TrainReport (acc/precision/recall/F1/
│   │                              ROC-AUC + feature importance).
│   ├── ensemble.py                EnsembleClassifier — per-symbol GBT +
│   │                              global fallback; train_ensemble() routes
│   │                              by symbol at scoring time, falls back
│   │                              when per-symbol model is missing.
│   ├── sequence.py                SequenceScorer — torch-optional tiny
│   │                              transformer encoder (~10k params) over
│   │                              the last seq_len bars of OHLCV-derived
│   │                              SEQUENCE_FEATURES. Returns 0.5 (no
│   │                              opinion) when torch isn't installed.
│   ├── rl.py                      TradingEnv (Gym-style env over OHLCV) +
│   │                              DQNAgent (numpy-only MLP Q-network with
│   │                              experience replay + target net) +
│   │                              RLScorer registry adapter. No new deps.
│   └── registry.py                ModelRegistry.from_dir(...) auto-loads
│                                  whatever artefacts exist on disk and
│                                  fuses them via weighted average
│                                  (FusionWeights). Each scorer implements
│                                  available() for graceful degradation.
│
└── dashboard/
    ├── server.py                  FastAPI app + WS push + uvicorn runner
    ├── state.py                   DashboardState (in-memory event ring
    │                              buffers, equity curve points, snapshots)
    └── static/                    index.html + style.css + app.js (SPA)

scripts/
├── scan_delta.py                  Read-only product/order-book scanner
├── run_delta.py                   Main entry point (paper or --live)
├── run_dashboard.py               run_delta + dashboard wrapper
├── simulate_growth.py             Pure-math Monte Carlo, no internet
├── fetch_history.py               Cache N days of Delta /v2/history/candles
│                                  into data/history/<SYM>/<res>.parquet
├── backtest.py                    Single-config backtest (one strategy /
│                                  symbol / resolution / leverage)
├── sweep_backtest.py              Full grid sweep; writes
│                                  data/backtest/sweep_summary.csv +
│                                  sweep_trades.csv +
│                                  data/money_printer/hour_maps.json
├── train_money_printer.py         Fits ProfitabilityClassifier from the
│                                  sweep's trade list; writes
│                                  data/money_printer/model.joblib
├── train_ensemble.py              Fits per-symbol ensemble; writes
│                                  data/money_printer/ensemble/
├── train_sequence_model.py        Fits torch transformer encoder; writes
│                                  data/money_printer/sequence_model.pt
│                                  (requires torch — pip install
│                                  -r requirements-ml-extras.txt)
└── train_rl_agent.py              Trains numpy DQN on candle history;
                                   writes data/money_printer/rl_policy.npz

config/config.yaml                 Single source of truth for runtime knobs
.env.example                       Credentials template
requirements-ml-extras.txt         Optional deep-learning extras (torch)
data/                              On-disk caches (ignored by git +
                                   cursorignore; created on first run):
  history/<SYM>/<res>.parquet      Cached OHLCV
  backtest/sweep_*.csv             Sweep outputs
  money_printer/{hour_maps.json,   MoneyPrinter live-trade artefacts —
                 model.joblib,     auto-discovered by ModelRegistry.
                 ensemble/,
                 sequence_model.pt,
                 rl_policy.npz}
tests/                             pytest suite (397 tests as of 2026-05)
```

---

## 4. The signal-to-fill pipeline

Every fresh entry goes through this pipeline **in order**. Reduce-only
(closing) legs short-circuit at greedy and brain — they always flow.

```
Strategy.scan(market)            ← microstructure logic per strategy
        │                          (first thing every per-symbol scan
        │                          does: sync_position_state(...) — see
        │                          §6.1 "phantom-position guard")
        ▼
Signal(legs=[Leg, ...])          ← one signal can have 1+ legs
        │
        ▼
GreedyTradeManager.intercept     ← if greedy.enabled: stamps leverage
        │                          override, decides position size,
        │                          (closes are produced separately at
        │                          the top of each scan tick)
        ▼
AdaptiveBrain.gate               ← five vetoes (mute, regime,
        │                          post-loss cool-down, daily-loss,
        │                          gross-exposure) + a size shrinker.
        │                          Every vetoed signal is TAGGED with
        │                          ``metadata["brain_veto_reason"]`` so
        │                          the dashboard surfaces it.
        ▼
Executor.submit                  ← RiskManager.vet, sizing to
        │                          trade_size_fraction × buying_power,
        │                          clamp reduce_only, atomic submit
        ▼
Exchange.submit (paper or live)  ← fills go back through Portfolio
        │
        ▼
Portfolio.apply_fill             ← updates cash, position, realised PnL
        │
        ▼
DashboardState.record_*          ← all listeners invoked off the
                                   DeltaEngine event hooks
```

**Critical invariant:** `reduce_only=True` legs **must always be allowed
through** — they shrink exposure rather than open it. The brain, greedy
manager, risk vetter, and executor each respect this. Don't add a new
gate without preserving the bypass.

**Dashboard surfacing:** vetoed signals are emitted to the
dashboard's ``on_signals`` listener BEFORE the brain drops them,
then individually marked rejected via a synthetic ExecutionResult
with the brain's veto reason. The user sees ``brain: post-loss
cool-down on BTCUSD (47s left)`` rather than the silent drop the
original implementation did. Don't move ``record_signals`` back
past the brain filter — vetoed entries would vanish again.

---

## 4.5 The offline backtest → ML → live-trade pipeline

A separate pipeline produces the artefacts the
``MoneyPrinter`` strategy reads at live-trade time:

```
                       ┌──────────────────┐  REST  ┌────────────────────┐
                       │ fetch_history.py │───────▶│  Delta /candles    │
                       └────────┬─────────┘        └────────────────────┘
                                ▼
                     ┌─────────────────────┐
                     │ data/history/*.parq │  CandleStore (incremental cache)
                     └────────┬────────────┘
                              ▼
               ┌──────────────────────────────────┐
               │ sweep_backtest.py                │
               │  ┌────────────────────────────┐  │
               │  │ BarReplay × every strategy │  │  ← reuses production strategies
               │  │ × symbol × resolution      │  │
               │  │ × leverage                 │  │
               │  └─────────────┬──────────────┘  │
               └────────────────┼─────────────────┘
                                ▼
       data/backtest/sweep_summary.csv  data/backtest/sweep_trades.csv
                                 │
                ┌────────────────┘
                ▼
        ┌────────────────────────┐       ┌────────────────────────────┐
        │ train_money_printer.py │──────▶│ data/money_printer/        │
        │ train_ensemble.py      │       │   hour_maps.json           │
        │ train_sequence_model.py│       │   model.joblib             │
        │ train_rl_agent.py      │       │   ensemble/                │
        │  • label trades        │       │   sequence_model.pt (.meta)│
        │  • walk-forward split  │       │   rl_policy.npz            │
        │  • GBT / Ensemble /    │       └────────────┬───────────────┘
        │    Transformer / DQN   │                    │ loaded on
        └────────────────────────┘                    │ MoneyPrinter
                                                      ▼ construction via
                                                        ModelRegistry.from_dir
                                          ┌─────────────────────────┐
                                          │ MoneyPrinter live       │
                                          │  • hour gate            │
                                          │  • Registry.combined()  │
                                          │    fused P(win) ≥ thr   │
                                          │  • ATR band check       │
                                          │  • ATR-tuned TP/SL      │
                                          └─────────────────────────┘
```

Important properties:

* **The sweep uses the SAME strategies the live runner uses.**
  ``BarReplay`` calls ``strategy.scan(market)`` against synthetic
  4-ticks-per-bar markets built from OHLCV; no parallel
  re-implementation. If a strategy is broken in live trading, it's
  broken in the sweep too — and vice versa.
* **Reuse the production ``Portfolio``.** ``BarReplay`` constructs
  a fresh ``Portfolio(bankroll=...)`` per replay so PnL math is
  byte-identical to live (fees, slippage, leverage included).
* **Walk-forward only.** ``train_model`` sorts by ``entry_ts``
  and holds out the last 20% — never random-shuffle a time-series
  split, you'll leak the future into training.
* **Models are auto-discovered.** The strategy doesn't hard-code
  which model to load — ``aera.ml.registry.ModelRegistry.from_dir``
  probes the models directory for ``model.joblib``, ``ensemble/``,
  ``sequence_model.pt``, ``rl_policy.npz`` and fuses whatever it
  finds via weighted average. Drop a new artefact in, restart the
  bot, and it's live in the loop.
* **MoneyPrinter degrades gracefully.** Missing hour map → no
  time gate, every hour allowed. No models at all → no ML gate,
  fall back to ATR mean-reversion + RSI bias (half-sized). So you can flip it on
  before training finishes; it just trades less precisely until
  the artefacts land.
* **The pipeline is ENTIRELY offline.** ``data/history/`` and
  ``data/money_printer/`` are in ``.cursorignore`` and
  ``.gitignore``; never check them in. Re-fetch is cheap because
  ``CandleStore`` is incremental.

See section 8 for the exact commands; see section 7.5 for the
authoring contract MoneyPrinter follows.

---

## 5. Configuration model

Three layers, highest priority first:

1. **Environment variables** (`BOT_*`, `DELTA_*`) — `.env` is loaded
   via `python-dotenv` at process start.
2. **`config/config.yaml`** — the canonical project config.
3. **Pydantic defaults** in `aera/settings.py`.

`get_settings()` (cached via `lru_cache`) returns the merged result.
**Never read env vars directly inside business logic** — go through
the settings models so overrides remain one-shot per process.

When adding a new tunable:
1. Add the field with a default to the matching `*Config` Pydantic model
   in `aera/settings.py`.
2. Document it (with units + invariants) in `config/config.yaml`.
3. If it's user-facing, add a `BOT_*` env override path in
   `_apply_env_overrides` in `settings.py`.
4. Add a startup `log.warning` for any cross-field invariant violations
   (e.g. `max_market_exposure < max_trade_fraction`) — see existing
   examples in `get_settings()`.

---

## 6. Risk model (read this before touching `core/`)

**Buying power formula** (used everywhere sizing happens):

```
buying_power = bankroll × leverage
```

* `risk.trade_size_fraction` — TARGET size for the largest leg of each
  signal, as a fraction of buying power. `0.5` = aim for a leg of size
  `0.5 × bankroll × leverage`.
* `risk.max_trade_fraction` — CEILING. Must be ≥ `trade_size_fraction`.
* `risk.max_market_exposure` — Per-symbol cumulative cap on existing +
  new notional. Must be ≥ `max_trade_fraction`, otherwise every fresh
  single-leg signal gets vet-rejected on its first attempt.

The startup warning in `settings.get_settings()` flags violations of
either invariant. Don't paper over the warning by lowering one knob —
fix the user-facing config or push a sensible default.

**Halts and cool-downs you must not weaken silently:**

| Halt                          | Default        | Where               | Kind        |
| ----------------------------- | -------------- | ------------------- | ----------- |
| Drawdown                      | 25%            | `core/risk.py`      | hard kill   |
| Consecutive losses → cool-down| 15 trades / 300s | `core/risk.py`    | time-limited|
| Brain daily-loss              | 10%            | `core/brain.py`     | daily       |
| Brain max strategy streak     | 2 → mute       | `core/brain.py`     | strategy    |
| Brain post-loss cool-down     | 60s / (strat,sym)| `core/brain.py`   | per pair    |

The intended trip order:
1. **Brain post-loss cool-down (60s)** — single losing trade →
   that (strategy, symbol) pair waits 60s before re-firing.
   Quiet revenge-trade suppressor.
2. **Brain per-strategy mute (2 losses)** — full strategy goes
   on probation for `mute_seconds`, returns at half size.
3. **Brain daily-loss cap (10%)** — every strategy stops opening
   fresh entries for the rest of the rolling 24h window.
4. **Risk loss-streak cool-down (15 trades → 300s)** — bot-wide
   pause; resumes automatically. Was a permanent halt in earlier
   versions; intentionally softened because scalpers can take
   double-digit losing streaks while still being net-positive.
5. **Risk drawdown halt (25%)** — hard kill. Only `resume()` (manual
   from dashboard) clears it.

Graceful per-pair → per-strategy → portfolio → kill ordering is
intentional. **Preserve it** when adding new safeties — slot them
between the existing levels, don't replace one.

---

## 7. Strategy authoring conventions

A new strategy should:

1. Subclass `Strategy` from `aera.strategies.base` and implement
   `scan(market) -> list[Signal] | None`. Treat `scan` as the only
   public entry — keep state on `self`.
2. Always check `self.enabled` first (the engine sets this from
   `enabled: <bool>` in YAML).
3. **Call `self.sync_position_state(state, self.portfolio, market_id,
   outcome_id)` at the very top of every per-symbol pass.** This
   reconciles the strategy's internal "I have an open position"
   bookkeeping against the actual `Portfolio` — without it, a
   stop-out flattened by greedy or the brain leaves the strategy
   convinced it still holds, and you'll see floods of "no open
   position to close" rejects. All eight existing live strategies
   do this; copy the pattern.
4. Be **idempotent under repeated scans** when nothing has moved.
   Use `rearm_distance_bps` (or equivalent) to debounce same-symbol
   re-fires.
5. Emit `Leg(reduce_only=True)` for *closes*, never for fresh entries.
   The executor uses this flag to clamp size to the open position.
6. Read live position state via the `Portfolio` reference passed in
   the runner if you need entry-mid / unrealised-PnL — don't recompute
   it from the strategy's own ghost state.
7. Stamp `metadata={"strategy": <self.name>, ...}` on every Leg so the
   brain's per-strategy tracker can attribute fills correctly.
8. Add an entry to the `STRATEGY_REGIME_PREFS` map in `core/brain.py`
   declaring which `Regime`s the strategy is allowed to fire in.
   ``MoneyPrinter`` is regime-agnostic (declared with all five
   regimes allowed) because its hour-map + ML score already encode
   regime context implicitly.
9. Wire it into:
   * `aera/strategies/__init__.py` exports,
   * `scripts/run_delta.py::build_strategies` factory,
   * `scripts/run_dashboard.py::build_delta_strategies`,
   * `aera/settings.py` (a new `*Config` Pydantic model + a field on
     `StrategiesConfig`),
   * `config/config.yaml` (new block under `strategies:` with comments
     explaining every knob and its units).
10. Add a test file `tests/test_<name>.py` covering: entry conditions,
    skip filters, every distinct exit path, the rearm debounce, AND
    a phantom-position scenario where the portfolio is flat but the
    strategy thinks it's long/short — `sync_position_state` must
    reset it without emitting a signal.

The eight existing strategies are good templates; copy the closest
match (mean-reversion vs. momentum vs. market-making vs.
backtest-driven adaptive) and adjust.

---

## 7.5 The MoneyPrinter contract

`MoneyPrinter` is the only strategy that depends on offline
artefacts. Its construction order matters:

1. **Always optional.** If `data/money_printer/hour_maps.json` or
   any model artefact is missing, MoneyPrinter must still scan,
   just without the corresponding gate. Lazy-load on first scan,
   log exactly once at INFO when an artefact is missing.
2. **Symbol-by-symbol state.** Maintain a `_SymbolState` per
   `market_id` with a rolling deque of `_Bar` objects (default
   200 bars). Aggregate the synthetic tick stream the engine
   feeds you — DO NOT call the historical fetcher at live-trade
   time.
3. **The three gates fire in this order:**
   - hour-of-day (cheapest): skip if `HourMap.lookup(strategy,
     symbol, hour_utc).expectancy_usd ≤ 0`,
   - ATR band: skip if ATR% outside
     `[min_atr_pct, max_atr_pct]`,
   - ML P(win): skip if `ModelRegistry.combined(ctx).fused <
     win_threshold` (default 0.55).
4. **Confidence sizes the trade.** Once all gates pass, the
   strategy emits a `Leg` whose `metadata["confidence"]` is the
   fused ML score (or 0.5 fallback when no models are loaded). The
   executor reads this and scales notional.
5. **Exits are ATR-tuned.** TP = `entry × (1 + tp_atr_mult × atr_pct)`,
   SL = `entry × (1 − sl_atr_mult × atr_pct)`, plus a
   `max_hold_seconds` safety. No reliance on greedy.
6. **Retraining cadence.** Re-run `sweep_backtest.py` +
   `train_money_printer.py` (and optionally `train_ensemble.py` /
   `train_sequence_model.py` / `train_rl_agent.py`) weekly (or
   after any strategy change). Models are loaded once at strategy
   construction; restart the bot to pick up new artefacts.

### 7.5.1 The multi-model registry

MoneyPrinter does NOT call a single classifier — it goes through
`aera.ml.registry.ModelRegistry` which auto-discovers any
combination of these scorers on disk:

| Disk artefact (under `models_dir`) | Scorer name | Module             | Default weight |
| ---------------------------------- | ----------- | ------------------ | -------------- |
| `model.joblib`                     | `gbt`       | `aera.ml.model`    | `1.0`          |
| `ensemble/`                        | `ensemble`  | `aera.ml.ensemble` | `1.5`          |
| `sequence_model.pt` (+ `.meta.json`) | `sequence`| `aera.ml.sequence` | `0.75` *       |
| `rl_policy.npz`                    | `rl`        | `aera.ml.rl`       | `0.5`          |

\* requires `torch`; silently skipped when not installed.

The registry's `combined(ctx)` returns a fused `P(win)` as the
weighted arithmetic mean of the available scorers' outputs and
also returns the per-scorer breakdown — which MoneyPrinter copies
into `metadata["scorers"]` on every fire so the dashboard and
trade-log show *why* the trade was approved.

**Adding a new scorer is two changes**: subclass `Scorer` in
`aera/ml/registry.py` with a `score(ctx) -> float` method, and add
a probe in `ModelRegistry.from_dir`. MoneyPrinter needs no edits.

Tests live in `tests/test_money_printer.py` (gate combinations
+ phantom-position), `tests/test_ml_ensemble.py` (per-symbol
routing + fallback), `tests/test_ml_registry.py` (fusion math +
auto-discovery), `tests/test_ml_rl.py` (env semantics + DQN smoke
+ scorer round-trip), and `tests/test_ml_sequence.py` (always-on
graceful-degradation tests + torch-only training tests).

---

## 8. Common workflows

### Day-to-day

```bash
# Run the full test suite (must pass before any commit)
python -m pytest -q

# Run a focused test file
python -m pytest tests/test_brain.py -q
python -m pytest tests/test_money_printer.py -q

# Read-only Delta scan (no credentials)
python -m scripts.scan_delta

# Paper-trade Delta perps + live dashboard
python -m scripts.run_delta --bankroll 27 --dashboard

# Pure-math growth simulator (offline)
python -m scripts.simulate_growth
```

### Train MoneyPrinter (offline pipeline)

Run these in order; each step caches its output, so you can re-run
any one step without redoing the others.

```bash
# 1. Cache OHLCV for the symbols you care about (run once per week).
#    --days 90 is enough for a first model; bump to 180+ for richer
#    hour-of-day signal.
python -m scripts.fetch_history \
    --symbols BTCUSD,ETHUSD,SOLUSD \
    --resolutions 1m,5m \
    --days 90

# 2. Sweep every strategy × symbol × resolution × leverage.
#    Outputs: data/backtest/sweep_summary.csv, sweep_trades.csv,
#             data/money_printer/hour_maps.json
python -m scripts.sweep_backtest \
    --strategies delta_perp_scalper,tick_reversal_scalp,micro_vwap_sniper \
    --symbols BTCUSD,ETHUSD,SOLUSD \
    --resolutions 1m,5m \
    --leverages 5,10,25

# 3. Train the profitability classifier from the sweep's trade list.
#    Outputs: data/money_printer/model.joblib + a training report.
python -m scripts.train_money_printer \
    --min-trades 200 \
    --min-win-rate 0.45

# 3a. (Optional) Per-symbol ensemble — sklearn only, no torch.
python -m scripts.train_ensemble \
    --trades-csv data/backtest/sweep_trades.csv \
    --history-dir data/history --resolution 5m \
    --min-per-symbol 250

# 3b. (Optional) Transformer encoder over the last 64 bars.
#     pip install -r requirements-ml-extras.txt   (one-time torch install)
python -m scripts.train_sequence_model \
    --trades-csv data/backtest/sweep_trades.csv \
    --history-dir data/history --resolution 5m \
    --seq-len 64 --epochs 30

# 3c. (Optional) Numpy DQN trading policy on one symbol's candles.
python -m scripts.train_rl_agent \
    --symbol BTCUSD --resolution 5m \
    --episodes 30 --hidden 32

# 4. Single-config sanity backtest (no grid; useful while iterating
#    on a strategy).
python -m scripts.backtest \
    --strategy money_printer --symbol BTCUSD \
    --resolution 5m --leverage 10 --bankroll 100

# 5. Flip MoneyPrinter on for live (paper) trading.
#    config/config.yaml -> strategies.money_printer.enabled: true
#    MoneyPrinter auto-discovers whichever of the above 4 models
#    you produced; no config change needed when you add another.
python -m scripts.run_delta --bankroll 27 --dashboard \
    --strategies money_printer
```

**Re-run cadence:** the full 1→3 pipeline takes 5–15 min on a laptop
for 3 symbols × 90 days × 2 resolutions. Recommended weekly, or
after any strategy change that affects entry / exit logic.

### Going live

**Live trading**: requires `DELTA_API_KEY` + `DELTA_API_SECRET` in `.env`
and `--live` on the CLI. Always `--duration-seconds 300` on a first
live deploy. Do NOT enable MoneyPrinter on `--live` until you've
inspected `data/money_printer/hour_maps.json` and the training
report — the strategy will compound aggressively if its ML score
agrees with its hour map.

---

## 9. Coding style + house rules

* `from __future__ import annotations` at the top of every module —
  the codebase uses postponed evaluation for forward refs.
* Type-annotate public functions and dataclasses; internal helpers can
  skip if obvious.
* Prefer `dataclass` over manual `__init__` for plain records;
  `pydantic.BaseModel` for things loaded from config.
* Pure functions where feasible. Effectful code (network, file I/O,
  exchange writes) lives in `markets/`, `execution/`, `dashboard/`.
* No bare `except:` — always name the exception class.
* All exchange-facing async ops must be cancel-safe. Use
  `asyncio.shield` only around critical close paths.
* Logger lines: leading namespace prefix (the module logger does this
  automatically), short imperative phrasing, include the symbol when
  relevant. The dashboard heartbeat formatter expects this.
* **No comments that just narrate the next line.** Comments explain
  *why*, not *what* (existing files follow this — match the tone).
* Run `python -m pytest -q` before declaring a change done.

---

## 10. Files an AI assistant should NEVER touch without explicit ask

| File / pattern                      | Why                                                |
| ----------------------------------- | -------------------------------------------------- |
| `.env`                              | Real API keys live here. Reading is OK if needed; never write or echo. |
| `requirements.txt`                  | Dependency edits should be flagged to the user.    |
| `aera/markets/delta_signing.py`     | Crypto signing — bugs here corrupt every order. Touch only on explicit request. |
| `core/risk.py` defaults             | Loosening risk caps must be intentional + reviewed.|
| `aera/dashboard/static/*`           | Hand-tuned SPA. Use small targeted edits.          |
| `data/history/**`, `data/backtest/**`, `data/money_printer/**` | Generated artefacts (OHLCV cache, sweep CSVs, hour maps, GBT model, ensemble, sequence_model.pt, rl_policy.npz). Re-create with the scripts; never hand-edit. |

---

## 11. Glossary (jargon used in the code)

| Term              | Meaning in this codebase                                           |
| ----------------- | ------------------------------------------------------------------ |
| `bps`             | Basis points = 0.01%. `5 bps = 0.05% = 0.0005`                     |
| `mid`             | `(best_bid + best_ask) / 2`                                        |
| `OFI`             | Order Flow Imbalance — EMA of signed top-of-book size deltas       |
| `taker / maker`   | Aggressive vs. resting order; Delta charges different fees per side|
| `reduce_only`     | A leg that can only shrink an existing position (close, not open)  |
| `notional`        | `price × contract_size × qty` — the dollar size of the position    |
| `buying_power`    | `bankroll × leverage` (see Risk model)                             |
| `wick`            | The high − close (or open − low) part of an OHLC bar               |
| `regime`          | RANGE / TREND_UP / TREND_DOWN / HIGH_VOL / NEWS_SPIKE classification|
| `probation`       | Brain state: muted strategy returned at half-size pending recovery |
| `ATR%`            | Average True Range / mid price — volatility band MoneyPrinter gates on |
| `hour_map`        | 24-bucket JSON of historical (strategy, symbol, hour_utc) → expectancy_usd, used by MoneyPrinter |
| `walk-forward`    | Time-ordered train/test split — older bars train, newest bars test |
| `P(win)`          | The fused win-probability the registry returns from all available scorers |
| `sweep`           | A grid-search backtest over (strategy × symbol × resolution × leverage) |
| `Scorer`          | A model wrapper that returns `P(win)` for a `ScoringContext`. Registered in `ModelRegistry` |
| `FusionWeights`   | Per-scorer weights for the registry's weighted-average fusion (`gbt`, `ensemble`, `sequence`, `rl`) |
| `Ensemble`        | Per-symbol GBT classifiers + a global fallback (`aera.ml.ensemble`) |
| `SequenceScorer`  | torch-optional tiny transformer encoder over the last seq_len bars (`aera.ml.sequence`) |
| `DQNAgent`        | Numpy-only deep-Q-network trading agent (`aera.ml.rl`); reward = unrealised-PnL delta + realised-PnL bonus on close |

---

## 12. Where to start reading code

If you're new to the codebase and need to make a change, read in this
order:

1. `aera/__init__.py` — package tagline + module map
2. `config/config.yaml` — every knob the bot exposes, with comments
3. `aera/settings.py` — the typed mirror of the YAML
4. `aera/strategies/base.py` — Strategy / Signal / Leg contract +
   `sync_position_state` reconciler
5. `aera/core/delta_engine.py::DeltaEngine.run` — the main loop
6. `aera/core/brain.py::AdaptiveBrain.filter_signals` — every veto
   and size-shrink (the bot's de-risking nervous system)
7. `aera/execution/executor.py::Executor.submit` — atomic submission
8. The strategy file closest to your change
9. `aera/ml/registry.py::ModelRegistry.from_dir` — how MoneyPrinter
   auto-loads and fuses scorers; subclass `Scorer` here to add a
   new ML modality (no MoneyPrinter edits required)

For the offline / ML pipeline, additionally read in this order:

1. `aera/data/history.py::CandleStore` — the on-disk cache contract
2. `aera/backtest/replay.py::BarReplay.run` — how a Strategy is
   driven by historical OHLCV
3. `aera/backtest/sweep.py::run_sweep` — the grid-search orchestration
4. `aera/ml/features.py` — what the model sees (FEATURE_COLUMNS)
5. `aera/ml/model.py::train_model` — walk-forward training + GBT
6. `aera/strategies/money_printer.py::MoneyPrinter.scan` — how
   those artefacts are consumed live

Most edits in this codebase are 1–3 files wide. If you find yourself
touching more than five files for a single feature, surface it for
review — that's usually a sign the change is bigger than it looks.

---

*Last updated: 2026-05-28*
