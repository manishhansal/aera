# CONTEXT.md

> **Compact project map for AI assistants and new contributors.**
> The README is the user-facing manual; this file is the navigation
> chart for *editing* the codebase. Read it once before making changes.

---

## 1. What this project is

**`aera`** — *Autonomous Edge & Risk Arbitrage* — is a Python 3.10+
async trading bot that scalps Delta Exchange USD-quoted **perpetual
futures** with seven independently-configurable strategies, a greedy
TP/SL/leverage autopilot, an adaptive performance + regime brain, and
a live FastAPI web dashboard. It supports paper-trading against the
real order book and live trading via Delta's REST/WS API.

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
| Logging          | `rich` console + per-strategy heartbeat logs              |
| CLI              | `typer` (limited), most entry points use stdlib `argparse`|
| Resilience       | `tenacity` retries, `aiolimiter` for rate caps            |
| Tests            | `pytest`, `pytest-asyncio` — **314 tests, all green**     |

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
├── strategies/                    All seven strategies emit Signals (1+ Legs):
│   ├── base.py                    Strategy ABC + Signal + Leg
│   ├── delta_perp_scalper.py      z-score + OFI mean reversion
│   ├── order_book_sniper.py       depth imbalance + tape + wall spoof
│   ├── tick_reversal_scalp.py     5+ tick exhaustion fade
│   ├── bid_ask_spread_fade.py     market-making lite (resting maker quotes)
│   ├── flow_scalp.py              whale-print taker flow continuation
│   ├── micro_vwap_sniper.py       VWAP deviation + volume drop-off fade
│   └── stop_hunt_reversal.py      engineered-wick / liquidity-grab fade
│
├── execution/
│   ├── slippage.py                LinearSlippageModel
│   ├── executor.py                Executor — atomic order router
│   │                              (sizing, risk vet, reduce-only clamp,
│   │                              partial-fill rollback, ExecutionResult)
│   └── delta_exchange.py          DeltaPaperExchange (book-fill sim) +
│                                  DeltaLiveExchange (real REST submit)
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
└── simulate_growth.py             Pure-math Monte Carlo, no internet

config/config.yaml                 Single source of truth for runtime knobs
.env.example                       Credentials template
tests/                             pytest suite (314 tests as of 2026-05)
```

---

## 4. The signal-to-fill pipeline

Every fresh entry goes through this pipeline **in order**. Reduce-only
(closing) legs short-circuit at greedy and brain — they always flow.

```
Strategy.scan(market)            ← microstructure logic per strategy
        │
        ▼
Signal(legs=[Leg, ...])          ← one signal can have 1+ legs
        │
        ▼
GreedyTradeManager.intercept     ← if greedy.enabled: stamps leverage
        │                          override, decides position size,
        │                          (closes are produced separately at
        │                          the top of each scan tick)
        ▼
AdaptiveBrain.gate               ← veto / shrink based on perf, regime,
        │                          daily-loss, gross-exposure
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

**Two halts you must not weaken silently:**

| Halt                      | Default | Where               |
| ------------------------- | ------- | ------------------- |
| Drawdown                  | 25%     | `core/risk.py`      |
| Consecutive losses        | 6       | `core/risk.py`      |
| Brain daily-loss          | 10%     | `core/brain.py`     |
| Brain max strategy streak | 4       | `core/brain.py`     |

The brain's per-strategy streak (4) trips before the global one (6); the
brain's daily-loss (10%) trips before the drawdown halt (25%). That
ordering is intentional: graceful single-strategy mute → graceful daily
pause → hard kill. **Preserve it.**

---

## 7. Strategy authoring conventions

A new strategy should:

1. Subclass `Strategy` from `aera.strategies.base` and implement
   `scan(market) -> list[Signal] | None`. Treat `scan` as the only
   public entry — keep state on `self`.
2. Always check `self.enabled` first (the engine sets this from
   `enabled: <bool>` in YAML).
3. Be **idempotent under repeated scans** when nothing has moved.
   Use `rearm_distance_bps` (or equivalent) to debounce same-symbol
   re-fires.
4. Emit `Leg(reduce_only=True)` for *closes*, never for fresh entries.
   The executor uses this flag to clamp size to the open position.
5. Read live position state via the `Portfolio` reference passed in
   the runner if you need entry-mid / unrealised-PnL — don't recompute
   it from the strategy's own ghost state.
6. Stamp `metadata={"strategy": <self.name>, ...}` on every Leg so the
   brain's per-strategy tracker can attribute fills correctly.
7. Add an entry to the `STRATEGY_REGIME_PREFS` map in `core/brain.py`
   declaring which `Regime`s the strategy is allowed to fire in.
8. Wire it into:
   * `aera/strategies/__init__.py` exports,
   * `scripts/run_delta.py::build_strategies` factory,
   * `scripts/run_dashboard.py::build_delta_strategies`,
   * `aera/settings.py` (a new `*Config` Pydantic model + a field on
     `StrategiesConfig`),
   * `config/config.yaml` (new block under `strategies:` with comments
     explaining every knob and its units).
9. Add a test file `tests/test_<name>.py` covering: entry conditions,
   skip filters, every distinct exit path, and the rearm debounce.

The seven existing strategies are good templates; copy the closest
match (mean-reversion vs. momentum vs. market-making) and adjust.

---

## 8. Common workflows

```bash
# Run the full test suite (must pass before any commit)
python -m pytest -q

# Run a focused test file
python -m pytest tests/test_brain.py -q

# Read-only Delta scan (no credentials)
python -m scripts.scan_delta

# Paper-trade Delta perps + live dashboard
python -m scripts.run_delta --bankroll 27 --dashboard

# Pure-math growth simulator (offline)
python -m scripts.simulate_growth
```

**Live trading**: requires `DELTA_API_KEY` + `DELTA_API_SECRET` in `.env`
and `--live` on the CLI. Always `--duration-seconds 300` on a first
live deploy.

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

| File / pattern              | Why                                                |
| --------------------------- | -------------------------------------------------- |
| `.env`                      | Real API keys live here. Reading is OK if needed; never write or echo. |
| `requirements.txt`          | Dependency edits should be flagged to the user.    |
| `aera/markets/delta_signing.py` | Crypto signing — bugs here corrupt every order. Touch only on explicit request. |
| `core/risk.py` defaults     | Loosening risk caps must be intentional + reviewed.|
| `aera/dashboard/static/*`   | Hand-tuned SPA. Use small targeted edits.          |

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

---

## 12. Where to start reading code

If you're new to the codebase and need to make a change, read in this
order:

1. `aera/__init__.py` — package tagline + module map
2. `config/config.yaml` — every knob the bot exposes, with comments
3. `aera/settings.py` — the typed mirror of the YAML
4. `aera/strategies/base.py` — Strategy / Signal / Leg contract
5. `aera/core/delta_engine.py::DeltaEngine.run` — the main loop
6. `aera/execution/executor.py::Executor.submit` — atomic submission
7. The strategy file closest to your change

Most edits in this codebase are 1–3 files wide. If you find yourself
touching more than five files for a single feature, surface it for
review — that's usually a sign the change is bigger than it looks.

---

*Last updated: 2026-05-28*
