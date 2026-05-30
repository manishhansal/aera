"""MoneyPrinter — backtest-trained adaptive strategy.

This is the consolidated "what we learned from the sweep" strategy.
It refuses to fire unless three conditions agree:

1. **Hour-of-day gate** — the (strategy, symbol) was profitable in
   this UTC hour during backtests. Loaded from
   ``data/money_printer/hour_maps.json`` produced by
   ``scripts/sweep_backtest.py``.
2. **ML probability gate** — the trained
   :class:`~aera.ml.model.ProfitabilityClassifier` reports
   ``P(win | current_features) >= win_threshold``. Loaded from
   ``data/money_printer/model.joblib`` produced by
   ``scripts/train_money_printer.py``.
3. **Volatility band** — entry rejected if recent ATR is below
   ``min_atr_pct`` (too quiet — fees will dominate) or above
   ``max_atr_pct`` (regime mismatch — most moves run past our TP).

Sizing
------

The trade's notional is the smaller of:

* ``notional_usd`` (config), and
* ``size_mult × bankroll × leverage`` where ``size_mult`` is
  proportional to ``(P(win) - win_threshold)``. The more confident
  the model, the larger the fire — but always under the
  per-trade cap.

Exit
----

ATR-tuned TP and SL — TP = ``tp_atr_mult × ATR``, SL =
``sl_atr_mult × ATR``. This adapts the exit envelope to current
volatility rather than using a fixed bps band that becomes too
tight in quiet hours and too loose in busy ones.

Feature collection runs off a rolling synthetic-bar window built
from the tick mid-price stream. The strategy holds its own bar
aggregator so it works regardless of which engine drives it.

When the model file is missing the strategy auto-degrades to
"hour gate only" (still tradeable, just less precise). When the
hour map is missing too, every fire is allowed — the strategy
behaves like a vanilla z-score reversion until training output
is dropped into place.
"""
from __future__ import annotations

import math
import os
import time as _time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, Iterable, List, Optional

from aera.logging import get_logger
from aera.markets import DELTA_OUTCOME_LABEL, Market

from .base import Leg, Signal, Strategy

if TYPE_CHECKING:
    from aera.core import Portfolio
    from aera.ml import ProfitabilityClassifier
    from aera.ml.registry import ModelRegistry
    from aera.backtest.analysis import HourMap


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# rolling bar aggregator (synthetic from tick mids)
# ---------------------------------------------------------------------------


@dataclass
class _Bar:
    ts_open: int
    ts_close: int
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    started: bool = False

    def reset(self, ts_open: int, ts_close: int, price: float) -> None:
        self.ts_open = ts_open
        self.ts_close = ts_close
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = 0.0
        self.started = True

    def update(self, price: float, qty: float = 0.0) -> None:
        if not self.started:
            self.open = price
            self.high = price
            self.low = price
            self.started = True
        else:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
        self.close = price
        self.volume += qty


@dataclass
class _SymbolState:
    """Per-symbol live state — rolling bars + open-position tracking."""

    bar: _Bar = field(default_factory=lambda: _Bar(0, 0))
    history: Deque[tuple] = field(default_factory=lambda: deque(maxlen=200))
    # Cached most-recent feature row. Recomputed only on bar close
    # because every feature is bar-derived (returns, RSI, EMA-dev,
    # ATR%, hour-of-day) — recomputing on every tick is wasteful and
    # was the main cost driver of the backtest sweep (172k ticks ×
    # 200-row DataFrame construction = minutes per replay).
    features_at_bars: int = -1
    features_row: object = None  # pandas Series, or None
    # Open position tracking (mirrors other strategies for the
    # ``sync_position_state`` helper on Strategy).
    position_side: Optional[str] = None
    entry_mid: float = 0.0
    entry_size_usd: float = 0.0
    entry_atr: float = 0.0
    entry_time: float = 0.0


# ---------------------------------------------------------------------------
# MoneyPrinter strategy
# ---------------------------------------------------------------------------


class MoneyPrinter(Strategy):
    """Backtest-trained adaptive money-printer.

    Parameters
    ----------
    bar_seconds : int
        Aggregation window for the rolling synthetic bars. Default
        60s matches the 1-minute resolution typically used for
        offline training; switch the trainer + this value together.
    feature_window_bars : int
        Number of bars kept in memory. Feature extraction needs ~60
        warm-up bars (the ``ret_60`` column); the default 200 lets
        the most stale features still build cleanly.
    win_threshold : float
        Minimum ``P(win)`` for the model gate. Higher = stricter,
        fewer fires, higher precision. 0.55 is a sensible starting
        point — well above coin-flip but not so strict that the
        strategy never fires on a small training sample.
    hour_map_path : str
        JSON file produced by ``scripts/sweep_backtest.py``.
    model_path : str
        ``joblib`` file produced by ``scripts/train_money_printer.py``.
    min_atr_pct / max_atr_pct : float
        Volatility band (% of close). Fires only inside.
    tp_atr_mult / sl_atr_mult : float
        Exit envelope in multiples of current ATR.
    max_hold_seconds : float
        Force-flatten after this long even if neither TP nor SL hit.
    notional_usd : float
        Per-trade USD notional ceiling (executor/risk caps still apply).
    leverage_hint : float
        Leverage to stamp on the leg. The executor still consults
        venue caps.
    portfolio : Portfolio, optional
        Live portfolio for the ``sync_position_state`` helper.
    """

    name = "money_printer"

    _DEFAULT_MODEL_PATH = "data/money_printer/model.joblib"
    _DEFAULT_MODELS_DIR = "data/money_printer"

    def __init__(
        self,
        *,
        bar_seconds: int = 60,
        feature_window_bars: int = 200,
        win_threshold: float = 0.55,
        hour_map_path: str = "data/money_printer/hour_maps.json",
        model_path: str = _DEFAULT_MODEL_PATH,
        models_dir: Optional[str] = None,
        min_atr_pct: float = 0.0005,   # 5 bps / bar — anything quieter is mostly fees
        max_atr_pct: float = 0.03,     # 3% / bar — anything wilder is news
        tp_atr_mult: float = 1.5,
        sl_atr_mult: float = 1.0,
        max_hold_seconds: float = 1800.0,   # 30 min
        notional_usd: float = 100.0,
        leverage_hint: float = 25.0,
        portfolio: Optional["Portfolio"] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.bar_seconds = int(bar_seconds)
        self.feature_window_bars = int(feature_window_bars)
        self.win_threshold = float(win_threshold)
        self.hour_map_path = Path(hour_map_path)
        self.model_path = Path(model_path)
        # If the caller overrode model_path but not models_dir, infer
        # the models directory from the model_path's parent — this
        # keeps test fixtures pointing at a tmp_path safe from picking
        # up the real data/ tree.
        if models_dir is None:
            if str(model_path) != self._DEFAULT_MODEL_PATH:
                models_dir = str(self.model_path.parent)
            else:
                models_dir = self._DEFAULT_MODELS_DIR
        self.models_dir = Path(models_dir)
        self.min_atr_pct = float(min_atr_pct)
        self.max_atr_pct = float(max_atr_pct)
        self.tp_atr_mult = float(tp_atr_mult)
        self.sl_atr_mult = float(sl_atr_mult)
        self.max_hold_seconds = float(max_hold_seconds)
        self.notional_usd = float(notional_usd)
        self.leverage_hint = float(leverage_hint)
        self.portfolio = portfolio
        self._state: Dict[str, _SymbolState] = {}

        # Lazy loads — keep the strategy importable / runnable even
        # before training artefacts exist.
        self._hour_maps = self._load_hour_maps()
        self._registry: Optional["ModelRegistry"] = None
        self._registry_loaded = False
        log.info(
            "money_printer: hour_map=%s models_dir=%s win_threshold=%.2f",
            "loaded" if self._hour_maps else "missing",
            str(self.models_dir),
            self.win_threshold,
        )

    # ------------------------------------------------------------------
    # artefact loading
    # ------------------------------------------------------------------

    def _load_hour_maps(self) -> Dict[tuple[str, str], "HourMap"]:
        try:
            from aera.backtest.analysis import load_hour_maps
            if not self.hour_map_path.exists():
                return {}
            return load_hour_maps(self.hour_map_path)
        except Exception as exc:
            log.warning("money_printer: failed to load hour map: %s", exc)
            return {}

    def _classifier_or_none(self):
        """Backward-compat: return the underlying GBT classifier if
        the registry has one, else ``None``. Older tests + callers
        used this method when the strategy held a single classifier.
        """
        reg = self._registry_or_none()
        if reg is None:
            return None
        for s in reg.scorers:
            if s.name == "gbt":
                return getattr(s, "classifier", None)
        return None

    def _registry_or_none(self):
        """Lazy-load the model registry. ``ModelRegistry.from_dir``
        gracefully handles a missing directory — it just returns an
        empty registry (which is "no opinion = pass through")."""
        if self._registry_loaded:
            return self._registry
        self._registry_loaded = True
        try:
            from aera.ml.registry import ModelRegistry
            self._registry = ModelRegistry.from_dir(self.models_dir)
            if len(self._registry) == 0:
                # Backwards-compat: maybe only the legacy single-file
                # model.joblib exists outside the standard layout.
                if self.model_path.exists() and self.model_path.parent != self.models_dir:
                    from aera.ml.model import load_model
                    from aera.ml.registry import GBTScorer
                    self._registry.add(GBTScorer(load_model(self.model_path)))
                    log.info("money_printer: loaded legacy single-model from %s", self.model_path)
            log.info(
                "money_printer: registry has %d scorer(s): %s",
                len(self._registry), self._registry.names() or ["<none>"],
            )
        except Exception as exc:
            log.warning("money_printer: failed to build model registry: %s", exc)
            self._registry = None
        return self._registry

    # ------------------------------------------------------------------
    # state plumbing
    # ------------------------------------------------------------------

    def _state_for(self, symbol: str) -> _SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = _SymbolState()
            self._state[symbol] = st
        return st

    def _bar_window(self, ts: int) -> tuple[int, int]:
        """Return ``(bar_open_ts, bar_close_ts)`` for the bar that
        encloses ``ts``. Aligned to the unix epoch."""
        bs = self.bar_seconds
        open_ts = (ts // bs) * bs
        return open_ts, open_ts + bs

    def _close_bar(self, st: _SymbolState) -> None:
        if not st.bar.started:
            return
        st.history.append(
            (st.bar.ts_open, st.bar.open, st.bar.high, st.bar.low, st.bar.close, st.bar.volume)
        )

    # ------------------------------------------------------------------
    # feature / atr helpers
    # ------------------------------------------------------------------

    def _atr_pct(self, history: Deque[tuple]) -> float:
        """Quick ATR estimate over the last 14 closed bars, as % of close."""
        if len(history) < 14:
            return 0.0
        recent = list(history)[-14:]
        trs: List[float] = []
        prev_close = recent[0][4]
        for ts_open, o, h, l, c, v in recent[1:]:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)
            prev_close = c
        if not trs:
            return 0.0
        atr = sum(trs) / len(trs)
        ref = recent[-1][4] or 1.0
        return atr / ref

    def _features(self, st: _SymbolState):
        """Build the feature row from the rolling bar window. Returns
        None when there isn't enough warm-up data.

        Cached per-(symbol × closed-bar count): every feature is
        bar-derived (returns over N bars, RSI on closes, ATR, EMA-dev,
        hour-of-day of the last bar), so within a single bar they are
        constant tick-to-tick. Without this cache the sweep spent
        almost all its time rebuilding 200-row pandas frames on
        every synthetic tick.
        """
        history = st.history
        if len(history) < 60:
            return None
        if st.features_at_bars == len(history) and st.features_row is not None:
            return st.features_row
        try:
            import pandas as pd
            from aera.ml.features import extract_features
        except ImportError:
            return None
        df = pd.DataFrame(
            list(history),
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        feats = extract_features(df)
        st.features_at_bars = len(history)
        if feats.empty:
            st.features_row = None
            return None
        row = feats.iloc[-1]
        if row.isna().any():
            st.features_row = None
            return None
        st.features_row = row
        return row

    def _hour_allows(self, symbol: str, ts: int) -> bool:
        """Hour-of-day gate. When no hour map exists for this symbol
        we default to ALLOWED (the strategy degrades to model-only)."""
        if not self._hour_maps:
            return True
        m = self._hour_maps.get((self.name, symbol))
        if m is None:
            return True
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        allowed = m.hours_allowed()
        if not allowed:
            return True  # map exists but no profitable hour found → don't block everything
        return hour in allowed

    # ------------------------------------------------------------------
    # core scan
    # ------------------------------------------------------------------

    def scan(self, markets: Iterable[Market]) -> List[Signal]:
        signals: List[Signal] = []
        now_int = int(_time.time())

        for m in markets:
            if m.venue != "delta":
                continue
            outcome = next(iter(m.outcomes.values()), None)
            if outcome is None or outcome.label != DELTA_OUTCOME_LABEL:
                continue
            bid = outcome.best_bid
            ask = outcome.best_ask
            if bid is None or ask is None or ask <= 0:
                continue
            mid = 0.5 * (bid + ask)
            if mid <= 0:
                continue

            st = self._state_for(m.id)
            # Reconcile internal state with the live portfolio so the
            # same phantom-position pattern that bit the other
            # strategies can't bite this one.
            self.sync_position_state(st, self.portfolio, m.id, outcome.id)

            # Bar aggregator: append to current bar, roll over when
            # the bar window changes.
            tick_ts = int(m.last_update or now_int)
            bar_open, bar_close_ts = self._bar_window(tick_ts)
            if st.bar.ts_open != bar_open:
                self._close_bar(st)
                st.bar.reset(bar_open, bar_close_ts, mid)
            else:
                st.bar.update(mid)

            # Exit path first: existing position?
            close = self._maybe_close(st, m, outcome, bid, ask, mid, tick_ts)
            if close is not None:
                signals.append(close)
                continue

            if st.position_side is not None:
                continue

            # Need enough warm-up bars + a closed bar history.
            atr_pct = self._atr_pct(st.history)
            if atr_pct < self.min_atr_pct or atr_pct > self.max_atr_pct:
                continue

            # Hour gate
            if not self._hour_allows(m.id, tick_ts):
                continue

            # Feature + ML gate (skipped when no models are present)
            feats = self._features(st)
            side, edge = self._decide_side(feats, st.history)
            if side is None:
                continue

            registry = self._registry_or_none()
            per_scorer: Dict[str, float] = {}
            if registry is not None and len(registry) > 0 and feats is not None:
                from aera.ml.registry import ScoringContext
                # Build a small candle window for sequence scorers (no-op
                # for scorers that ignore it).
                window_df: Optional["pd.DataFrame"] = None
                if len(st.history) >= 64:
                    try:
                        import pandas as pd
                        window_df = pd.DataFrame(
                            list(st.history)[-128:],
                            columns=["ts", "open", "high", "low", "close", "volume"],
                        )
                    except Exception:
                        window_df = None
                ctx = ScoringContext(
                    features=feats, candle_window=window_df,
                    symbol=m.id, side=side,
                )
                p_win, per_scorer = registry.combined(ctx)
                if p_win < self.win_threshold:
                    continue
                # Confidence-scaled sizing: more confidence → bigger fire.
                size_scale = max(0.25, min(1.0, 0.5 + (p_win - self.win_threshold) * 2.0))
            else:
                p_win = 0.5
                size_scale = 0.5  # No models = trade small until we have some.

            notional = self.notional_usd * size_scale
            atr_abs = atr_pct * mid
            tp = atr_abs * self.tp_atr_mult
            sl = atr_abs * self.sl_atr_mult

            limit_price = ask if side == "BUY" else bid
            try:
                leverage = float(m.metadata.get("leverage", self.leverage_hint) or self.leverage_hint)
            except (TypeError, ValueError):
                leverage = self.leverage_hint

            leg = Leg(
                market_id=m.id,
                outcome_id=outcome.id,
                side=side,
                limit_price=float(limit_price),
                size_usd=float(notional),
                reason=(
                    f"money_printer fire: p_win={p_win:.2f} atr%={atr_pct*1e4:.1f}bps "
                    f"tp=${tp:.4f} sl=${sl:.4f} size_x{size_scale:.2f}"
                ),
                leverage=leverage,
            )
            meta = {
                "symbol": m.id,
                "p_win": float(p_win),
                "atr_pct": float(atr_pct),
                "tp_usd_abs": float(tp),
                "sl_usd_abs": float(sl),
                "size_scale": float(size_scale),
                "mid": float(mid),
            }
            if per_scorer:
                meta["scorers"] = {k: float(v) for k, v in per_scorer.items()}
            signals.append(Signal(
                strategy=self.name,
                confidence=float(p_win),
                edge=float(edge),
                legs=[leg],
                metadata=meta,
            ))
            st.position_side = "LONG" if side == "BUY" else "SHORT"
            st.entry_mid = mid
            st.entry_size_usd = notional
            st.entry_atr = atr_abs
            st.entry_time = float(tick_ts)
            log.info(
                "money_printer FIRE %s %s mid=%.4f p_win=%.2f atr%%=%.2fbps "
                "tp=%.4f sl=%.4f",
                m.id, side, mid, p_win, atr_pct * 1e4, tp, sl,
            )

        return signals

    # ------------------------------------------------------------------
    # entry decision
    # ------------------------------------------------------------------

    def _decide_side(self, feats, history) -> tuple[Optional[str], float]:
        """Decide BUY / SELL / SKIP using the most-recent feature row.

        With the model present this is mostly a "pick the side that
        matches the recent drift; the model already filtered for
        edge". Without the model we fall back to a simple
        mean-reversion bias on ``ret_5`` and ``ema_dev_20``.
        """
        if feats is None:
            return None, 0.0
        ret5 = float(feats.get("ret_5", 0.0))
        ema_dev = float(feats.get("ema_dev_20", 0.0))
        rsi = float(feats.get("rsi_14", 50.0))
        # Mean-reversion bias: oversold → BUY, overbought → SELL.
        if rsi < 35 and ema_dev < -0.001:
            return "BUY", min(0.01, abs(ema_dev) * 5)
        if rsi > 65 and ema_dev > 0.001:
            return "SELL", min(0.01, abs(ema_dev) * 5)
        # Momentum continuation: small drift in same direction with low vol.
        vol = float(feats.get("vol_15", 0.0))
        if vol > 0 and abs(ret5) > 2 * vol:
            return ("BUY" if ret5 > 0 else "SELL"), min(0.01, abs(ret5))
        return None, 0.0

    # ------------------------------------------------------------------
    # exit logic
    # ------------------------------------------------------------------

    def _maybe_close(
        self, st: _SymbolState, market: Market, outcome,
        bid: float, ask: float, mid: float, ts: int,
    ) -> Optional[Signal]:
        if st.position_side is None or st.entry_mid <= 0:
            return None

        # Hold timeout
        if self.max_hold_seconds > 0 and (ts - st.entry_time) > self.max_hold_seconds:
            return self._emit_close(
                st, market, outcome, bid, ask, mid,
                reason=f"hold timeout ({ts - st.entry_time:.0f}s > {self.max_hold_seconds:.0f}s)",
            )

        tp_abs = st.entry_atr * self.tp_atr_mult
        sl_abs = st.entry_atr * self.sl_atr_mult
        move = (mid - st.entry_mid) if st.position_side == "LONG" else (st.entry_mid - mid)

        if move >= tp_abs and tp_abs > 0:
            return self._emit_close(
                st, market, outcome, bid, ask, mid,
                reason=f"TP hit (move={move:.4f} >= tp={tp_abs:.4f})",
            )
        if move <= -sl_abs and sl_abs > 0:
            return self._emit_close(
                st, market, outcome, bid, ask, mid,
                reason=f"SL hit (move={move:.4f} <= -sl={sl_abs:.4f})",
            )
        return None

    def _emit_close(
        self, st: _SymbolState, market: Market, outcome,
        bid: float, ask: float, mid: float, *, reason: str,
    ) -> Signal:
        close_side = "SELL" if st.position_side == "LONG" else "BUY"
        limit_price = bid if close_side == "SELL" else ask
        try:
            leverage = float(market.metadata.get("leverage", self.leverage_hint) or self.leverage_hint)
        except (TypeError, ValueError):
            leverage = self.leverage_hint
        leg = Leg(
            market_id=market.id, outcome_id=outcome.id, side=close_side,
            limit_price=float(limit_price), size_usd=float(st.entry_size_usd),
            reason=f"money_printer close: {reason}",
            leverage=leverage, reduce_only=True,
        )
        log.info(
            "money_printer CLOSE %s %s mid=%.4f entry=%.4f side=%s reason=%s",
            market.id, close_side, mid, st.entry_mid, st.position_side, reason,
        )
        # Wipe state — sync_position_state will reconcile on the next tick.
        st.position_side = None
        st.entry_mid = 0.0
        st.entry_size_usd = 0.0
        st.entry_atr = 0.0
        st.entry_time = 0.0
        return Signal(
            strategy=self.name, confidence=1.0, edge=0.01, legs=[leg],
            metadata={"symbol": market.id, "exit_reason": reason},
        )
