"""Feature engineering for the profitability classifier.

The classifier in :mod:`aera.ml.model` answers ONE question per
candidate fire: **"will this trade close in profit?"**. The features
below describe the market state immediately preceding the fire.
They are deliberately simple — recent returns, volatility, range,
volume ratio, time-of-day — so the model trains in seconds on a
laptop and runs in microseconds in the live decision loop.

Feature list (15 numeric features)
==================================

Returns
-------
* ``ret_1``  : last-bar close-to-close return
* ``ret_5``  : 5-bar close-to-close return
* ``ret_15`` : 15-bar close-to-close return
* ``ret_60`` : 60-bar close-to-close return

Volatility
----------
* ``vol_15`` : stdev of last 15 close-to-close returns
* ``atr_14`` : Average True Range over last 14 bars (% of close)
* ``range_now`` : current bar's (high-low)/close

Mean reversion
--------------
* ``ema_dev_20`` : (close − EMA20) / EMA20
* ``rsi_14``     : classic RSI on 14 bars

Volume
------
* ``vol_ratio_20`` : current volume / mean(last 20 volumes)

Microstructure proxy
--------------------
* ``upper_wick`` : (high − max(open, close)) / (high − low + eps)
* ``lower_wick`` : (min(open, close) − low) / (high − low + eps)
* ``body_pct``   : |close − open| / (high − low + eps)

Time
----
* ``hour_sin`` : sin(2π · hour_of_day / 24)
* ``hour_cos`` : cos(2π · hour_of_day / 24)

The trigonometric encoding lets the classifier learn smooth time-of-
day effects (e.g. "Asian session quieter than US open") without
treating midnight and 11pm as far-apart classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

import math
import numpy as np
import pandas as pd

from aera.backtest.replay import TradeRecord
from aera.logging import get_logger

log = get_logger(__name__)


FEATURE_COLUMNS: List[str] = [
    "ret_1", "ret_5", "ret_15", "ret_60",
    "vol_15", "atr_14", "range_now",
    "ema_dev_20", "rsi_14",
    "vol_ratio_20",
    "upper_wick", "lower_wick", "body_pct",
    "hour_sin", "hour_cos",
]


# ---------------------------------------------------------------------------
# bar-level feature extraction
# ---------------------------------------------------------------------------


def extract_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per candle and ``FEATURE_COLUMNS``.

    The output is aligned to the SAME index as ``candles`` (so the
    caller can join by row position); rows where any feature is NaN
    (warm-up window) are kept — drop them at the model layer.
    """
    if candles is None or candles.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = candles.copy().reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df.get("volume", pd.Series([0.0] * len(df))).astype(float)

    eps = 1e-12

    out = pd.DataFrame(index=df.index)

    # ---- returns ---------------------------------------------------
    out["ret_1"]  = close.pct_change(1)
    out["ret_5"]  = close.pct_change(5)
    out["ret_15"] = close.pct_change(15)
    out["ret_60"] = close.pct_change(60)

    # ---- volatility ------------------------------------------------
    log_ret = np.log(close / close.shift(1) + eps)
    out["vol_15"] = log_ret.rolling(15).std()

    # ATR — true range = max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1,
    ).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / (close + eps)

    out["range_now"] = (high - low) / (close + eps)

    # ---- mean reversion --------------------------------------------
    ema20 = close.ewm(span=20, adjust=False).mean()
    out["ema_dev_20"] = (close - ema20) / (ema20 + eps)

    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.rolling(14).mean()
    avg_dn = down.rolling(14).mean()
    rs = avg_up / (avg_dn + eps)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ---- volume ----------------------------------------------------
    vol_mean = volume.rolling(20).mean()
    out["vol_ratio_20"] = volume / (vol_mean + eps)

    # ---- microstructure proxy --------------------------------------
    rng = (high - low).replace(0.0, eps)
    out["upper_wick"] = (high - np.maximum(open_, close)) / rng
    out["lower_wick"] = (np.minimum(open_, close) - low) / rng
    out["body_pct"]   = (close - open_).abs() / rng

    # ---- time ------------------------------------------------------
    ts = df["ts"].astype("int64")
    hours = ((ts // 3600) % 24).astype(float)
    out["hour_sin"] = np.sin(2.0 * math.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2.0 * math.pi * hours / 24.0)

    return out[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# trade-level labelling
# ---------------------------------------------------------------------------


def label_trades(
    trades: Iterable[TradeRecord],
    candles_by_symbol: dict[str, pd.DataFrame],
    *,
    lookback_bars: int = 60,
) -> pd.DataFrame:
    """Build a labelled (X, y) frame from a list of trades.

    For each trade we look up the candle whose open is at or
    immediately BEFORE the trade's ``entry_ts`` and extract the
    feature row at that bar. The label is the trade's outcome
    (``is_win`` → 1, else 0).

    Trades whose symbol isn't in ``candles_by_symbol`` are silently
    skipped. So are trades that land before bar ``lookback_bars``
    (insufficient feature warm-up).
    """
    rows: List[dict] = []
    for sym, candles in candles_by_symbol.items():
        feats = extract_features(candles)
        if feats.empty:
            continue
        ts_arr = candles["ts"].astype("int64").to_numpy()
        for t in trades:
            if t.symbol != sym:
                continue
            # Find the largest bar whose open <= entry_ts
            idx = int(np.searchsorted(ts_arr, t.entry_ts, side="right") - 1)
            if idx < lookback_bars:
                continue
            row = feats.iloc[idx].to_dict()
            row["label"] = int(t.is_win)
            row["pnl_usd"] = float(t.pnl_usd)
            row["strategy"] = t.strategy
            row["symbol"] = t.symbol
            row["side"] = t.side
            row["entry_ts"] = int(t.entry_ts)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# live extractor: one row from a streaming context
# ---------------------------------------------------------------------------


@dataclass
class FeatureExtractor:
    """Maintain a rolling window of recent OHLCV samples and produce
    one feature row on demand.

    Used by the money printer at decision time: each tick the
    extractor is fed the latest synthetic-bar close (live) and asked
    for the current feature row. The model then scores it.

    Implementation is the same vectorised pandas path as
    ``extract_features`` so live + offline scoring always agree.
    """

    window: int = 200
    _bars: List[tuple[int, float, float, float, float, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._bars = []

    def feed_bar(
        self, ts: int, open_: float, high: float, low: float, close: float, volume: float,
    ) -> None:
        self._bars.append((int(ts), float(open_), float(high), float(low), float(close), float(volume)))
        if len(self._bars) > self.window:
            del self._bars[:-self.window]

    def latest_features(self) -> Optional[pd.Series]:
        if len(self._bars) < 60:
            return None
        df = pd.DataFrame(self._bars, columns=["ts", "open", "high", "low", "close", "volume"])
        feats = extract_features(df)
        row = feats.iloc[-1]
        if row.isna().any():
            return None
        return row

    def __len__(self) -> int:
        return len(self._bars)
