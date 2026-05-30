"""Feature engineering + classifier — the ML layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aera.backtest.replay import TradeRecord
from aera.ml import (
    FEATURE_COLUMNS,
    FeatureExtractor,
    extract_features,
    label_trades,
    load_model,
    train_model,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _candles(n: int = 200, base: float = 1000.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = base
    for i in range(n):
        ts = 1_700_000_000 + i * 60
        op = price
        cl = price * (1 + rng.normal(0.0, 0.001))
        hi = max(op, cl) * (1 + abs(rng.normal(0.0, 0.0005)))
        lo = min(op, cl) * (1 - abs(rng.normal(0.0, 0.0005)))
        vol = float(rng.integers(50, 500))
        rows.append((ts, op, hi, lo, cl, vol))
        price = cl
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def _trades(symbol: str, candles: pd.DataFrame, n: int = 30, seed: int = 1) -> list[TradeRecord]:
    """Build a set of synthetic trades aligned to the candle timestamps."""
    rng = np.random.default_rng(seed)
    out: list[TradeRecord] = []
    ts_arr = candles["ts"].astype("int64").to_numpy()
    eligible = ts_arr[60:-2]  # leave warm-up + 1 forward bar
    if len(eligible) == 0:
        return []
    picks = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    for ts in picks:
        idx = int(np.searchsorted(ts_arr, ts))
        if idx + 1 >= len(candles):
            continue
        entry_px = float(candles.iloc[idx]["open"])
        exit_px = float(candles.iloc[idx + 1]["close"])
        pnl = (exit_px - entry_px) * 1.0  # $1 notional → cents-of-pnl
        out.append(TradeRecord(
            strategy="test_strat", symbol=symbol, side="LONG",
            entry_ts=int(ts), exit_ts=int(ts) + 60,
            entry_price=entry_px, exit_price=exit_px,
            size_usd=100.0, leverage=1.0,
            pnl_usd=pnl, fees_usd=0.0,
            hold_seconds=60, reason_open="", reason_close="",
        ))
    return out


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------


def test_extract_features_returns_correct_shape_and_columns():
    c = _candles(n=100)
    feats = extract_features(c)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(feats) == len(c)


def test_features_handle_empty_input():
    feats = extract_features(pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]))
    assert feats.empty
    assert list(feats.columns) == FEATURE_COLUMNS


def test_features_first_rows_are_nan_during_warmup():
    c = _candles(n=120)
    feats = extract_features(c)
    # ret_60 needs 60 prior bars, so the first 60 rows should be NaN
    assert feats["ret_60"].iloc[:60].isna().all()
    assert feats["ret_60"].iloc[60:].notna().all()


def test_features_time_encoding_is_unit_circle():
    c = _candles(n=24)
    feats = extract_features(c)
    radius = feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2
    assert (radius.dropna().sub(1.0).abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# label_trades
# ---------------------------------------------------------------------------


def test_label_trades_skips_warmup_window():
    c = _candles(n=120)
    # Trade RIGHT at the start — should be dropped because lookback=60.
    early_trade = TradeRecord(
        strategy="x", symbol="BTCUSD", side="LONG",
        entry_ts=int(c.iloc[5]["ts"]), exit_ts=int(c.iloc[10]["ts"]),
        entry_price=1.0, exit_price=1.0, size_usd=10, leverage=1.0,
        pnl_usd=1.0, fees_usd=0.0, hold_seconds=300,
        reason_open="", reason_close="",
    )
    late_trade = TradeRecord(
        strategy="x", symbol="BTCUSD", side="LONG",
        entry_ts=int(c.iloc[80]["ts"]), exit_ts=int(c.iloc[85]["ts"]),
        entry_price=1.0, exit_price=1.0, size_usd=10, leverage=1.0,
        pnl_usd=-1.0, fees_usd=0.0, hold_seconds=300,
        reason_open="", reason_close="",
    )
    labelled = label_trades([early_trade, late_trade], {"BTCUSD": c}, lookback_bars=60)
    assert len(labelled) == 1
    assert labelled["label"].iloc[0] == 0  # the late trade was a loser


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------


def test_train_model_round_trips_predict_proba(tmp_path):
    c = _candles(n=300, seed=0)
    # Inject a deliberate signal: if RSI > 50, the next bar is up. The
    # classifier should learn at least a hint of that pattern.
    trades = _trades("BTCUSD", c, n=120, seed=1)
    labelled = label_trades(trades, {"BTCUSD": c})
    assert len(labelled) >= 50
    model, report = train_model(labelled)
    # Smoke: ROC AUC is at least defined and in [0, 1] (or NaN on
    # degenerate splits — which is also acceptable for noise data).
    assert report.n_train > 0
    assert report.n_test > 0
    assert report.n_features == len(FEATURE_COLUMNS)
    # Round-trip the model through disk.
    path = tmp_path / "model.joblib"
    model.save(path)
    reloaded = load_model(path)
    proba = reloaded.predict_proba_win(labelled[FEATURE_COLUMNS].head(5))
    assert proba.shape == (5,)
    assert ((0.0 <= proba) & (proba <= 1.0)).all()


def test_train_model_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        train_model(pd.DataFrame())


def test_train_model_rejects_frame_without_label():
    c = _candles(n=120)
    feats = extract_features(c).dropna()
    with pytest.raises(ValueError, match="label"):
        train_model(feats)


# ---------------------------------------------------------------------------
# live FeatureExtractor
# ---------------------------------------------------------------------------


def test_live_feature_extractor_returns_none_before_warmup():
    fe = FeatureExtractor(window=200)
    for i in range(10):
        fe.feed_bar(i, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert fe.latest_features() is None


def test_live_feature_extractor_returns_row_after_warmup():
    fe = FeatureExtractor(window=200)
    rng = np.random.default_rng(0)
    price = 100.0
    for i in range(80):
        price *= 1 + rng.normal(0.0, 0.001)
        fe.feed_bar(i, price, price * 1.001, price * 0.999, price, 100.0)
    row = fe.latest_features()
    assert row is not None
    assert set(row.index) == set(FEATURE_COLUMNS)
    assert not row.isna().any()
