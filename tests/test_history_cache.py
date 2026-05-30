"""CandleStore — local OHLCV cache."""
from __future__ import annotations

import pandas as pd
import pytest

from aera.data import Candle, CandleStore


def test_empty_load_returns_typed_frame(tmp_path):
    store = CandleStore(root=tmp_path)
    df = store.load("BTCUSD", "1m")
    assert df.empty
    for col in ("ts", "open", "high", "low", "close", "volume"):
        assert col in df.columns


def test_save_then_load_round_trips_rows(tmp_path):
    store = CandleStore(root=tmp_path)
    candles = [
        Candle(ts=100, open=10.0, high=11.0, low=9.0, close=10.5, volume=100),
        Candle(ts=160, open=10.5, high=12.0, low=10.0, close=11.5, volume=200),
        Candle(ts=220, open=11.5, high=11.6, low=10.8, close=11.0, volume=150),
    ]
    added = store.save("BTCUSD", "1m", candles)
    assert added == 3
    df = store.load("BTCUSD", "1m")
    assert len(df) == 3
    assert list(df["ts"]) == [100, 160, 220]


def test_save_dedupes_overlapping_timestamps(tmp_path):
    store = CandleStore(root=tmp_path)
    base = [
        Candle(ts=100, open=10.0, high=11.0, low=9.0, close=10.5, volume=100),
        Candle(ts=160, open=10.5, high=12.0, low=10.0, close=11.5, volume=200),
    ]
    store.save("BTCUSD", "1m", base)
    overlap = [
        Candle(ts=160, open=10.5, high=12.0, low=10.0, close=11.6, volume=250),
        Candle(ts=220, open=11.5, high=11.6, low=10.8, close=11.0, volume=150),
    ]
    added = store.save("BTCUSD", "1m", overlap)
    assert added == 1, "only the genuinely new row at ts=220 should count"
    df = store.load("BTCUSD", "1m")
    assert len(df) == 3
    # Last-write-wins: ts=160's close came from the overlap batch.
    assert float(df.loc[df["ts"] == 160, "close"].iloc[0]) == 11.6


def test_coverage_reports_first_last_and_count(tmp_path):
    store = CandleStore(root=tmp_path)
    store.save("BTCUSD", "1m", [
        Candle(ts=300, open=1, high=1, low=1, close=1, volume=0),
        Candle(ts=100, open=1, high=1, low=1, close=1, volume=0),
        Candle(ts=200, open=1, high=1, low=1, close=1, volume=0),
    ])
    first, last, n = store.coverage("BTCUSD", "1m")
    assert (first, last, n) == (100, 300, 3)


def test_coverage_for_missing_pair_returns_none_none_zero(tmp_path):
    store = CandleStore(root=tmp_path)
    assert store.coverage("NONEXISTENT", "1m") == (None, None, 0)


def test_save_dataframe_directly_also_works(tmp_path):
    store = CandleStore(root=tmp_path)
    df = pd.DataFrame({
        "ts": [10, 20], "open": [1.0, 2.0], "high": [1.5, 2.5],
        "low": [0.5, 1.5], "close": [1.2, 2.2], "volume": [10, 20],
    })
    added = store.save("ETHUSD", "5m", df)
    assert added == 2
    loaded = store.load("ETHUSD", "5m")
    assert len(loaded) == 2
