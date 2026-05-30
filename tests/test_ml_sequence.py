"""Sequence model — graceful degradation + (when torch is present) training smoke."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aera.ml.sequence import (
    SEQUENCE_FEATURES,
    SequenceConfig,
    SequenceScorer,
    bars_to_sequence_matrix,
)


# ---------------------------------------------------------------------------
# always-on tests (no torch needed)
# ---------------------------------------------------------------------------


def _toy_candles(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts0 = 1_700_000_000
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    opens = np.roll(closes, 1); opens[0] = closes[0]
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    return pd.DataFrame({
        "ts":    [ts0 + i * 60 for i in range(n)],
        "open":  opens, "high": highs, "low": lows, "close": closes,
        "volume": np.abs(rng.normal(100, 20, n)),
    })


def test_bars_to_sequence_matrix_shape():
    df = _toy_candles(50)
    mat = bars_to_sequence_matrix(df)
    assert mat.shape == (50, len(SEQUENCE_FEATURES))
    assert mat.dtype == np.float32
    assert not np.isnan(mat).any()


def test_bars_to_sequence_matrix_empty_input():
    mat = bars_to_sequence_matrix(pd.DataFrame())
    assert mat.shape == (0, len(SEQUENCE_FEATURES))


def test_scorer_no_model_returns_neutral():
    scorer = SequenceScorer()
    val = scorer.predict_proba_win(_toy_candles(80))
    assert val == pytest.approx(0.5)


def test_scorer_empty_input_returns_neutral():
    scorer = SequenceScorer()
    val = scorer.predict_proba_win(np.zeros((0, len(SEQUENCE_FEATURES)), dtype=np.float32))
    assert val == pytest.approx(0.5)


def test_scorer_save_without_model_raises():
    scorer = SequenceScorer()
    if not SequenceScorer.available():
        pytest.skip("torch not installed; skipping save-failure test")
    with pytest.raises(RuntimeError):
        scorer.save("/tmp/never_written.pt")


# ---------------------------------------------------------------------------
# torch-only tests
# ---------------------------------------------------------------------------


pytestmark_torch = pytest.mark.skipif(
    not SequenceScorer.available(),
    reason="torch is not installed — sequence-model training tests skipped",
)


@pytestmark_torch
def test_train_sequence_model_smoke(tmp_path):
    from aera.ml.sequence import train_sequence_model, load_sequence_scorer
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(64, 32, len(SEQUENCE_FEATURES))).astype(np.float32)
    # Make label depend on average return of first feature — model
    # should learn this in a couple of epochs.
    y = (X[:, :, 0].mean(axis=1) > 0).astype(np.int64)
    cfg = SequenceConfig(seq_len=32, d_model=16, n_heads=2, n_layers=1)
    scorer, report = train_sequence_model(X, y, config=cfg, epochs=2, batch_size=16, lr=1e-2)
    assert report.n_train > 0
    p = scorer.predict_proba_win(X[0])
    assert 0.0 <= p <= 1.0

    out = tmp_path / "seq.pt"
    scorer.save(out)
    loaded = load_sequence_scorer(out)
    p_after = loaded.predict_proba_win(X[0])
    assert p == pytest.approx(p_after, abs=1e-4)


@pytestmark_torch
def test_build_sequences_from_trades(tmp_path):
    from aera.ml.sequence import build_sequences_from_trades
    candles = {"BTCUSD": _toy_candles(200), "ETHUSD": _toy_candles(200)}
    trades = pd.DataFrame([
        {"symbol": "BTCUSD", "entry_ts": int(candles["BTCUSD"].iloc[100]["ts"]), "is_win": 1},
        {"symbol": "BTCUSD", "entry_ts": int(candles["BTCUSD"].iloc[150]["ts"]), "is_win": 0},
        {"symbol": "ETHUSD", "entry_ts": int(candles["ETHUSD"].iloc[120]["ts"]), "is_win": 1},
    ])
    X, y, syms = build_sequences_from_trades(trades, candles, seq_len=64)
    assert X.shape == (3, 64, len(SEQUENCE_FEATURES))
    assert y.shape == (3,)
    assert syms == ["BTCUSD", "BTCUSD", "ETHUSD"]
