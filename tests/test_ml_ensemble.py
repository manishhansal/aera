"""Per-symbol ensemble classifier — wiring + persistence + routing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aera.ml import (
    FEATURE_COLUMNS,
    EnsembleClassifier,
    load_ensemble,
    train_ensemble,
)


def _synthetic_labelled(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym, sign in [("BTCUSD", +1), ("ETHUSD", -1), ("SOLUSD", +1)]:
        for i in range(300):
            base = rng.normal(0.0, 1.0, size=len(FEATURE_COLUMNS))
            # Make the label depend on symbol so per-symbol models can
            # learn a discriminative pattern (vs. one global average).
            label = int((base.sum() * sign) > 0)
            row = {c: float(v) for c, v in zip(FEATURE_COLUMNS, base)}
            row["label"] = label
            row["symbol"] = sym
            row["entry_ts"] = 1_700_000_000 + i * 60
            rows.append(row)
    return pd.DataFrame(rows)


def test_train_ensemble_produces_per_symbol_models():
    df = _synthetic_labelled(seed=1)
    ens, report = train_ensemble(df, min_per_symbol=100)
    assert isinstance(ens, EnsembleClassifier)
    assert ens.fallback is not None
    # Every symbol exceeds 100 rows → all should have per-symbol models.
    assert set(ens.per_symbol.keys()) == {"BTCUSD", "ETHUSD", "SOLUSD"}
    assert "BTCUSD" in report.per_symbol


def test_train_ensemble_falls_back_for_small_symbols():
    df = _synthetic_labelled(seed=2)
    ens, report = train_ensemble(df, min_per_symbol=10_000)  # impossible bar
    assert ens.fallback is not None
    assert ens.per_symbol == {}
    assert set(report.skipped_symbols.keys()) == {"BTCUSD", "ETHUSD", "SOLUSD"}


def test_ensemble_routes_to_per_symbol_when_present():
    df = _synthetic_labelled(seed=3)
    ens, _ = train_ensemble(df, min_per_symbol=100)
    row = df.iloc[0][FEATURE_COLUMNS]
    used = ens.models_used(["BTCUSD", "ETHUSD", "ADAUSD"])
    assert used["BTCUSD"] == "per_symbol"
    assert used["ETHUSD"] == "per_symbol"
    assert used["ADAUSD"] == "fallback"


def test_ensemble_score_is_in_unit_interval():
    df = _synthetic_labelled(seed=4)
    ens, _ = train_ensemble(df, min_per_symbol=100)
    feats = df[FEATURE_COLUMNS].iloc[0]
    p = ens.predict_proba_win(feats, symbol="BTCUSD")
    assert 0.0 <= float(p[0]) <= 1.0


def test_ensemble_round_trip_save_load(tmp_path):
    df = _synthetic_labelled(seed=5)
    ens, _ = train_ensemble(df, min_per_symbol=100)
    out = tmp_path / "ensemble"
    ens.save(out)

    loaded = load_ensemble(out)
    assert loaded.fallback is not None
    assert set(loaded.per_symbol.keys()) == set(ens.per_symbol.keys())

    # Same input should produce the same output post-roundtrip.
    feats = df[FEATURE_COLUMNS].iloc[0]
    p_before = float(ens.predict_proba_win(feats, symbol="BTCUSD")[0])
    p_after = float(loaded.predict_proba_win(feats, symbol="BTCUSD")[0])
    assert p_before == pytest.approx(p_after, abs=1e-6)


def test_ensemble_raises_when_no_models_or_fallback():
    ens = EnsembleClassifier(fallback=None, per_symbol={})
    with pytest.raises(RuntimeError, match="neither"):
        ens.predict_proba_win(pd.Series({c: 0.0 for c in FEATURE_COLUMNS}), symbol="X")


def test_load_ensemble_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ensemble(tmp_path / "does_not_exist")
