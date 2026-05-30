"""Model registry — fusion math, auto-discovery, graceful degradation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aera.ml import FEATURE_COLUMNS
from aera.ml.registry import (
    FusionWeights,
    ModelRegistry,
    Scorer,
    ScoringContext,
)


class _ConstScorer(Scorer):
    """Always returns the same score — used to drive fusion math tests."""

    def __init__(self, value: float, name: str, available: bool = True):
        self._v = value
        self.name = name
        self._available = available

    def available(self) -> bool:
        return self._available

    def score(self, ctx: ScoringContext) -> float:
        return self._v


def _ctx() -> ScoringContext:
    feats = pd.Series({c: 0.0 for c in FEATURE_COLUMNS})
    return ScoringContext(features=feats, symbol="BTCUSD", side="BUY")


def test_empty_registry_returns_neutral_score():
    reg = ModelRegistry()
    fused, per = reg.combined(_ctx())
    assert per == {}
    assert fused == pytest.approx(0.5)


def test_registry_skips_unavailable_scorers():
    reg = ModelRegistry()
    reg.add(_ConstScorer(0.9, "gbt", available=False))
    reg.add(_ConstScorer(0.7, "ensemble", available=True))
    assert reg.names() == ["ensemble"]


def test_registry_fuses_weighted_average():
    reg = ModelRegistry(weights=FusionWeights(gbt=1.0, ensemble=2.0))
    reg.add(_ConstScorer(0.6, "gbt"))
    reg.add(_ConstScorer(0.8, "ensemble"))
    fused, per = reg.combined(_ctx())
    # (1.0 * 0.6 + 2.0 * 0.8) / (1.0 + 2.0) = 2.2 / 3.0 ≈ 0.7333
    assert fused == pytest.approx((1.0 * 0.6 + 2.0 * 0.8) / 3.0, abs=1e-6)
    assert per == {"gbt": 0.6, "ensemble": 0.8}


def test_registry_handles_unknown_scorer_name_with_zero_weight():
    reg = ModelRegistry(weights=FusionWeights())  # default weights
    reg.add(_ConstScorer(0.9, "mystery_scorer"))   # not in FusionWeights
    fused, per = reg.combined(_ctx())
    # Unknown scorer gets 0 weight → fused score falls back to 0.5
    # (denom is zero, so the registry treats it as no opinion).
    assert per == {"mystery_scorer": 0.9}
    assert fused == pytest.approx(0.5)


def test_from_dir_loads_gbt_when_present(tmp_path):
    # Build a tiny GBT and persist it under the expected filename.
    from aera.ml import train_model
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(120):
        r = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
        r["label"] = int(sum(r.values()) > 0)
        rows.append(r)
    clf, _ = train_model(pd.DataFrame(rows))
    out = tmp_path / "model.joblib"
    clf.save(out)

    reg = ModelRegistry.from_dir(tmp_path)
    assert "gbt" in reg.names()
    fused, per = reg.combined(_ctx())
    assert 0.0 <= per["gbt"] <= 1.0


def test_from_dir_loads_ensemble_when_present(tmp_path):
    # Build a tiny ensemble and persist under ensemble/
    from aera.ml import train_ensemble
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for sym in ("BTCUSD", "ETHUSD"):
        for _ in range(120):
            r = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
            r["label"] = int(sum(r.values()) > 0)
            r["symbol"] = sym
            r["entry_ts"] = 1700000000
            rows.append(r)
    ens, _ = train_ensemble(pd.DataFrame(rows), min_per_symbol=100)
    ens.save(tmp_path / "ensemble")

    reg = ModelRegistry.from_dir(tmp_path)
    assert "ensemble" in reg.names()


def test_from_dir_with_empty_directory_is_empty_registry(tmp_path):
    reg = ModelRegistry.from_dir(tmp_path)
    assert len(reg) == 0
    fused, _ = reg.combined(_ctx())
    assert fused == pytest.approx(0.5)
