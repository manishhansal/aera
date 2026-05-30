"""Profitability classifier: predict win/loss probability per setup.

A gradient-boosted classifier (sklearn) trained on the labelled
trade features from :func:`aera.ml.features.label_trades`. Returns
``P(win | features)`` on demand; the money printer fires only when
that probability is above a configured threshold.

Why gradient boosting (and not deep learning)?
---------------------------------------------

* Tabular financial data is the canonical "GBT > NN" regime —
  XGBoost / LightGBM / sklearn's HistGradientBoosting consistently
  beat MLP / LSTM on tick-bar features in published benchmarks
  while training in seconds.
* Production scoring is a single ``predict_proba`` call per tick;
  no GPU, no warm-up cost.
* Feature importance lands for free, which is the right output for
  "tell me which signals actually matter" debugging.

The trainer does a walk-forward train/test split (last 20% held
out) so the reported metrics aren't biased by future-leak.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from aera.logging import get_logger

from .features import FEATURE_COLUMNS

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# report container
# ---------------------------------------------------------------------------


@dataclass
class TrainReport:
    """Metrics produced by :func:`train_model` for one fit."""

    n_train: int
    n_test: int
    n_features: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    class_balance: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "feature_importance": self.feature_importance,
            "class_balance": self.class_balance,
        }


# ---------------------------------------------------------------------------
# wrapper class: persistable + simple predict_proba surface
# ---------------------------------------------------------------------------


class ProfitabilityClassifier:
    """Wraps the sklearn model + the feature column list for
    deterministic serve-time prediction.

    Use :func:`train_model` to fit; use :func:`load_model` to load
    a previously persisted instance.
    """

    def __init__(self, model=None, features: Optional[Sequence[str]] = None) -> None:
        self.model = model
        self.features: List[str] = list(features) if features is not None else list(FEATURE_COLUMNS)

    def predict_proba_win(self, X: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Return ``P(win)`` for each row."""
        if self.model is None:
            raise RuntimeError("model not trained / loaded")
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if isinstance(X, pd.DataFrame):
            X = X[self.features].to_numpy()
        proba = self.model.predict_proba(X)
        # sklearn classes_ orders [0, 1] for binary; column 1 is P(win=1)
        classes = list(self.model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        # Single-class fit fallback — every prediction is the same.
        return np.full(proba.shape[0], float(classes[0]))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features}, path)
        return path


def load_model(path: str | Path) -> ProfitabilityClassifier:
    blob = joblib.load(path)
    return ProfitabilityClassifier(model=blob["model"], features=blob.get("features"))


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


def _walk_forward_split(
    df: pd.DataFrame, test_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by ``entry_ts`` if present (otherwise by row order) and
    keep the LAST ``test_frac`` for test. Walk-forward is the only
    honest split for time-series; random shuffling leaks the future
    into training."""
    if "entry_ts" in df.columns:
        df = df.sort_values("entry_ts").reset_index(drop=True)
    n = len(df)
    if n < 50:
        # Too small for a hold-out; report on the same set.
        return df, df
    cut = int(n * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def train_model(
    labelled: pd.DataFrame,
    *,
    test_frac: float = 0.20,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[ProfitabilityClassifier, TrainReport]:
    """Fit a HistGradientBoostingClassifier on ``labelled``.

    ``labelled`` must have ``FEATURE_COLUMNS`` + ``label`` columns
    (the output of :func:`aera.ml.features.label_trades`). Returns
    the wrapped classifier and a :class:`TrainReport` with hold-out
    metrics.
    """
    # Import here so the rest of the package stays importable even
    # without sklearn (e.g. for unit tests that don't exercise ML).
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )

    if labelled.empty:
        raise ValueError("labelled trade frame is empty — nothing to train on")
    missing = [c for c in FEATURE_COLUMNS if c not in labelled.columns]
    if missing:
        raise ValueError(f"labelled frame missing feature columns: {missing}")
    if "label" not in labelled.columns:
        raise ValueError("labelled frame must contain a 'label' column (0/1)")

    train, test = _walk_forward_split(labelled, test_frac=test_frac)
    X_train = train[FEATURE_COLUMNS].to_numpy()
    y_train = train["label"].astype(int).to_numpy()
    X_test = test[FEATURE_COLUMNS].to_numpy()
    y_test = test["label"].astype(int).to_numpy()

    if len(set(y_train)) < 2:
        log.warning(
            "ml.train: training set is single-class (all %d); "
            "the classifier will degenerate to a constant predictor",
            int(y_train[0]) if len(y_train) else -1,
        )

    model = HistGradientBoostingClassifier(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    proba = (
        model.predict_proba(X_test)[:, 1]
        if (hasattr(model, "predict_proba") and len(set(y_train)) > 1)
        else np.zeros_like(y_test, dtype=float)
    )
    preds = (proba >= 0.5).astype(int)

    def _safe(metric, *args, **kwargs) -> float:
        try:
            return float(metric(*args, **kwargs))
        except (ValueError, ZeroDivisionError):
            return float("nan")

    report = TrainReport(
        n_train=len(train),
        n_test=len(test),
        n_features=len(FEATURE_COLUMNS),
        accuracy=_safe(accuracy_score, y_test, preds),
        precision=_safe(precision_score, y_test, preds, zero_division=0),
        recall=_safe(recall_score, y_test, preds, zero_division=0),
        f1=_safe(f1_score, y_test, preds, zero_division=0),
        roc_auc=(
            _safe(roc_auc_score, y_test, proba)
            if len(set(y_test)) > 1 else float("nan")
        ),
        feature_importance=_extract_feature_importance(model),
        class_balance={
            "train_pos": int((y_train == 1).sum()),
            "train_neg": int((y_train == 0).sum()),
            "test_pos":  int((y_test == 1).sum()),
            "test_neg":  int((y_test == 0).sum()),
        },
    )
    return ProfitabilityClassifier(model=model, features=list(FEATURE_COLUMNS)), report


def _extract_feature_importance(model) -> Dict[str, float]:
    """Best-effort importance extraction.

    HistGradientBoosting in newer sklearn exposes ``model.feature_importances_``
    (gini-based proxy from permutation if requested); when missing we
    fall back to an empty dict rather than crashing the trainer.
    """
    fi = getattr(model, "feature_importances_", None)
    if fi is None:
        return {}
    return {name: float(score) for name, score in zip(FEATURE_COLUMNS, fi)}
