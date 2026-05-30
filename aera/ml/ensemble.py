"""Per-symbol ensemble of profitability classifiers.

A single global gradient-boosted classifier (``aera.ml.model``) treats
BTC, ETH, SOL etc. as one population. That's fine when the trade
dataset is small, but as soon as we have a few hundred trades per
symbol the per-symbol microstructure differences (BTC's deep books
vs. SOL's reflexive liquidity, different funding-rate dynamics, etc.)
start to matter more than the marginal sample-size gain from
pooling.

This module fits ONE classifier per symbol that has enough trades,
plus a global fallback fit on everything. At scoring time the
ensemble routes to the per-symbol model and gracefully falls back to
the global one when the per-symbol model is missing — so the
strategy stays operational for newly-added symbols immediately.

Persistence layout on disk::

    <root>/
      fallback.joblib              ← global model (always trained)
      per_symbol/
        BTCUSD.joblib
        ETHUSD.joblib
        SOLUSD.joblib
        ...

This module is import-light — it doesn't pull torch — and is the
"baseline upgrade" of the ML stack. Sequence / RL scorers in
``aera.ml.sequence`` and ``aera.ml.rl`` plug in alongside via
``aera.ml.registry.ModelRegistry``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from aera.logging import get_logger

from .features import FEATURE_COLUMNS
from .model import ProfitabilityClassifier, TrainReport, train_model

log = get_logger(__name__)


@dataclass
class EnsembleReport:
    """Diagnostic bundle from :func:`train_ensemble`."""

    fallback: TrainReport
    per_symbol: Dict[str, TrainReport] = field(default_factory=dict)
    skipped_symbols: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fallback": self.fallback.to_dict(),
            "per_symbol": {k: v.to_dict() for k, v in self.per_symbol.items()},
            "skipped_symbols": dict(self.skipped_symbols),
        }


class EnsembleClassifier:
    """A pool of per-symbol ``ProfitabilityClassifier``s + a global
    fallback. At scoring time the ensemble routes by ``symbol``.

    Construct via :func:`train_ensemble` (offline) or
    :func:`load_ensemble` (live). Prefer the latter in the live
    strategy — the former requires sklearn + a labelled trade
    dataframe.
    """

    DIR_PER_SYMBOL = "per_symbol"
    FALLBACK_FILE = "fallback.joblib"
    META_FILE = "ensemble.json"

    def __init__(
        self,
        fallback: Optional[ProfitabilityClassifier] = None,
        per_symbol: Optional[Dict[str, ProfitabilityClassifier]] = None,
    ) -> None:
        self.fallback = fallback
        self.per_symbol: Dict[str, ProfitabilityClassifier] = dict(per_symbol or {})

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def has_symbol(self, symbol: str) -> bool:
        return symbol.upper() in self.per_symbol

    def predict_proba_win(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        symbol: Optional[str] = None,
    ) -> np.ndarray:
        """Return ``P(win)`` for each row.

        Routes by ``symbol`` — when the per-symbol model exists it's
        used, otherwise the fallback. If neither exists the call
        raises (the caller should treat the ensemble as missing).
        """
        sym = (symbol or "").upper()
        model = self.per_symbol.get(sym) or self.fallback
        if model is None:
            raise RuntimeError(
                "EnsembleClassifier has neither a per-symbol model for "
                f"{sym!r} nor a fallback — nothing to score against."
            )
        return model.predict_proba_win(X)

    def models_used(self, symbols: Iterable[str]) -> Dict[str, str]:
        """For each symbol return which member would score it (``"per_symbol"``,
        ``"fallback"``, or ``"none"``). Useful for logging."""
        out: Dict[str, str] = {}
        for s in symbols:
            sym = s.upper()
            if sym in self.per_symbol:
                out[sym] = "per_symbol"
            elif self.fallback is not None:
                out[sym] = "fallback"
            else:
                out[sym] = "none"
        return out

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, root: str | Path) -> Path:
        """Persist the ensemble under ``root/`` (creates the dir)."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if self.fallback is not None:
            self.fallback.save(root / self.FALLBACK_FILE)
        ps_dir = root / self.DIR_PER_SYMBOL
        ps_dir.mkdir(parents=True, exist_ok=True)
        for sym, clf in self.per_symbol.items():
            clf.save(ps_dir / f"{sym.upper()}.joblib")
        meta = {
            "fallback": self.fallback is not None,
            "per_symbol": sorted(self.per_symbol.keys()),
        }
        (root / self.META_FILE).write_text(json.dumps(meta, indent=2))
        return root


def load_ensemble(root: str | Path) -> EnsembleClassifier:
    """Load an :class:`EnsembleClassifier` from disk. Tolerant of a
    partial layout — a missing fallback or empty per_symbol/ is
    treated as "that side is absent"."""
    from .model import load_model
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ensemble directory does not exist: {root}")

    fallback: Optional[ProfitabilityClassifier] = None
    fb_path = root / EnsembleClassifier.FALLBACK_FILE
    if fb_path.exists():
        try:
            fallback = load_model(fb_path)
        except Exception as exc:
            log.warning("ensemble: fallback load failed at %s: %s", fb_path, exc)

    per_symbol: Dict[str, ProfitabilityClassifier] = {}
    ps_dir = root / EnsembleClassifier.DIR_PER_SYMBOL
    if ps_dir.exists():
        for fp in ps_dir.glob("*.joblib"):
            sym = fp.stem.upper()
            try:
                per_symbol[sym] = load_model(fp)
            except Exception as exc:
                log.warning("ensemble: per-symbol load failed for %s: %s", sym, exc)

    log.info(
        "ensemble: loaded fallback=%s per_symbol=%s",
        "yes" if fallback else "no",
        sorted(per_symbol.keys()),
    )
    return EnsembleClassifier(fallback=fallback, per_symbol=per_symbol)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_ensemble(
    labelled: pd.DataFrame,
    *,
    min_per_symbol: int = 200,
    test_frac: float = 0.20,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[EnsembleClassifier, EnsembleReport]:
    """Fit a global fallback + one classifier per symbol that has at
    least ``min_per_symbol`` rows.

    The labelled dataframe is the output of
    ``aera.ml.features.label_trades`` (i.e. ``FEATURE_COLUMNS`` +
    ``label`` + ``symbol``).
    """
    if labelled.empty:
        raise ValueError("labelled trade frame is empty")
    if "symbol" not in labelled.columns:
        raise ValueError("labelled frame must contain a 'symbol' column")

    fallback_model, fallback_report = train_model(
        labelled,
        test_frac=test_frac,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    per_symbol: Dict[str, ProfitabilityClassifier] = {}
    per_reports: Dict[str, TrainReport] = {}
    skipped: Dict[str, int] = {}

    for sym, group in labelled.groupby("symbol"):
        sym = str(sym).upper()
        if len(group) < min_per_symbol:
            skipped[sym] = len(group)
            log.info(
                "ensemble: skipping %s (only %d rows < min_per_symbol=%d) "
                "— fallback will handle it",
                sym, len(group), min_per_symbol,
            )
            continue
        try:
            clf, rep = train_model(
                group,
                test_frac=test_frac,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        except ValueError as exc:
            # Single-class sub-population — fall through to fallback.
            skipped[sym] = len(group)
            log.warning("ensemble: %s train failed (%s); falling back", sym, exc)
            continue
        per_symbol[sym] = clf
        per_reports[sym] = rep
        log.info(
            "ensemble: fit %s on %d rows  acc=%.3f  roc=%.3f",
            sym, len(group), rep.accuracy, rep.roc_auc,
        )

    ensemble = EnsembleClassifier(fallback=fallback_model, per_symbol=per_symbol)
    report = EnsembleReport(
        fallback=fallback_report,
        per_symbol=per_reports,
        skipped_symbols=skipped,
    )
    return ensemble, report
