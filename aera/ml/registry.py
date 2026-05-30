"""Multi-model scorer registry for MoneyPrinter.

The strategy used to call a single ``ProfitabilityClassifier`` with
a feature row and gate on ``P(win)``. As soon as we have more than
one ML asset (per-symbol ensemble, sequence-model encoder, RL
policy) we need a uniform way to:

1. Auto-discover whatever artefacts exist on disk.
2. Score the candidate fire with each.
3. Fuse the scores into one ``P(win)`` the strategy can gate on.

That's all this module does. Each scorer is OPT-IN — the registry
loads what it finds and ignores what isn't there. Every scorer
ALSO implements graceful degradation (returns 0.5 = no opinion
when its model isn't loaded), so a partially-trained pipeline
behaves the same as a fully-trained one, just with less precision.

Default disk layout under ``data/money_printer/``::

    model.joblib                       ← single global GBT (legacy)
    ensemble/
      fallback.joblib
      per_symbol/
        BTCUSD.joblib
        ETHUSD.joblib
        ...
    sequence_model.pt                  ← transformer encoder (optional)
    sequence_model.pt.meta.json
    rl_policy.npz                      ← DQN policy (optional)

Adding a new scorer is two changes:

1. Subclass :class:`Scorer` with a ``score(ctx) -> float`` method.
2. Add a load step in :meth:`ModelRegistry.from_dir` that probes
   the expected path and instantiates the scorer.

MoneyPrinter will pick the new scorer up automatically on next
construction. No edits to the strategy needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from aera.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# scoring context
# ---------------------------------------------------------------------------


@dataclass
class ScoringContext:
    """Bundle of inputs every scorer might want.

    Each scorer reads what it needs (and ignores the rest), so we
    can introduce richer modalities (orderbook snapshots, news
    sentiment embeddings, ...) without touching the registry's
    plumbing.
    """

    features: Optional[pd.Series] = None        # 15 GBT features
    candle_window: Optional[pd.DataFrame] = None  # last N OHLCV bars for sequence model
    symbol: str = ""
    side: str = ""                              # "BUY" / "SELL" / "" (unknown)


# ---------------------------------------------------------------------------
# scorer protocol + concrete adapters
# ---------------------------------------------------------------------------


class Scorer:
    """Interface every member of the registry implements.

    Subclasses must populate ``name`` and implement :meth:`score`.
    Optional :meth:`available` lets the registry skip a scorer
    cleanly when its deps aren't installed (e.g. torch missing).
    """

    name: str = "scorer"
    default_weight: float = 1.0

    def available(self) -> bool:
        return True

    def score(self, ctx: ScoringContext) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


class GBTScorer(Scorer):
    """Wraps a single :class:`aera.ml.ProfitabilityClassifier`."""

    name = "gbt"

    def __init__(self, classifier) -> None:
        self.classifier = classifier

    def score(self, ctx: ScoringContext) -> float:
        if self.classifier is None or ctx.features is None:
            return 0.5
        try:
            arr = self.classifier.predict_proba_win(ctx.features)
            return float(arr[0]) if hasattr(arr, "__len__") else float(arr)
        except Exception as exc:  # pragma: no cover - safety net
            log.debug("gbt scorer failed: %s", exc)
            return 0.5


class EnsembleScorer(Scorer):
    """Per-symbol :class:`aera.ml.ensemble.EnsembleClassifier`."""

    name = "ensemble"

    def __init__(self, ensemble) -> None:
        self.ensemble = ensemble

    def score(self, ctx: ScoringContext) -> float:
        if self.ensemble is None or ctx.features is None:
            return 0.5
        try:
            arr = self.ensemble.predict_proba_win(ctx.features, symbol=ctx.symbol)
            return float(arr[0]) if hasattr(arr, "__len__") else float(arr)
        except Exception as exc:  # pragma: no cover
            log.debug("ensemble scorer failed: %s", exc)
            return 0.5


class SequenceWrappedScorer(Scorer):
    """Wraps an :class:`aera.ml.sequence.SequenceScorer`. The
    underlying class already returns 0.5 when torch is missing or
    the model isn't loaded, so this adapter is mostly plumbing."""

    name = "sequence"
    default_weight = 0.75

    def __init__(self, sequence_scorer) -> None:
        self.sequence_scorer = sequence_scorer

    def available(self) -> bool:
        from .sequence import SequenceScorer  # local import — keeps torch optional
        return SequenceScorer.available() and self.sequence_scorer is not None

    def score(self, ctx: ScoringContext) -> float:
        if self.sequence_scorer is None or ctx.candle_window is None:
            return 0.5
        try:
            return float(self.sequence_scorer.predict_proba_win(ctx.candle_window))
        except Exception as exc:  # pragma: no cover
            log.debug("sequence scorer failed: %s", exc)
            return 0.5


class RLWrappedScorer(Scorer):
    """Wraps an :class:`aera.ml.rl.RLScorer` (DQN policy)."""

    name = "rl"
    default_weight = 0.5

    def __init__(self, rl_scorer) -> None:
        self.rl_scorer = rl_scorer

    def score(self, ctx: ScoringContext) -> float:
        if self.rl_scorer is None or ctx.features is None:
            return 0.5
        try:
            return float(self.rl_scorer.predict_proba_win(ctx.features))
        except Exception as exc:  # pragma: no cover
            log.debug("rl scorer failed: %s", exc)
            return 0.5


# ---------------------------------------------------------------------------
# the registry itself
# ---------------------------------------------------------------------------


@dataclass
class FusionWeights:
    """Per-scorer weights for the fused P(win). Anything missing
    from the registry contributes 0 to the weighted average."""

    gbt: float = 1.0
    ensemble: float = 1.5
    sequence: float = 0.75
    rl: float = 0.5

    def for_(self, name: str) -> float:
        return float(getattr(self, name, 0.0))


@dataclass
class ModelRegistry:
    """Holds a heterogeneous set of scorers and fuses their outputs.

    Construct via :meth:`from_dir` to auto-load whatever is on disk,
    or build one manually for tests by adding :class:`Scorer`
    instances directly.
    """

    scorers: List[Scorer] = field(default_factory=list)
    weights: FusionWeights = field(default_factory=FusionWeights)

    def __len__(self) -> int:
        return len(self.scorers)

    def names(self) -> List[str]:
        return [s.name for s in self.scorers]

    def add(self, scorer: Scorer) -> None:
        if not scorer.available():
            log.info(
                "registry: skipping unavailable scorer %s (%s)",
                scorer.name,
                getattr(scorer, "reason_unavailable", lambda: "")() or "no reason",
            )
            return
        self.scorers.append(scorer)

    def score_all(self, ctx: ScoringContext) -> Dict[str, float]:
        return {s.name: float(s.score(ctx)) for s in self.scorers}

    def combined(self, ctx: ScoringContext) -> tuple[float, Dict[str, float]]:
        """Return ``(fused_p_win, {scorer_name: p_win})``. The fused
        score is a weighted arithmetic mean of the available
        scorers' outputs; if no scorer is available the fused
        score is 0.5 (no opinion)."""
        per = self.score_all(ctx)
        if not per:
            return 0.5, per
        num = 0.0
        denom = 0.0
        for name, p in per.items():
            w = self.weights.for_(name)
            num += w * p
            denom += w
        fused = num / denom if denom > 0 else 0.5
        return float(fused), per

    # ------------------------------------------------------------------
    # auto-discovery
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(
        cls,
        root: str | Path,
        *,
        weights: Optional[FusionWeights] = None,
        gbt_filename: str = "model.joblib",
        ensemble_dirname: str = "ensemble",
        sequence_filename: str = "sequence_model.pt",
        rl_filename: str = "rl_policy.npz",
    ) -> "ModelRegistry":
        """Probe ``root/`` for known artefact names and build the
        registry with whatever is present. Always succeeds — a
        missing artefact just means that scorer is omitted.
        """
        root = Path(root)
        registry = cls(weights=weights or FusionWeights())

        # ---- single-model GBT (legacy money_printer) -----------------
        gbt_path = root / gbt_filename
        if gbt_path.exists():
            try:
                from .model import load_model
                registry.add(GBTScorer(load_model(gbt_path)))
                log.info("registry: loaded GBT model from %s", gbt_path)
            except Exception as exc:
                log.warning("registry: failed to load GBT model: %s", exc)

        # ---- per-symbol ensemble -------------------------------------
        ens_dir = root / ensemble_dirname
        if ens_dir.exists():
            try:
                from .ensemble import load_ensemble
                registry.add(EnsembleScorer(load_ensemble(ens_dir)))
                log.info("registry: loaded ensemble from %s", ens_dir)
            except Exception as exc:
                log.warning("registry: failed to load ensemble: %s", exc)

        # ---- sequence model (torch optional) -------------------------
        seq_path = root / sequence_filename
        if seq_path.exists():
            try:
                from .sequence import SequenceScorer, load_sequence_scorer
                if SequenceScorer.available():
                    registry.add(SequenceWrappedScorer(load_sequence_scorer(seq_path)))
                    log.info("registry: loaded sequence model from %s", seq_path)
                else:
                    log.info(
                        "registry: sequence model file present at %s but %s",
                        seq_path, SequenceScorer.reason_unavailable(),
                    )
            except Exception as exc:
                log.warning("registry: failed to load sequence model: %s", exc)

        # ---- RL agent ------------------------------------------------
        rl_path = root / rl_filename
        if rl_path.exists():
            try:
                from .rl import load_rl_scorer
                registry.add(RLWrappedScorer(load_rl_scorer(rl_path)))
                log.info("registry: loaded RL policy from %s", rl_path)
            except Exception as exc:
                log.warning("registry: failed to load RL policy: %s", exc)

        log.info(
            "registry: %d scorer(s) active: %s",
            len(registry), registry.names() or ["<none>"],
        )
        return registry
