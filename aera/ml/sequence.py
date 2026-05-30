"""Sequence model — small transformer encoder over recent bars.

The gradient-boosted tree in :mod:`aera.ml.model` sees a SINGLE 15-
feature vector representing the state at the candidate-fire bar. It
can't model the SHAPE of the lead-up: did we ramp into this RSI=70
slowly (probably continues), or did we spike there in one bar
(probably reverses)?

This module fits a tiny Transformer encoder over the last ``N`` bars
of OHLCV-derived features and predicts ``P(win)`` for the current
setup. Architecture is deliberately small (8 attention heads × 32
dims × 2 layers, ~10k params) so it trains in a couple of minutes
on a laptop and scores in microseconds.

Torch is an OPTIONAL dependency. If ``import torch`` fails the
module still imports — ``SequenceScorer.available()`` returns
``False`` and the live strategy treats this scorer as absent. The
test suite skips torch-dependent tests in that case. To enable::

    pip install torch>=2.0

Input shape: ``(batch, seq_len, n_features)`` where ``n_features``
is the OHLCV-derived per-bar feature vector (close return, log-vol,
upper/lower wick, body, hour_sin, hour_cos) — 7 dims per bar; the
default ``seq_len=64`` gives a model that sees roughly the last hour
of 1-minute bars.

Disk format::

    <path>.pt          ← torch state_dict
    <path>.meta.json   ← {"seq_len": 64, "n_features": 7, "config": ...}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from aera.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# soft torch import
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in CI without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# per-bar feature row used by the sequence model
# ---------------------------------------------------------------------------


SEQUENCE_FEATURES: List[str] = [
    "ret",          # close-to-close return
    "log_vol",      # log(1 + volume)
    "upper_wick",   # (high - max(open,close)) / range
    "lower_wick",   # (min(open,close) - low) / range
    "body",         # (close - open) / range  (signed)
    "hour_sin",
    "hour_cos",
]


def bars_to_sequence_matrix(candles: pd.DataFrame) -> np.ndarray:
    """Convert an OHLCV candle DataFrame to a per-bar feature matrix
    of shape ``(n_bars, len(SEQUENCE_FEATURES))``.

    NaN rows (from the first bar where ret is undefined) are
    forward-filled with 0 — the sequence model handles zero-padding
    natively (no warm-up gate needed at the dataset boundary).
    """
    if candles is None or candles.empty:
        return np.zeros((0, len(SEQUENCE_FEATURES)), dtype=np.float32)

    df = candles.copy().reset_index(drop=True)
    close = df["close"].astype(float).to_numpy()
    open_ = df["open"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    volume = df.get("volume", pd.Series([0.0] * len(df))).astype(float).to_numpy()
    ts = df["ts"].astype("int64").to_numpy()

    eps = 1e-12
    rng = np.maximum(high - low, eps)
    ret = np.zeros_like(close)
    if len(close) > 1:
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + eps)

    upper_wick = (high - np.maximum(open_, close)) / rng
    lower_wick = (np.minimum(open_, close) - low) / rng
    body = (close - open_) / rng
    log_vol = np.log1p(np.abs(volume))
    hours = (ts // 3600) % 24
    hour_sin = np.sin(2.0 * math.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * math.pi * hours / 24.0)

    mat = np.stack(
        [ret, log_vol, upper_wick, lower_wick, body, hour_sin, hour_cos],
        axis=1,
    ).astype(np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat


# ---------------------------------------------------------------------------
# model definition (torch optional)
# ---------------------------------------------------------------------------


@dataclass
class SequenceConfig:
    seq_len: int = 64
    n_features: int = len(SEQUENCE_FEATURES)
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


if _TORCH_AVAILABLE:

    class _SinusoidalPositionalEncoding(nn.Module):
        """Standard transformer sinusoidal positional embedding.
        Pre-computed so it has zero parameters."""

        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.pe[:, : x.size(1), :]


    class _TransformerWinClassifier(nn.Module):
        """Tiny encoder-only transformer with a binary classification head.

        Input  : ``(batch, seq_len, n_features)``
        Output : ``(batch,)`` logits  → sigmoid → P(win)
        """

        def __init__(self, cfg: SequenceConfig):
            super().__init__()
            self.cfg = cfg
            self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
            self.pos_enc = _SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.seq_len)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.head = nn.Linear(cfg.d_model, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.input_proj(x)
            h = self.pos_enc(h)
            h = self.encoder(h)
            # Mean-pool across the time dimension before the head — more
            # stable than CLS-token pooling on small datasets.
            pooled = h.mean(dim=1)
            logit = self.head(pooled).squeeze(-1)
            return logit


# ---------------------------------------------------------------------------
# public scorer wrapper
# ---------------------------------------------------------------------------


class SequenceScorer:
    """High-level wrapper used by the registry. Always importable
    regardless of torch presence — when torch is absent, every
    ``predict_proba_win`` call returns 0.5 (no opinion).
    """

    def __init__(
        self,
        config: Optional[SequenceConfig] = None,
        model=None,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config or SequenceConfig()
        self.model = model
        # Per-feature mean / stdev for input normalisation, fit on
        # the training set. Stored alongside the weights so live
        # scoring sees the exact same distribution as training.
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std

    @staticmethod
    def available() -> bool:
        return _TORCH_AVAILABLE

    @staticmethod
    def reason_unavailable() -> str:
        return "" if _TORCH_AVAILABLE else (
            "torch not installed (pip install torch) — sequence model "
            "scorer disabled, only GBT / ensemble / RL scorers will fire."
        )

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def _prep_batch(self, X: np.ndarray) -> np.ndarray:
        """Normalise + reshape an (seq_len, n_features) matrix into a
        (1, seq_len, n_features) batch ready for the model."""
        if X.ndim == 2:
            X = X[np.newaxis, ...]  # add batch dim
        if X.shape[1] < self.config.seq_len:
            pad = np.zeros(
                (X.shape[0], self.config.seq_len - X.shape[1], X.shape[2]),
                dtype=X.dtype,
            )
            X = np.concatenate([pad, X], axis=1)
        elif X.shape[1] > self.config.seq_len:
            X = X[:, -self.config.seq_len:, :]
        if self.scaler_mean is not None and self.scaler_std is not None:
            X = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        return X.astype(np.float32)

    def predict_proba_win(self, sequence: np.ndarray | pd.DataFrame) -> float:
        """Return P(win) for a single sequence (numpy array or candles
        DataFrame). Returns 0.5 when torch is unavailable or the model
        hasn't been trained yet."""
        if not _TORCH_AVAILABLE or self.model is None:
            return 0.5
        if isinstance(sequence, pd.DataFrame):
            mat = bars_to_sequence_matrix(sequence)
        else:
            mat = np.asarray(sequence, dtype=np.float32)
        if mat.size == 0:
            return 0.5
        batch = self._prep_batch(mat)
        with torch.no_grad():
            self.model.eval()
            tensor = torch.from_numpy(batch)
            logit = self.model(tensor)
            prob = torch.sigmoid(logit).item()
        return float(prob)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist model state + config + scaler. Requires torch."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch not installed; cannot save sequence model")
        if self.model is None:
            raise RuntimeError("model is uninitialised; nothing to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        meta = {
            "config": {
                "seq_len": self.config.seq_len,
                "n_features": self.config.n_features,
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "dropout": self.config.dropout,
            },
            "scaler_mean": None if self.scaler_mean is None else self.scaler_mean.tolist(),
            "scaler_std":  None if self.scaler_std is None else self.scaler_std.tolist(),
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
        return path


def load_sequence_scorer(path: str | Path) -> SequenceScorer:
    """Load a saved sequence model + metadata.

    Raises ``ImportError`` when torch is unavailable. Callers that
    want to keep going without the model should check
    ``SequenceScorer.available()`` first.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(SequenceScorer.reason_unavailable())
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"sequence model metadata missing: {meta_path}")
    meta = json.loads(meta_path.read_text())
    cfg = SequenceConfig(**meta["config"])
    model = _TransformerWinClassifier(cfg)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32) if meta.get("scaler_mean") else None
    scaler_std = np.array(meta["scaler_std"], dtype=np.float32) if meta.get("scaler_std") else None
    return SequenceScorer(config=cfg, model=model, scaler_mean=scaler_mean, scaler_std=scaler_std)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


@dataclass
class SequenceTrainReport:
    n_train: int
    n_test: int
    train_loss: float
    test_loss: float
    test_accuracy: float
    test_roc_auc: float = float("nan")
    epochs: int = 0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
            "test_roc_auc": self.test_roc_auc,
            "epochs": self.epochs,
            "config": self.config,
        }


def build_sequences_from_trades(
    trades_df: pd.DataFrame,
    candles_by_symbol: dict[str, pd.DataFrame],
    *,
    seq_len: int = 64,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build the (X, y, symbols) supervised dataset.

    For each trade in ``trades_df`` (must have ``entry_ts`` +
    ``symbol`` + ``is_win`` columns), grab the last ``seq_len`` bars
    of features ENDING at the bar that contains ``entry_ts``. The
    label is ``int(is_win)``.

    Skips trades whose symbol isn't in ``candles_by_symbol`` and
    trades that don't have ``seq_len`` warm-up bars of history.
    """
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    syms: List[str] = []

    seq_cache: dict[str, np.ndarray] = {}
    ts_cache: dict[str, np.ndarray] = {}
    for sym, candles in candles_by_symbol.items():
        if candles is None or candles.empty:
            continue
        seq_cache[sym] = bars_to_sequence_matrix(candles)
        ts_cache[sym] = candles["ts"].astype("int64").to_numpy()

    for _, t in trades_df.iterrows():
        sym = str(t.get("symbol", "")).upper()
        mat = seq_cache.get(sym)
        ts_arr = ts_cache.get(sym)
        if mat is None or ts_arr is None:
            continue
        idx = int(np.searchsorted(ts_arr, int(t["entry_ts"]), side="right") - 1)
        if idx < seq_len:
            continue
        Xs.append(mat[idx - seq_len + 1 : idx + 1])
        ys.append(int(bool(t.get("is_win", t.get("label", 0)))))
        syms.append(sym)
    if not Xs:
        return (
            np.zeros((0, seq_len, len(SEQUENCE_FEATURES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.int64), syms


def train_sequence_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: Optional[SequenceConfig] = None,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    test_frac: float = 0.20,
    device: str = "cpu",
) -> Tuple[SequenceScorer, SequenceTrainReport]:
    """Train the transformer encoder on ``(X, y)`` pairs.

    ``X`` is ``(N, seq_len, n_features)`` and ``y`` is ``(N,)`` 0/1.
    Walk-forward split — the last ``test_frac`` of rows held out.

    Raises ``ImportError`` when torch is missing.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(SequenceScorer.reason_unavailable())
    if X.shape[0] == 0:
        raise ValueError("training set is empty")
    cfg = config or SequenceConfig(seq_len=X.shape[1], n_features=X.shape[2])

    # Walk-forward split (the dataset builder already preserves order).
    n = X.shape[0]
    cut = max(1, int(n * (1 - test_frac))) if n >= 5 else n
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    # Fit scaler on TRAIN ONLY to avoid leakage.
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8

    def _norm(a: np.ndarray) -> np.ndarray:
        return (a - mean) / std

    X_train_n = _norm(X_train).astype(np.float32)
    X_test_n = _norm(X_test).astype(np.float32) if X_test.size else X_test

    model = _TransformerWinClassifier(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = None
    if (y_train == 1).sum() > 0 and (y_train == 0).sum() > 0:
        pos_weight = torch.tensor(
            (y_train == 0).sum() / max(1, (y_train == 1).sum()),
            dtype=torch.float32, device=device,
        )

    last_train_loss = float("nan")
    Xt_t = torch.from_numpy(X_train_n).to(device)
    yt_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(Xt_t.shape[0])
        epoch_losses = []
        for i in range(0, Xt_t.shape[0], batch_size):
            batch_idx = perm[i : i + batch_size]
            xb = Xt_t[batch_idx]
            yb = yt_t[batch_idx]
            optim.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
        last_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

    # ---- eval ------------------------------------------------------
    test_loss = float("nan")
    test_acc = float("nan")
    test_auc = float("nan")
    if X_test_n.size:
        with torch.no_grad():
            model.eval()
            xtest = torch.from_numpy(X_test_n).to(device)
            ytest = torch.from_numpy(y_test.astype(np.float32)).to(device)
            logits = model(xtest)
            test_loss = float(F.binary_cross_entropy_with_logits(logits, ytest).item())
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            test_acc = float((preds == y_test).mean())
            try:
                from sklearn.metrics import roc_auc_score
                if len(set(y_test.tolist())) > 1:
                    test_auc = float(roc_auc_score(y_test, probs))
            except Exception:
                pass

    scorer = SequenceScorer(
        config=cfg, model=model,
        scaler_mean=mean.astype(np.float32),
        scaler_std=std.astype(np.float32),
    )
    report = SequenceTrainReport(
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        train_loss=last_train_loss,
        test_loss=test_loss,
        test_accuracy=test_acc,
        test_roc_auc=test_auc,
        epochs=epochs,
        config=dict(
            seq_len=cfg.seq_len, n_features=cfg.n_features,
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.n_layers, dropout=cfg.dropout,
        ),
    )
    return scorer, report
