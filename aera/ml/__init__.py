"""Machine-learning layer for the money-printer strategy."""
from .features import (
    FEATURE_COLUMNS,
    FeatureExtractor,
    extract_features,
    label_trades,
)
from .model import (
    ProfitabilityClassifier,
    TrainReport,
    load_model,
    train_model,
)
from .ensemble import (
    EnsembleClassifier,
    EnsembleReport,
    load_ensemble,
    train_ensemble,
)
from .registry import (
    FusionWeights,
    GBTScorer,
    EnsembleScorer,
    ModelRegistry,
    RLWrappedScorer,
    Scorer,
    ScoringContext,
    SequenceWrappedScorer,
)
from .sequence import (
    SEQUENCE_FEATURES,
    SequenceConfig,
    SequenceScorer,
    SequenceTrainReport,
    bars_to_sequence_matrix,
    build_sequences_from_trades,
    load_sequence_scorer,
    train_sequence_model,
)
from .rl import (
    DQNAgent,
    DQNConfig,
    RLScorer,
    TradingEnv,
    TradingEnvConfig,
    TrainStats,
    load_rl_scorer,
    train_dqn_agent,
)

__all__ = [
    # features
    "FEATURE_COLUMNS",
    "FeatureExtractor",
    "extract_features",
    "label_trades",
    # GBT model
    "ProfitabilityClassifier",
    "TrainReport",
    "load_model",
    "train_model",
    # ensemble
    "EnsembleClassifier",
    "EnsembleReport",
    "load_ensemble",
    "train_ensemble",
    # registry
    "FusionWeights",
    "GBTScorer",
    "EnsembleScorer",
    "ModelRegistry",
    "RLWrappedScorer",
    "Scorer",
    "ScoringContext",
    "SequenceWrappedScorer",
    # sequence model
    "SEQUENCE_FEATURES",
    "SequenceConfig",
    "SequenceScorer",
    "SequenceTrainReport",
    "bars_to_sequence_matrix",
    "build_sequences_from_trades",
    "load_sequence_scorer",
    "train_sequence_model",
    # RL
    "DQNAgent",
    "DQNConfig",
    "RLScorer",
    "TradingEnv",
    "TradingEnvConfig",
    "TrainStats",
    "load_rl_scorer",
    "train_dqn_agent",
]
