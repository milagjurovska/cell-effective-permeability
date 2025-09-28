from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from gin_model.data import DESC_COLS


@dataclass
class GraphStackConfig:
    in_dim: int
    hidden_dim: int = 128
    dropout: float = 0.2


@dataclass
class PredictorConfig:
    pred_input_channels: int = 128
    desc_dim: int = len(DESC_COLS)
    hidden_dim: int = 128
    dropout: float = 0.2


@dataclass
class ModelConfig:
    model_head: GraphStackConfig
    predictor: PredictorConfig
    model_head_path: Optional[str] = None
    predictor_path: Optional[str] = None
    freeze_head: bool = False
