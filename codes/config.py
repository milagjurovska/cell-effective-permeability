from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os, json, shutil
from datetime import datetime

from codes.data.constants import DESC_COLS

class BaseConfig:
    @staticmethod
    def get_model_folder(experiment_name: str,
                         model_name: Optional[str] = None,
                         model_root: str = "runs") -> str:

        path = os.path.join(model_root, experiment_name)
        if model_name:
            path = os.path.join(path, model_name)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_model_by_name(model_name: str, model_root: str = "runs") -> str:
        path = os.path.join(model_root, model_name)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")
        return path

    @staticmethod
    def get_registry_folder(registry_root: str = "best_models") -> str:
        """
        NOTE: no longer hard-coded to 'best_gat_models'.
        Pass registry_root='best_gat_models' / 'best_gcn_models' if you want.
        """
        os.makedirs(registry_root, exist_ok=True)
        return registry_root

    @staticmethod
    def promote_run_to_registry(src_run_dir: str,
                                tag: str,
                                score_name: str,
                                score_value: float,
                                registry_root: str = "best_models") -> str:
        """
        Copies model artifacts and writes metadata + leaderboard entry.
        """
        BaseConfig.get_registry_folder(registry_root)
        safe_tag = tag.replace("/", "_").replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(registry_root, f"{safe_tag}-{score_name}{score_value:.4f}-{stamp}")
        os.makedirs(dest, exist_ok=True)

        for fname in ("model_head.pth", "predictor.pth", "config.json"):
            src = os.path.join(src_run_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, fname))

        meta = {
            "tag": tag,
            "score_name": score_name,
            "score_value": float(score_value),
            "source_run_dir": os.path.abspath(src_run_dir),
            "registry_dir": os.path.abspath(dest),
            "timestamp": stamp,
        }
        with open(os.path.join(dest, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(registry_root, "leaderboard.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")
        return dest

@dataclass
class GraphStackConfig:
    in_dim: int
    hidden_dim: int = 128
    heads: int = 4
    dropout: float = 0.2
    attn_dropout: float = 0.0

@dataclass
class PredictorConfig:
    pred_input_channels: int = 128
    desc_dim: int = len(DESC_COLS)
    hidden_dim: int = 128
    dropout: float = 0.2

@dataclass
class ModelConfig:
    # Add the model name so one config can serve GAT/GCN/GIN
    name: str  # "gat" | "gcn" | "gin"
    model_head: GraphStackConfig
    predictor: PredictorConfig
    model_head_path: Optional[str] = None
    predictor_path: Optional[str] = None
    freeze_head: bool = False
