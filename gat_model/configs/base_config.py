import os, json, shutil
from datetime import datetime
from typing import Optional


class BaseConfig:
    @staticmethod
    def get_model_folder(experiment_name: str, model_name: Optional[str] = None, model_root: str = "runs") -> str:
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
    def get_registry_folder(registry_root: str = "best_gat_models") -> str:
        os.makedirs(registry_root, exist_ok=True)
        return registry_root

    @staticmethod
    def promote_run_to_registry(src_run_dir: str, tag: str, score_name: str, score_value: float,
                                registry_root: str = "best_gat_models") -> str:
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
            "tag": tag, "score_name": score_name, "score_value": float(score_value),
            "source_run_dir": os.path.abspath(src_run_dir), "registry_dir": os.path.abspath(dest),
            "timestamp": stamp,
        }
        with open(os.path.join(dest, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(registry_root, "leaderboard.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")
        return dest
