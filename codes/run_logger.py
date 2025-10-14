from __future__ import annotations
import os, json, time, hashlib
from dataclasses import asdict, is_dataclass
from typing import Optional, Dict, Any

def _now_stamp():
    return time.strftime("%Y%m%d-%H%M%S")

def _short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:6]

class RunLogger:

    def __init__(self, base_dir: str, tag: str, params: Dict[str, Any] | None = None):
        os.makedirs(base_dir, exist_ok=True)
        uid = f"{_now_stamp()}-{_short_hash(tag)}"
        self.run_dir = os.path.join(base_dir, uid)
        os.makedirs(self.run_dir, exist_ok=True)

        if params is not None:
            self.save_json("params.json", params)

        self._metrics_file = open(os.path.join(self.run_dir, "metrics.jsonl"), "a", encoding="utf-8")
        self.tag = tag
        self.base_dir = base_dir

    def path(self, *parts: str) -> str:
        p = os.path.join(self.run_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def save_json(self, name: str, payload: Dict[str, Any] | object):
        if is_dataclass(payload): payload = asdict(payload)
        with open(self.path(name), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def log_metrics(self, step: int, **metrics):
        row = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        self._metrics_file.write(json.dumps(row) + "\n")
        self._metrics_file.flush()

    def save_artifact(self, src_path: str, name: Optional[str] = None):
        if not os.path.exists(src_path): return
        import shutil
        dst = self.path(name or os.path.basename(src_path))
        shutil.copy2(src_path, dst)
        return dst

    def finalize(self, score_name: str, score_value: float, extra: Dict[str, Any] | None = None):
        final = {"score_name": score_name, "score_value": float(score_value)}
        if extra: final.update(extra)
        self.save_json("final.json", final)

        lb_path = os.path.join(self.base_dir, "leaderboard.jsonl")
        with open(lb_path, "a", encoding="utf-8") as f:
            row = {
                "tag": self.tag,
                "run_dir": os.path.abspath(self.run_dir),
                "score_name": score_name,
                "score_value": float(score_value),
                "timestamp": _now_stamp(),
            }
            if extra: row.update(extra)
            f.write(json.dumps(row) + "\n")

    def close(self):
        try:
            self._metrics_file.close()
        except Exception:
            pass
