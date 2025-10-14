from __future__ import annotations
import os, json, math
from dataclasses import asdict, is_dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from codes.run_logger import RunLogger

@torch.no_grad()
def _evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.append(out.detach().float().cpu())
        ys.append(batch.y.detach().float().cpu().view(-1))
    y = torch.cat(ys).numpy()
    p = torch.cat(preds).numpy()
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(math.sqrt(np.mean((y - p) ** 2)))
    ybar = float(np.mean(y))
    ss_tot = float(np.sum((y - ybar) ** 2)) + 1e-12
    ss_res = float(np.sum((y - p) ** 2))
    r2 = 1.0 - (ss_res / ss_tot)
    return {"rmse": rmse, "mae": mae, "r2": r2}

class Trainer:

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 80,
        patience: int = 10,
        run_dir: Optional[str] = None,
        grad_clip_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        verbose: bool = True,
    ):
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = run_dir

    def _maybe_save_config(self, cfg_obj) -> None:
        if self.run_dir is None or cfg_obj is None:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        if is_dataclass(cfg_obj):
            payload = asdict(cfg_obj)
        elif isinstance(cfg_obj, dict):
            payload = cfg_obj
        else:
            return
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(payload, f, indent=2)

    # codes/training/trainer.py  (only diffs shown; paste into your file)

    def fit(
            self,
            model: nn.Module,
            loaders: Dict[str, DataLoader],
            model_head_path: Optional[str] = None,
            predictor_path: Optional[str] = None,
            save_best: bool = True,
            cfg_for_logging: Optional[object] = None,
            run_logger: Optional[RunLogger] = None,
    ) -> Dict[str, float]:
        model = model.to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_rmse, best_state, wait = float("inf"), None, 0
        history = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in loaders["train"]:
                batch = batch.to(self.device)
                opt.zero_grad(set_to_none=True)
                pred = model(batch).view(-1)
                loss = criterion(pred, batch.y.view(-1).float())
                loss.backward()
                if self.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= max(1, len(loaders["train"]))

            val_metrics = _evaluate_regression(model, loaders["val"], self.device)

            if run_logger is not None:
                run_logger.log_metrics(step=epoch, train_loss=epoch_loss,
                                       val_rmse=val_metrics["rmse"], val_mae=val_metrics["mae"],
                                       val_r2=val_metrics["r2"])

            history.append({
                "epoch": epoch,
                "train_loss": float(epoch_loss),
                "val_rmse": float(val_metrics["rmse"]),
                "val_mae": float(val_metrics["mae"]),
                "val_r2": float(val_metrics["r2"]),
            })

            rmse = val_metrics["rmse"]
            if self.verbose:
                print(f"[{epoch:03d}] train_loss={epoch_loss:.4f}  val_rmse={rmse:.4f}  "
                      f"val_mae={val_metrics['mae']:.4f}  r2={val_metrics['r2']:.4f}")

            improved = rmse < best_rmse - 1e-6
            if improved:
                best_rmse = rmse
                wait = 0
                if save_best:
                    best_state = {
                        "model": model.state_dict(),
                        "best_val": val_metrics,
                        "epoch": epoch,
                    }
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}, best val RMSE {best_rmse:.4f}")
                    break

        if save_best and self.run_dir is not None and best_state is not None:
            os.makedirs(self.run_dir, exist_ok=True)
            torch.save(best_state["model"], os.path.join(self.run_dir, "weights.pth"))
            if hasattr(model, "model_head") and hasattr(model.model_head, "state_dict"):
                torch.save(model.model_head.state_dict(), os.path.join(self.run_dir, "model_head.pth"))
            if hasattr(model, "predictor") and hasattr(model.predictor, "state_dict"):
                torch.save(model.predictor.state_dict(), os.path.join(self.run_dir, "predictor.pth"))
            self._maybe_save_config(cfg_for_logging)
            if cfg_for_logging is not None and run_logger is not None:
                run_logger.save_json("config.json", cfg_for_logging)

        final_metrics = _evaluate_regression(model, loaders["val"], self.device)

        if run_logger is not None:
            run_logger.finalize(score_name="val_rmse", score_value=final_metrics["rmse"], extra=final_metrics)
            run_logger.close()

        return {
            "val_rmse": float(final_metrics["rmse"]),
            "val_mae": float(final_metrics["mae"]),
            "val_r2": float(final_metrics["r2"]),
            "history": history,
        }
