import argparse
import json
from pathlib import Path
from torch_geometric.loader import DataLoader

from codes.training.trainer import Trainer
from codes.models import model_factory
from codes.data.datamodule import DataModule, DataConfig
from codes.data.constants import DESC_COLS
from codes.config import ModelConfig, GraphStackConfig, PredictorConfig


def _safe_number(x):
    try:
        return float(x)
    except Exception:
        try:
            return x.item()
        except Exception:
            return x

def _extract_history(trainer, metrics):
    if isinstance(metrics, dict) and isinstance(metrics.get("history"), list):
        return metrics["history"]
    if hasattr(trainer, "history") and isinstance(trainer.history, list):
        return trainer.history
    return []

def _best_val_rmse_from_history(history):
    best = None
    for rec in history:
        if isinstance(rec, dict) and "val_rmse" in rec:
            v = _safe_number(rec["val_rmse"])
            if isinstance(v, (int, float)):
                best = v if best is None else min(best, v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gat", "gcn", "gin"], default="gat")
    ap.add_argument("--csv", default="data/caco2_processed.csv")
    ap.add_argument("--target", default="target")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    dm = DataModule(DataConfig(csv_path=args.csv, target_col=args.target, batch_size=args.batch_size))
    loaders = dm.loaders_for_fold(args.fold)

    sample = dm.ds[0]
    in_dim = sample.x.size(-1)
    desc_dim = len(DESC_COLS) if hasattr(sample, "desc") else 0

    mcfg = ModelConfig(
        name=args.model,
        model_head=GraphStackConfig(in_dim=in_dim),
        predictor=PredictorConfig(desc_dim=desc_dim),
    )

    ModelCls = model_factory(args.model)
    model = ModelCls.load(mcfg) if hasattr(ModelCls, "load") else ModelCls(mcfg)

    out_dir = Path("runs") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        run_dir=str(out_dir),
    )
    metrics = trainer.fit(model, loaders)

    # History/final metrics
    history = _extract_history(trainer, metrics)
    if history:
        last = history[-1]
        final_metrics = {
            "val_rmse": _safe_number(last.get("val_rmse")),
            "val_mae": _safe_number(last.get("val_mae")),
            "val_r2": _safe_number(last.get("val_r2")),
        }
    elif isinstance(metrics, dict):
        final_metrics = {
            "val_rmse": _safe_number(metrics.get("val_rmse")),
            "val_mae": _safe_number(metrics.get("val_mae")),
            "val_r2": _safe_number(metrics.get("val_r2")),
        }
    else:
        final_metrics = {"val_rmse": None, "val_mae": None, "val_r2": None}

    best_val_rmse = _best_val_rmse_from_history(history)
    if best_val_rmse is None and isinstance(metrics, dict):
        best_val_rmse = _safe_number(metrics.get("best_val_rmse"))

    payload = {
        "algo": None,
        "model": args.model,
        "seed": args.seed,
        "fold": args.fold,
        "search": {
            "trials": None,
            "best_val_rmse": best_val_rmse,
            "best_params": {
                "lr": args.lr
            }
        },
        "final_training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "final_metrics": final_metrics,
            "history": history,
        }
    }

    out_path = out_dir / f"no_hpo_{args.model}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved run summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
