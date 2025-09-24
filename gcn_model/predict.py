from __future__ import annotations
from dataclasses import asdict
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from gcn_model.configs.base_config import BaseConfig
from gcn_model.configs.model_config import ModelConfig, GraphStackConfig, PredictorConfig
from gcn_model.model import Model
from gcn_model.data import DESC_COLS, mol_to_graph
from gcn_model.models.graph_stack import GraphStack
from gcn_model.models.predictor import Predictor


def seed_everything(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y = batch.y.view(-1)
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            out = model(batch)[mask]
            ys.append(y[mask].cpu().numpy())
            preds.append(out.cpu().numpy())
    if not ys:
        return float('nan'), float('nan'), float('nan')
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(np.nan_to_num(1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()))
    return mae, rmse, r2


def build_model_from_sample(sample, hidden_dim: int = 128, dropout: float = 0.2) -> tuple[Model, ModelConfig]:
    in_dim = sample.x.shape[1]
    g_cfg = GraphStackConfig(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    p_cfg = PredictorConfig(pred_input_channels=hidden_dim, desc_dim=len(DESC_COLS), hidden_dim=hidden_dim, dropout=dropout)
    m_cfg = ModelConfig(model_head=g_cfg, predictor=p_cfg, freeze_head=False)

    model_head = GraphStack(in_dim=g_cfg.in_dim, hidden_dim=g_cfg.hidden_dim, dropout=g_cfg.dropout)
    predictor = Predictor.from_config(p_cfg)
    return Model(model_head=model_head, prediction_head=predictor), m_cfg


def train(
    csv_path: str,
    target_col: str = 'target',
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_split: float = 0.2,
    device: Optional[str] = None,
    save_dir: str = 'runs',
    experiment_name: str = 'gcn_with_desc',
    registry_root: str = 'best_gcn_models',
    registry_tag: Optional[str] = None,
    score_name: str = 'rmse',
) -> str:

    seed_everything()
    ds = Caco2CSVData(csv_path, target_col=target_col, use_graph=True)

    n = len(ds)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    model, model_cfg = build_model_from_sample(ds[0])
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    exp_dir = BaseConfig.get_model_folder(experiment_name=experiment_name, model_root=save_dir)
    os.makedirs(exp_dir, exist_ok=True)

    best_rmse = float('inf')
    best_metrics = {"mae": float('inf'), "rmse": float('inf'), "r2": float('-inf')}
    patience = 20

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.view(-1)
            mask = ~torch.isnan(y)
            if mask.sum() == 0:
                continue
            preds = model(batch)
            loss = nn.functional.mse_loss(preds[mask], y[mask])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_losses.append(loss.item())

        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={np.mean(train_losses):.4f} | val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} val_r2={val_r2:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_metrics = {"mae": val_mae, "rmse": val_rmse, "r2": val_r2}
            # save weights + config.json
            model.save(exp_dir)
            with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
                json.dump(asdict(model_cfg), f, indent=2)
            patience = 20
        else:
            patience -= 1
            if patience <= 0:
                print('Early stopping')
                break

    print(f"Best model saved to {exp_dir}")

    tag = registry_tag or experiment_name
    score_val = best_metrics.get(score_name, best_rmse)
    registry_path = BaseConfig.promote_run_to_registry(
        src_run_dir=exp_dir,
        tag=tag,
        score_name=score_name,
        score_value=float(score_val),
        registry_root=registry_root,
    )
    print(f"Promoted best checkpoint to: {registry_path}")

    with open(os.path.join(exp_dir, 'best_metrics.json'), 'w') as f:
        json.dump(best_metrics, f, indent=2)

    return exp_dir


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--target', default='target')
    ap.add_argument('--experiment', default='gcn_with_desc')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--runs_root', default='runs')
    ap.add_argument('--registry_root', default='best_gcn_models')
    ap.add_argument('--registry_tag', default=None)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    args = ap.parse_args()

    train(
        csv_path=args.csv,
        target_col=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.runs_root,
        experiment_name=args.experiment,
        registry_root=args.registry_root,
        registry_tag=args.registry_tag,
    )
