import os, json
from codes.training.trainer import Trainer
from codes.models import model_factory
from codes.config import ModelConfig, GraphStackConfig, PredictorConfig
from codes.data.datamodule import DataModule, DataConfig
from codes.data.constants import DESC_COLS

def train_with_best_to_summary(model_name: str, best: dict, epochs: int, patience: int,
                               csv: str, target: str, batch_size: int, seed: int,
                               out_dir: str, algo_name: str, fold: int,
                               best_val_rmse_from_search: float, trials:int) -> str:

    dm = DataModule(DataConfig(csv_path=csv, target_col=target, batch_size=batch_size, seed=seed))
    sample = dm.ds[0]
    in_dim  = int(sample.x.size(-1))
    desc_dim = int(len(DESC_COLS)) if hasattr(sample, "desc") else int(sample.desc.numel())

    if model_name == "gat":
        mh = GraphStackConfig(in_dim=in_dim, hidden_dim=best["enc_hidden"],
                              heads=best["heads"], dropout=best["enc_drop"], attn_dropout=0.0)
    else:
        mh = GraphStackConfig(in_dim=in_dim, hidden_dim=best["enc_hidden"], dropout=best["enc_drop"])
    pcfg = PredictorConfig(pred_input_channels=0, desc_dim=desc_dim,
                           hidden_dim=best["pred_hidden"], dropout=best["pred_drop"])
    mcfg = ModelConfig(name=model_name, model_head=mh, predictor=pcfg)

    ModelCls = model_factory(model_name)
    model = ModelCls.load(mcfg)
    if hasattr(model, "model_head") and hasattr(model.model_head, "graph_output_channels"):
        mcfg.predictor.pred_input_channels = int(model.model_head.graph_output_channels)

    loaders = dm.loaders_for_fold(fold)
    trainer = Trainer(lr=best["lr"], epochs=epochs, patience=patience, run_dir=None, verbose=True)
    final = trainer.fit(model, loaders, save_best=False, cfg_for_logging=None, run_logger=None)

    os.makedirs(os.path.join(out_dir, model_name), exist_ok=True)
    out_path = os.path.join(out_dir, model_name, f"hpo_{model_name}_{algo_name}.json")
    payload = {
        "algo": algo_name,
        "model": model_name,
        "seed":  seed,
        "fold":  fold,
        "search": {
            "trials": None,
            "best_val_rmse": float(best_val_rmse_from_search),
            "best_params": {
                "enc_hidden": int(best["enc_hidden"]),
                **({"heads": int(best["heads"])} if model_name == "gat" else {}),
                "enc_drop": float(best["enc_drop"]),
                "pred_hidden": int(best["pred_hidden"]),
                "pred_drop": float(best["pred_drop"]),
                "lr": float(best["lr"]),
            },
        },
        "final_training": {
            "epochs": epochs,
            "patience": patience,
            "final_metrics": {
                "val_rmse": float(final["val_rmse"]),
                "val_mae":  float(final["val_mae"]),
                "val_r2":   float(final["val_r2"]),
            },
            "history": final.get("history", []),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(out_path)
