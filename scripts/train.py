import argparse
from torch_geometric.loader import DataLoader

from codes.training.trainer import Trainer
from codes.models import model_factory
from codes.config import ModelConfig, GraphStackConfig, PredictorConfig
from codes.data.datamodule import DataModule, DataConfig
from codes.data.constants import DESC_COLS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gat","gcn","gin"], default="gat")
    ap.add_argument("--csv", default="data/caco2_processed.csv")
    ap.add_argument("--target", default="target")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--run_dir", default="runs/exp1")
    args = ap.parse_args()

    dm = DataModule(DataConfig(csv_path=args.csv, target_col=args.target, batch_size=args.batch_size))
    loaders = dm.loaders_for_fold(0)

    sample = dm.ds[0]
    in_dim = sample.x.size(-1)
    desc_dim = len(DESC_COLS) if hasattr(sample, "desc") else 0

    mcfg = ModelConfig(
        name=args.model,
        model_head=GraphStackConfig(in_dim=in_dim),
        predictor=PredictorConfig(desc_dim=desc_dim),
    )

    ModelCls = model_factory(args.model)
    model = ModelCls.load(mcfg)
    trainer = Trainer(lr=args.lr, epochs=args.epochs, patience=args.patience, run_dir=args.run_dir)
    metrics = trainer.fit(model, loaders, cfg_for_logging=mcfg)
    print(metrics)

if __name__ == "__main__":
    main()
