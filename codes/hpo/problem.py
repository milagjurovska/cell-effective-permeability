import numpy as np
from niapy.problems import Problem

from codes.training.trainer import Trainer
from codes.models import model_factory
from codes.config import ModelConfig, GraphStackConfig, PredictorConfig
from codes.data.datamodule import DataModule, DataConfig
from codes.data.constants import DESC_COLS
from .space import bounds, decode_vector

class HyperParamProblem(Problem):
    def __init__(self, model_name, dm, in_dim, desc_dim, epochs, patience, fold, seed):
        self.model_name, self.dm = model_name, dm
        self.in_dim, self.desc_dim = in_dim, desc_dim
        self.epochs, self.patience, self.fold = epochs, patience, fold
        self.seed = seed
        lb, ub = bounds(model_name)
        super().__init__(dimension=len(lb), lower=lb, upper=ub)

    def _evaluate(self, x):
        hp = decode_vector(np.array(x, float), self.model_name)

        if self.model_name == "gat":
            mh = GraphStackConfig(in_dim=self.in_dim, hidden_dim=hp["enc_hidden"],
                                  heads=hp["heads"], dropout=hp["enc_drop"], attn_dropout=0.0)
        else:
            mh = GraphStackConfig(in_dim=self.in_dim, hidden_dim=hp["enc_hidden"], dropout=hp["enc_drop"])
        pcfg = PredictorConfig(pred_input_channels=0, desc_dim=self.desc_dim,
                               hidden_dim=hp["pred_hidden"], dropout=hp["pred_drop"])
        mcfg = ModelConfig(name=self.model_name, model_head=mh, predictor=pcfg)

        ModelCls = model_factory(self.model_name)
        model = ModelCls.load(mcfg)
        if hasattr(model, "model_head") and hasattr(model.model_head, "graph_output_channels"):
            mcfg.predictor.pred_input_channels = int(model.model_head.graph_output_channels)

        loaders = self.dm.loaders_for_fold(self.fold)
        trainer = Trainer(lr=hp["lr"], epochs=self.epochs, patience=self.patience,
                          run_dir=None, verbose=False)
        metrics = trainer.fit(model, loaders, save_best=False, cfg_for_logging=None, run_logger=None)
        return float(metrics["val_rmse"])

def build_problem(model: str, csv: str, target: str, batch_size: int,
                  epochs: int, patience: int, fold: int, seed: int):
        dm = DataModule(DataConfig(csv_path=csv, target_col=target, batch_size=batch_size, seed=seed))
        sample = dm.ds[0]
        in_dim = int(sample.x.size(-1))
        desc_dim = int(len(DESC_COLS)) if hasattr(sample, "desc") else int(sample.desc.numel())
        return HyperParamProblem(model, dm, in_dim, desc_dim, epochs, patience, fold, seed)
