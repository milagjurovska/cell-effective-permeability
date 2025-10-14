from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

from .datasets import Caco2CSVData

@dataclass
class DataConfig:
    csv_path: str
    target_col: str = "target"
    batch_size: int = 64
    num_workers: int = 0
    folds: int = 5
    seed: int = 42
    use_graph: bool = True
    pyg_loader: bool = True

class DataModule:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.ds = Caco2CSVData(cfg.csv_path, target_col=cfg.target_col, use_graph=cfg.use_graph)
        self._folds = None

    def kfold_indices(self):
        if self._folds is not None:
            return self._folds
        n = len(self.ds)
        idx = np.arange(n)
        rng = np.random.RandomState(self.cfg.seed)
        rng.shuffle(idx)
        self._folds = np.array_split(idx, self.cfg.folds)
        return self._folds

    def loaders_for_fold(self, k: int) -> Dict[str, object]:
        folds = self.kfold_indices()
        val_idx = folds[k]
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != k])

        train_ds = Subset(self.ds, train_idx)
        val_ds   = Subset(self.ds, val_idx)

        Loader = PyGDataLoader if self.cfg.pyg_loader else DataLoader
        return {
            "train": Loader(train_ds, batch_size=self.cfg.batch_size, shuffle=True,  num_workers=self.cfg.num_workers),
            "val":   Loader(val_ds,   batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers),
        }
