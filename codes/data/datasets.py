from typing import Optional
import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from .constants import DESC_COLS
from .graph_utils import mol_to_graph

class Caco2CSVData(InMemoryDataset):
    def __init__(self, csv_path: str, target_col: str = "target", use_graph: bool = True):
        self.csv_path = csv_path
        self.target_col = target_col
        self.use_graph = use_graph
        super().__init__(root=os.path.dirname(csv_path) or ".")
        processed = self.processed_paths[0]
        if os.path.exists(processed):
            self.data, self.slices = torch.load(processed, weights_only=False)
        else:
            self.process()
            self.data, self.slices = torch.load(processed, weights_only=False)

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_path)]

    @property
    def processed_file_names(self):
        suffix = "g" if self.use_graph else "nog"
        name = os.path.splitext(os.path.basename(self.csv_path))[0]
        return [f"{name}_{suffix}.pt"]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        if "smiles" not in df.columns:
            raise ValueError("CSV must contain 'smiles' column")
        if self.target_col not in df.columns:
            raise ValueError(f"CSV must contain target column '{self.target_col}'")
        for c in DESC_COLS:
            if c not in df.columns:
                raise ValueError(f"Missing descriptor column: {c}")

        data_list, bad = [], []
        for idx, row in df.iterrows():
            smiles = str(row["smiles"])
            y = float(row[self.target_col]) if not pd.isna(row[self.target_col]) else float("nan")
            desc = torch.tensor(row[DESC_COLS].values.astype(float), dtype=torch.float)

            if self.use_graph:
                g = mol_to_graph(smiles)
                if g is None:
                    bad.append(idx)
                    continue
                g.y = torch.tensor([y], dtype=torch.float)
                g.desc = desc
                g.smiles = smiles
                data_list.append(g)
            else:
                n = Data(x=torch.zeros((1, 1)),
                         edge_index=torch.empty((2, 0), dtype=torch.long))
                n.y = torch.tensor([y], dtype=torch.float)
                n.desc = desc
                n.smiles = smiles
                data_list.append(n)

        if bad:
            print(f"[WARN] Skipped {len(bad)} invalid SMILES rows")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
