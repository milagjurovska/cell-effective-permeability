from typing import List, Optional
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gin_model.model import Model
from gin_model.data import DESC_COLS, mol_to_graph


def predict(csv_path: str, model_dir: str, out_csv: str = 'preds.csv', batch_size: int = 256, device: Optional[str] = None) -> pd.DataFrame:
    model = Model.by_name(model_dir)

    df = pd.read_csv(csv_path)
    for c in ['smiles'] + DESC_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    in_dim = model.model_head.conv1.nn[0].in_features if hasattr(model.model_head, 'conv1') else 1

    samples: List[Data] = []
    for _, row in df.iterrows():
        smiles = str(row['smiles'])
        desc = torch.tensor(row[DESC_COLS].values.astype(float), dtype=torch.float)
        g = mol_to_graph(smiles)
        if g is None:
            g = Data(x=torch.zeros((1, in_dim)), edge_index=torch.empty((2, 0), dtype=torch.long))
        g.desc = desc
        g.smiles = smiles
        samples.append(g)

    loader = DataLoader(samples, batch_size=batch_size)
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).cpu().numpy().tolist()
            preds.extend(out)

    df_out = df.copy()
    df_out['prediction'] = preds
    df_out.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")
    return df_out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--out', default='preds.csv')
    args = ap.parse_args()

    predict(csv_path=args.csv, model_dir=args.model_dir, out_csv=args.out)
