from typing import List, Optional, Tuple
import os

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit import Chem
from rdkit.Chem import rdchem

DESC_COLS = ['mol_weight','logP','num_atoms','num_bonds','num_rotatable_bonds','num_h_donors','num_h_acceptors']

ATOM_LIST = [
    'H','B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','Mg','Na','Ca','Fe','Al','Cu','Zn','K','Li','Mn','Co','Ni','V','Ti','Cr','Ag','Au','Cd','Hg','Pb','Sn','Sr','Ba','Bi','Zr'
]
ATOM_TO_IDX = {sym:i for i,sym in enumerate(ATOM_LIST)}
HYB_LIST = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3,
            rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2]


def atom_features(atom: rdchem.Atom) -> List[float]:
    feat = []
    atom_type = [0]* (len(ATOM_LIST)+1)
    atom_type[ATOM_TO_IDX.get(atom.GetSymbol(), len(ATOM_LIST))] = 1
    feat.extend(atom_type)

    deg = [0]*6
    d = int(atom.GetDegree())
    deg[min(d,5)] = 1
    feat.extend(deg)

    feat.append(float(atom.GetFormalCharge()))

    hyb = [0]*(len(HYB_LIST)+1)
    h = atom.GetHybridization()
    hyb[HYB_LIST.index(h) if h in HYB_LIST else len(HYB_LIST)] = 1
    feat.extend(hyb)

    feat.append(1.0 if atom.GetIsAromatic() else 0.0)
    feat.append(float(atom.GetTotalNumHs()))
    feat.append(1.0 if atom.IsInRing() else 0.0)
    return feat


def mol_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    edges_src = []
    edges_dst = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edges_src += [a, b]
        edges_dst += [b, a]

    if len(edges_src) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

class Caco2CSVData(InMemoryDataset):
    def __init__(self, csv_path: str, target_col: str = 'target', use_graph: bool = True):
        self.csv_path = csv_path
        self.target_col = target_col
        self.use_graph = use_graph
        super().__init__(root=os.path.dirname(csv_path) or '.')
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
        suffix = 'g' if self.use_graph else 'nog'
        name = os.path.splitext(os.path.basename(self.csv_path))[0]
        return [f'{name}_{suffix}.pt']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_path)
        if 'smiles' not in df.columns:
            raise ValueError("CSV must contain 'smiles' column")
        if self.target_col not in df.columns:
            raise ValueError(f"CSV must contain target column '{self.target_col}'")
        for c in DESC_COLS:
            if c not in df.columns:
                raise ValueError(f"Missing descriptor column: {c}")

        data_list = []
        bad = []
        for idx, row in df.iterrows():
            smiles = str(row['smiles'])
            y = float(row[self.target_col]) if not pd.isna(row[self.target_col]) else float('nan')
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
                n = Data(x=torch.zeros((1,1)), edge_index=torch.empty((2,0), dtype=torch.long))
                n.y = torch.tensor([y], dtype=torch.float)
                n.desc = desc
                n.smiles = smiles
                data_list.append(n)

        if bad:
            print(f"[WARN] Skipped {len(bad)} invalid SMILES rows")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, desc_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + desc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)

        g = global_mean_pool(x, batch)
        desc = data.desc
        batch_size = g.size(0)
        desc_dim = desc.size(0) // batch_size

        desc = desc.view(batch_size, desc_dim)

        out = torch.cat([g, desc], dim=-1)
        return self.head(out).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, desc_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(desc_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, data: Data):
        return self.net(data.desc).squeeze(-1)


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(csv_path: str, target_col: str = 'target', use_graph: bool = True,
          epochs: int = 100, batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-5,
          val_split: float = 0.2, device: Optional[str] = None, save_dir: str = 'runs') -> str:
    seed_everything()
    ds = Caco2CSVData(csv_path, target_col=target_col, use_graph=use_graph)
    n = len(ds)
    n_val = int(n * val_split)
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    in_dim = ds[0].x.shape[1] if use_graph else 0
    desc_dim = len(DESC_COLS)

    hidden_dim = 128  # or 64 if you want smaller

    model = (
        GCNRegressor(in_dim=in_dim, hidden_dim=hidden_dim, desc_dim=desc_dim)
        if use_graph
        else MLPRegressor(desc_dim=desc_dim)
    )

    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_gcn.pt')
    best_rmse = float('inf')
    patience = 20

    for epoch in range(1, epochs+1):
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
            torch.save(
                {'model_state': model.state_dict(), 'use_graph': use_graph, 'in_dim': in_dim, 'desc_dim': desc_dim,
                 'hidden_dim': hidden_dim}, best_path)
            patience = 20
        else:
            patience -= 1
            if patience <= 0:
                print('Early stopping')
                break

    print(f"Best model saved to {best_path}")
    return best_path


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float,float,float]:
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
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    r2 = float(np.nan_to_num(1 - ((y_true - y_pred)**2).sum() / ((y_true - y_true.mean())**2).sum()))
    return mae, rmse, r2



def predict(csv_path: str, checkpoint: str, out_csv: str = 'preds.csv', batch_size: int = 256,
            device: Optional[str] = None):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    use_graph = ckpt.get('use_graph', True)
    in_dim = ckpt.get('in_dim', None)
    desc_dim = ckpt.get('desc_dim', len(DESC_COLS))
    hidden_dim = ckpt.get('hidden_dim', 128)

    df = pd.read_csv(csv_path)
    for c in ['smiles'] + DESC_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    samples = []
    for _, row in df.iterrows():
        smiles = str(row['smiles'])
        desc = torch.tensor(row[DESC_COLS].values.astype(float), dtype=torch.float)
        if use_graph:
            g = mol_to_graph(smiles)
            if g is None:
                g = Data(x=torch.zeros((1, in_dim if in_dim else 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
            g.desc = desc
            g.smiles = smiles
            samples.append(g)
        else:
            n = Data(x=torch.zeros((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
            n.desc = desc
            n.smiles = smiles
            samples.append(n)

    loader = DataLoader(samples, batch_size=batch_size)
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = GCNRegressor(in_dim=in_dim if in_dim else 1, hidden_dim=hidden_dim,
                         desc_dim=desc_dim) if use_graph else MLPRegressor(desc_dim=desc_dim)

    model.load_state_dict(ckpt['model_state'])
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