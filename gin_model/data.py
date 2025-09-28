from typing import List, Optional
import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem
from rdkit.Chem import rdchem

DESC_COLS = [
    'mol_weight','logP','num_atoms','num_bonds','num_rotatable_bonds','num_h_donors','num_h_acceptors'
]

ATOM_LIST = [
    'H','B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','Mg','Na','Ca','Fe','Al','Cu','Zn','K','Li','Mn','Co','Ni','V','Ti','Cr','Ag','Au','Cd','Hg','Pb','Sn','Sr','Ba','Bi','Zr'
]
ATOM_TO_IDX = {sym: i for i, sym in enumerate(ATOM_LIST)}
HYB_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]


def atom_features(atom: rdchem.Atom) -> List[float]:
    feat = []
    atom_type = [0] * (len(ATOM_LIST) + 1)
    atom_type[ATOM_TO_IDX.get(atom.GetSymbol(), len(ATOM_LIST))] = 1
    feat.extend(atom_type)

    deg = [0] * 6
    d = int(atom.GetDegree())
    deg[min(d, 5)] = 1
    feat.extend(deg)

    feat.append(float(atom.GetFormalCharge()))

    hyb = [0] * (len(HYB_LIST) + 1)
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

    x = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    edges_src, edges_dst = [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edges_src += [a, b]
        edges_dst += [b, a]

    if len(edges_src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
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

        data_list, bad = [], []
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
                n = Data(x=torch.zeros((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))
                n.y = torch.tensor([y], dtype=torch.float)
                n.desc = desc
                n.smiles = smiles
                data_list.append(n)

        if bad:
            print(f"[WARN] Skipped {len(bad)} invalid SMILES rows")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
