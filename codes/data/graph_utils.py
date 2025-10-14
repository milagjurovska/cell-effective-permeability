from typing import List, Optional
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem

from .constants import ATOM_LIST, ATOM_TO_IDX, HYB_LIST

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

    edge_index = (
        torch.empty((2, 0), dtype=torch.long)
        if len(edges_src) == 0
        else torch.tensor([edges_src, edges_dst], dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index)
