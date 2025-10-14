import os
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from tdc.single_pred import ADME

def canonicalize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol, canonical=True)
    return can, mol

def compute_rdkit_descriptors(mol):
    return {
        "mol_weight": Descriptors.MolWt(mol),
        "logP": Descriptors.MolLogP(mol),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_h_donors": Lipinski.NumHDonors(mol),
        "num_h_acceptors": Lipinski.NumHAcceptors(mol),
    }

def try_deepchem_descriptors(smiles_list):
    try:
        import deepchem as dc
    except Exception:
        return None

    try:
        featurizer = dc.feat.RDKitDescriptors()
    except Exception:
        return None

    try:
        X = featurizer.featurize(smiles_list)
    except Exception:
        return None

    names = None
    for attr in ("feature_names", "features", "descriptor_names", "desc_list"):
        if hasattr(featurizer, attr):
            names = getattr(featurizer, attr)
            break
    if names is None and hasattr(featurizer, "_desc_list"):
        names = getattr(featurizer, "_desc_list")

    if names is None:
        return None

    import numpy as np
    import pandas as pd
    names = list(names)
    if X is None or len(X) == 0 or (hasattr(X, "shape") and X.shape[1] != len(names)):
        return None

    df_desc = pd.DataFrame(np.asarray(X), columns=names)
    return df_desc

def main(output_csv: str):
    data = ADME(name="Caco2_Wang")
    df = data.get_data().copy()
    df = df.rename(columns={"Drug": "smiles", "Y": "target"})
    df = df.dropna(subset=["smiles", "target"]).reset_index(drop=True)

    canon, mols = zip(*(canonicalize(s) for s in df["smiles"]))
    df["smiles"] = canon
    df["mol"] = mols
    df = df.dropna(subset=["smiles", "mol"]).reset_index(drop=True)

    dc_df = try_deepchem_descriptors(df["smiles"].tolist())

    rows = []
    for mol, smi, y in zip(df["mol"], df["smiles"], df["target"]):
        d = compute_rdkit_descriptors(mol)
        d["smiles"] = smi
        d["target"] = float(y)
        rows.append(d)
    out = pd.DataFrame(rows, columns=[
        "smiles", "target",
        "mol_weight", "logP",
        "num_atoms", "num_bonds",
        "num_rotatable_bonds", "num_h_donors", "num_h_acceptors"
    ])

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"✅ Saved processed dataset to {output_csv}")
    print(f"N = {len(out)} rows")
    print("Columns:", list(out.columns))
    print(out.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Caco-2 (Wang) CSV")
    parser.add_argument("-o", "--output", default="data/caco2_processed.csv",
                        help="Path to output CSV (default: data/caco2_processed.csv)")
    args = parser.parse_args()
    main(args.output)
