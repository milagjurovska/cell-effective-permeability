from rdkit.Chem import rdchem

DESC_COLS = [
    "mol_weight","logP","num_atoms","num_bonds","num_rotatable_bonds",
    "num_h_donors","num_h_acceptors",
]

ATOM_LIST = [
    "H","B","C","N","O","F","Si","P","S","Cl","As","Se","Br","Te","I","At",
    "Mg","Na","Ca","Fe","Al","Cu","Zn","K","Li","Mn","Co","Ni","V","Ti",
    "Cr","Ag","Au","Cd","Hg","Pb","Sn","Sr","Ba","Bi","Zr",
]
ATOM_TO_IDX = {sym: i for i, sym in enumerate(ATOM_LIST)}

HYB_LIST = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
