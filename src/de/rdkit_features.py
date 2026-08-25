"""2D RDKit descriptors from SMILES: Crippen octanol-water logP, molecular weight,
topological polar surface area. Fast enough to skip caching. NaN for an unparseable SMILES."""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

RDKIT_FEATURES = {"MolLogP": Crippen.MolLogP, "MolWt": Descriptors.MolWt, "TPSA": Descriptors.TPSA}


def compute_rdkit_features(smiles_list):
    """(n, len(RDKIT_FEATURES)) DataFrame aligned to smiles_list's order; NaN row on parse fail."""
    def row(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return [np.nan] * len(RDKIT_FEATURES) if mol is None else [fn(mol) for fn in RDKIT_FEATURES.values()]

    return pd.DataFrame([row(s) for s in smiles_list], columns=list(RDKIT_FEATURES))


def add_to_tabular(X, smiles_path, log=print):
    """Append the RDKit descriptor columns to X, computed from smiles_path (row-aligned with X)."""
    smiles = pd.read_csv(smiles_path)["Smiles"].tolist()
    feats = compute_rdkit_features(smiles)
    n_failed = int(feats.isna().any(axis=1).sum())
    if n_failed:
        log(f"RDKit descriptors: {n_failed}/{len(smiles)} molecules failed (NaN)")
    return pd.concat([X.reset_index(drop=True), feats], axis=1)
