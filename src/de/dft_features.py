"""HOMO-LUMO gap feature: SMILES -> 3D conformer -> GFN1-xTB
single-point calculation (dxtb) -> gap between highest occupied and lowest unoccupied
molecular orbitals, in eV. Results are cached to disk since each calculation is
expensive relative to the rest of the pipeline.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import dxtb

CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "homo_lumo_gap.csv"
SMILES_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "smiles_data.csv"
HARTREE_TO_EV = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897259886
DXTB_OPTS = {"cache_mo_energies": True, "cache_occupation": True, "verbosity": 0}


def _embed_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 1
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def homo_lumo_gap(smiles):
    """GFN1-xTB HOMO-LUMO gap in eV for one SMILES string. NaN if 3D embedding or
    the xTB single-point calculation fails."""
    mol = _embed_3d(smiles)
    if mol is None:
        return np.nan

    numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    conf = mol.GetConformer()
    positions = torch.tensor(conf.GetPositions(), dtype=torch.double) * BOHR_PER_ANGSTROM

    try:
        calc = dxtb.calculators.GFN1Calculator(numbers, opts=DXTB_OPTS, dtype=torch.double)
        calc.calculate(["mo_energies", "occupation"], positions)
        mo_energies, occupation = calc.cache.mo_energies, calc.cache.occupation
        homo_idx = (occupation > 0.5).nonzero().max().item()
        gap_hartree = (mo_energies[homo_idx + 1] - mo_energies[homo_idx]).item()
    except Exception:
        return np.nan

    return gap_hartree * HARTREE_TO_EV


def compute_homo_lumo_gaps(smiles_list, cache_path=CACHE_PATH, log=print):
    """Compute (or load from cache) the HOMO-LUMO gap for each SMILES in smiles_list,
    returning a Series aligned to smiles_list's order. Cache is a SMILES -> gap CSV;
    only SMILES missing from it are (re)computed."""
    cache = (pd.read_csv(cache_path, index_col="smiles")["homo_lumo_gap_ev"] if cache_path.exists()
             else pd.Series(dtype=float, name="homo_lumo_gap_ev"))

    missing = [s for s in smiles_list if s not in cache.index]
    for i, smiles in enumerate(missing):
        cache[smiles] = homo_lumo_gap(smiles)
        if (i + 1) % 25 == 0 or i + 1 == len(missing):
            log(f"HOMO-LUMO gap: {i + 1}/{len(missing)} new molecules computed")

    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.rename_axis("smiles").to_csv(cache_path)

    n_failed = cache.loc[smiles_list].isna().sum()
    if n_failed:
        log(f"HOMO-LUMO gap: {n_failed}/{len(smiles_list)} molecules failed (NaN)")

    return cache.loc[smiles_list].reset_index(drop=True)


def add_to_tabular(X, smiles_path=SMILES_DATA_PATH, log=print):
    """Append a homo_lumo_gap column to X, computed from smiles_data.csv (same row order as
    X, verified elsewhere to be row-aligned with biodeg.csv). Failed molecules keep a NaN gap
    to drop."""
    smiles = pd.read_csv(smiles_path)["Smiles"].tolist()
    return X.assign(homo_lumo_gap=compute_homo_lumo_gaps(smiles, log=log).to_numpy())
