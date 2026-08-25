"""GFN1-xTB electronic descriptors from SMILES, via a 3D conformer (RDKit ETKDGv3 + MMFF)
and dxtb single-point calculations.

From the neutral state (one SCF): HOMO-LUMO gap, Wiberg bond orders, D3 dispersion energy.
From two extra single-points at charge +/-1 (delta-SCF): ionisation potential, electron
affinity, Parr electrophilicity, and the condensed Fukui maxima. Everything is reduced to
per-molecule scalars and cached to disk (one SMILES -> one row) since the three SCFs per
molecule dominate the pipeline's cost.

Failure policy (no imputation): a molecule whose geometry or *neutral* SCF fails gets an
all-NaN row (dropped downstream). A molecule whose *charged* SCFs fail keeps its neutral
features and NaNs only the charged-derived ones -- anion SCFs in particular often don't
bind, so this avoids discarding otherwise-good molecules for a missing EA.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import dxtb

XTB_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "xtb_features.csv"
SMILES_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "smiles_data.csv"
HARTREE_TO_EV = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897259886

# Physical sanity bounds (eV) on the delta-SCF ion energies. A charged SCF that converges to a
# spurious state produces a wildly out-of-range IP/EA (seen up to ~1800 eV); such a result is a
# failure, not signal, so the charged-derived features for that ion are dropped (NaN, never
# imputed) -- the molecule keeps its neutral features. Bounds are generous: real organic IPs sit
# ~12-17 eV and EAs ~0-8 eV here, and a genuinely unbound electron gives only a mildly negative EA.
IP_BOUNDS = (3.0, 30.0)
EA_BOUNDS = (-10.0, 15.0)

# per-molecule scalar descriptors, in cache/column order
XTB_FEATURE_NAMES = [
    "homo_lumo_gap", "ip", "ea", "electrophilicity",
    "fukui_plus_max", "fukui_minus_max", "wbo_total", "wbo_max", "dispersion_energy",
]

# neutral SCF needs density + overlap cached (for bond orders) and the MO data (for the gap)
_NEUTRAL_OPTS = {"verbosity": 0, "cache_overlap": True, "cache_density": True,
                 "cache_mo_energies": True, "cache_occupation": True, "cache_charges": True}
_CHARGED_OPTS = {"verbosity": 0, "cache_charges": True}


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


def _geometry(smiles):
    """(atomic_numbers, positions_in_bohr) from a 3D-embedded conformer, or None if embedding fails."""
    mol = _embed_3d(smiles)
    if mol is None:
        return None
    numbers = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    positions = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.double) * BOHR_PER_ANGSTROM
    return numbers, positions


def _atom_charges(calc, positions, chrg=None):
    """Per-atom partial charges (reduce the orbital-resolved charges dxtb returns) as a numpy array."""
    kwargs = {} if chrg is None else {"chrg": chrg}
    orbital_q = calc.get_charges(positions, **kwargs)
    return calc.ihelp.reduce_orbital_to_atom(orbital_q).detach().numpy()


def _neutral_features(numbers, positions):
    """Neutral GFN1-xTB single-point -> (feature dict, per-atom charges, total energy in Hartree).
    Raises if the SCF fails (caller turns that into an all-NaN row)."""
    calc = dxtb.calculators.GFN1Calculator(numbers, opts=_NEUTRAL_OPTS, dtype=torch.double)
    energy = calc.get_energy(positions).item()

    mo, occ = calc.get_mo_energies(positions), calc.get_occupation(positions)
    homo = (occ > 0.5).nonzero().max().item()
    gap = (mo[homo + 1] - mo[homo]).item() * HARTREE_TO_EV

    bo = calc.get_bond_orders(positions)
    disp = calc.classicals.get_interaction("DispersionD3")
    disp_energy = disp.get_energy(positions, disp.get_cache(numbers=numbers, ihelp=calc.ihelp)).sum().item()

    feats = {
        "homo_lumo_gap": gap,
        "wbo_total": (bo.sum() / 2).item(),   # symmetric matrix -> sum unique bonds
        "wbo_max": bo.max().item(),
        "dispersion_energy": disp_energy,
    }
    return feats, _atom_charges(calc, positions), energy


def _charged_state(numbers, positions, charge):
    """(total energy in Hartree, per-atom charges) for a +/-1 ion, or None if the SCF fails."""
    try:
        calc = dxtb.calculators.GFN1Calculator(numbers, opts=_CHARGED_OPTS, dtype=torch.double)
        chrg = torch.tensor(float(charge), dtype=torch.double)
        energy = calc.get_energy(positions, chrg=chrg).item()
        return energy, _atom_charges(calc, positions, chrg=chrg)
    except Exception:
        return None


def compute_xtb_features(smiles):
    """All GFN1-xTB descriptors for one SMILES as a dict over XTB_FEATURE_NAMES. All-NaN if the
    geometry or neutral SCF fails; only the charged-derived entries are NaN if just the ions fail.

    Condensed Fukui (finite difference on per-atom populations N = -q + const):
        f+(A) = q_neutral(A) - q_anion(A)   -- site electrophilicity (accepts an electron)
        f-(A) = q_cation(A)  - q_neutral(A) -- site nucleophilicity (donates an electron)
    We keep each molecule's maximum over atoms (its most reactive site)."""
    row = {k: np.nan for k in XTB_FEATURE_NAMES}

    geom = _geometry(smiles)
    if geom is None:
        return row
    numbers, positions = geom
    try:
        feats, q0, e0 = _neutral_features(numbers, positions)
    except Exception:
        return row
    row.update(feats)

    cation = _charged_state(numbers, positions, +1)
    if cation is not None:
        e_cat, q_cat = cation
        ip = (e_cat - e0) * HARTREE_TO_EV
        if IP_BOUNDS[0] <= ip <= IP_BOUNDS[1]:      # else: spurious cation SCF -> leave NaN
            row["ip"] = ip
            row["fukui_minus_max"] = float((q_cat - q0).max())
    anion = _charged_state(numbers, positions, -1)
    if anion is not None:
        e_an, q_an = anion
        ea = (e0 - e_an) * HARTREE_TO_EV
        if EA_BOUNDS[0] <= ea <= EA_BOUNDS[1]:      # else: spurious anion SCF -> leave NaN
            row["ea"] = ea
            row["fukui_plus_max"] = float((q0 - q_an).max())
    if not np.isnan(row["ip"]) and not np.isnan(row["ea"]):
        mu = -(row["ip"] + row["ea"]) / 2      # chemical potential
        eta = row["ip"] - row["ea"]            # chemical hardness
        row["electrophilicity"] = mu * mu / (2 * eta) if eta > 0 else np.nan
    return row


def compute_xtb_features_frame(smiles_list, cache_path=XTB_CACHE_PATH, log=print):
    """Compute (or load from cache) all xTB descriptors for each SMILES, returning a DataFrame with
    XTB_FEATURE_NAMES columns aligned to smiles_list's order. Cache is a SMILES-indexed CSV; only
    SMILES missing from it are (re)computed. Logs per-feature NaN counts for the returned rows."""
    if cache_path.exists():
        cache = pd.read_csv(cache_path, index_col="smiles")
    else:
        cache = pd.DataFrame(columns=XTB_FEATURE_NAMES)
        cache.index.name = "smiles"

    missing = [s for s in dict.fromkeys(smiles_list) if s not in cache.index]
    for i, smiles in enumerate(missing):
        cache.loc[smiles] = pd.Series(compute_xtb_features(smiles))
        if (i + 1) % 25 == 0 or i + 1 == len(missing):
            log(f"xTB features: {i + 1}/{len(missing)} new molecules computed")

    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.rename_axis("smiles").to_csv(cache_path)

    out = cache.loc[smiles_list, XTB_FEATURE_NAMES].reset_index(drop=True)
    failed = out.isna().sum()
    for name in XTB_FEATURE_NAMES:
        if failed[name]:
            log(f"xTB {name}: {int(failed[name])}/{len(smiles_list)} molecules NaN")
    return out


def add_to_tabular(X, smiles_path=SMILES_DATA_PATH, log=print):
    """Append all xTB descriptor columns to X, computed from smiles_data.csv (row-aligned with X).
    Failed molecules carry NaN in the affected columns."""
    smiles = pd.read_csv(smiles_path)["Smiles"].tolist()
    feats = compute_xtb_features_frame(smiles, log=log)
    return pd.concat([X.reset_index(drop=True), feats], axis=1)
