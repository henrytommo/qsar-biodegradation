"""
tabular QSAR descriptors or SMILES-derived PyG graphs, 
with the GFN1-xTB HOMO-LUMO gap included either way. Dataset
splits / scales / subsets same for both shapes.

keep torch/rdkit AFTER `import xgboost`
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data

from de.data import BINARY_FEATS, FEATURE_DICT, FEATURE_NAMES, load_data
from de.dft_features import add_to_tabular, compute_homo_lumo_gaps

SCALERS = {"normalize": MinMaxScaler, "standardize": StandardScaler}
SMILES_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "smiles_data.csv"

# smiles_data.csv uses the original bracket notation (e.g. "F01[N-N]")
_RENAME_MAP = {k: k.replace("[", "(").replace("]", ")") for k in FEATURE_DICT if k != "experimental class"}
GLOBAL_FEATURE_NAMES = FEATURE_NAMES[:-1]
ALL_GLOBAL_FEATURE_NAMES = GLOBAL_FEATURE_NAMES + ["homo_lumo_gap"]
_SCALE_IDX = [i for i, name in enumerate(ALL_GLOBAL_FEATURE_NAMES) if name not in BINARY_FEATS]


# --- graph construction ---------------------------------------------------------

def smiles_to_graph(smiles, global_feats=None, y=None):
    """Convert one SMILES string into a PyG Data object. None for unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [
        [atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(), int(atom.GetIsAromatic())]
        for atom in mol.GetAtoms()
    ]
    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    edge_index = (torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                  if edge_indices else torch.empty((2, 0), dtype=torch.long))

    graph = Data(x=x, edge_index=edge_index)
    if global_feats is not None:
        graph.global_feats = torch.tensor(global_feats, dtype=torch.float).unsqueeze(0)
    if y is not None:
        graph.y = torch.tensor([y], dtype=torch.float)
    return graph


def _load_graphs(log=print):
    """One PyG graph per molecule from smiles_data.csv: 41 QSAR descriptors + the HOMO-LUMO
    gap as global features, Class (RB=1/NRB=0) as label. Molecules with an unparseable SMILES
    or any missing global feature (e.g. xTB failed for the gap) are dropped."""
    df = pd.read_csv(SMILES_DATA_PATH).rename(columns=_RENAME_MAP)
    y = np.where(df["Class"] == "RB", 1, 0)
    gaps = compute_homo_lumo_gaps(df["Smiles"].tolist(), log=log)

    graphs = []
    n_skipped = 0
    for i, row in df.iterrows():
        global_feats = np.append(row[GLOBAL_FEATURE_NAMES].to_numpy(dtype=float), gaps.iloc[i])
        graph = None if np.isnan(global_feats).any() else smiles_to_graph(
            row["Smiles"], global_feats=global_feats, y=y[i])
        if graph is None:
            n_skipped += 1
            continue
        graph.status = row["Status"]
        graphs.append(graph)
    return graphs, n_skipped


def _global_feats_matrix(graphs):
    return torch.cat([g.global_feats for g in graphs], dim=0).numpy()


def _with_global_feats(graphs, matrix):
    scaled = [g.clone() for g in graphs]
    for g, row in zip(scaled, matrix):
        g.global_feats = torch.tensor(row, dtype=torch.float).unsqueeze(0)
    return scaled


# --- the dataset ----------------------------------------------------------------

@dataclass
class Dataset:
    data: object            # pd.DataFrame (tabular) or list[Data] (graph)
    y: object = None        # labels array (tabular); None for graphs (label rides on g.y)
    binary_feats: list = None   # tabular: names left unscaled
    scale_idx: list = None      # graph: indices into global_feats that get scaled
    feature_names: list = None  # tabular column order == graph global_feats order
    n_skipped: int = 0          # graphs dropped for unparseable SMILES (load only)

    @classmethod
    def load(cls, graph=False, log=print):
        """Load the modelling dataset (HOMO-LUMO gap included either way). graph=False ->
        tabular DataFrame; graph=True -> PyG graphs with the same features as global_feats."""
        if graph:
            graphs, n_skipped = _load_graphs(log=log)
            return cls(data=graphs, feature_names=list(ALL_GLOBAL_FEATURE_NAMES),
                       scale_idx=list(_SCALE_IDX), n_skipped=n_skipped)
        X, y = load_data()
        X = add_to_tabular(X, log=log)
        keep = X.notna().all(axis=1).to_numpy()   # drop any row with a missing feature (no imputation)
        if not keep.all():
            log(f"Dropped {int((~keep).sum())} rows with missing feature values")
            X, y = X[keep].reset_index(drop=True), y[keep]
        binary = [f for f in BINARY_FEATS if f in X.columns]
        return cls(data=X, y=y, binary_feats=binary, feature_names=list(X.columns))

    @property
    def is_graph(self):
        return not isinstance(self.data, pd.DataFrame)

    @property
    def dims(self):
        """What a model needs to size its layers; models read only the keys they use."""
        if self.is_graph:
            g = self.data[0]
            return {"node_feat_dim": g.x.shape[1], "global_feat_dim": g.global_feats.shape[1]}
        return {"n_features": self.data.shape[1]}

    def feature_frame(self):
        """A tabular (features_df, y) view for ranking -- so feature selection works for graphs
        too, ranking on their own global_feats (aligned, no cross-representation mismatch)."""
        if self.is_graph:
            matrix = _global_feats_matrix(self.data)
            y = np.array([int(g.y.item()) for g in self.data])
            return pd.DataFrame(matrix, columns=self.feature_names), y
        return self.data, self.y

    def _like(self, data, y=None):
        """A new Dataset over `data` sharing this one's feature metadata (same feature set)."""
        return Dataset(data=data, y=y, binary_feats=self.binary_feats,
                       scale_idx=self.scale_idx, feature_names=self.feature_names)

    # --- splitting & scaling ---
    def holdout(self, test_size=0.2, random_state=1):
        """Stratified train/test split (unscaled); scaling happens later, per fold / final fit."""
        if self.is_graph:
            labels = [int(g.y.item()) for g in self.data]
            train, test = train_test_split(self.data, test_size=test_size, stratify=labels,
                                           random_state=random_state)
            return self._like(train), self._like(test)
        X_tr, X_te, y_tr, y_te = train_test_split(self.data, self.y, test_size=test_size,
                                                  stratify=self.y, random_state=random_state)
        return self._like(X_tr, y_tr), self._like(X_te, y_te)

    def transform(self, scaler):
        """Apply an already-fit scaler to self's continuous features (binary feats untouched)."""
        if self.is_graph:
            matrix = _global_feats_matrix(self.data)
            matrix[:, self.scale_idx] = scaler.transform(matrix[:, self.scale_idx])
            return self._like(_with_global_feats(self.data, matrix))
        cols = [c for c in self.data.columns if c not in (self.binary_feats or [])]
        out = self.data.copy()
        out[cols] = scaler.transform(self.data[cols])
        return self._like(out, self.y)

    def scale(self, other, method="standardize"):
        """Fit a scaler on self, apply to self and `other`. Returns (self_s, other_s, scaler)."""
        scaler = SCALERS[method]()
        if self.is_graph:
            matrix = _global_feats_matrix(self.data)
            matrix[:, self.scale_idx] = scaler.fit_transform(matrix[:, self.scale_idx])
            return self._like(_with_global_feats(self.data, matrix)), other.transform(scaler), scaler
        cols = [c for c in self.data.columns if c not in (self.binary_feats or [])]
        tr = self.data.copy()
        tr[cols] = scaler.fit_transform(self.data[cols])
        return self._like(tr, self.y), other.transform(scaler), scaler

    def cv_folds(self, method="standardize", n_splits=5, random_state=1):
        """Stratified CV folds with per-fold scaling (fit on the train fold only, no leakage);
        yields (train_ds, val_ds), both scaled."""
        labels = np.array([int(g.y.item()) for g in self.data]) if self.is_graph else self.y
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(self.data, labels):
            if self.is_graph:
                train = self._like([self.data[i] for i in train_idx])
                val = self._like([self.data[i] for i in val_idx])
            else:
                train = self._like(self.data.iloc[train_idx], self.y[train_idx])
                val = self._like(self.data.iloc[val_idx], self.y[val_idx])
            train_s, val_s, _ = train.scale(val, method=method)
            yield train_s, val_s

    # --- feature selection ---
    def subset(self, names):
        """A new Dataset keeping only `names` (in the given order). Tabular: select columns.
        Graph: index the global_feats and recompute which of them stay scaled."""
        names = list(names)
        if self.is_graph:
            keep = [self.feature_names.index(n) for n in names]
            sub = [g.clone() for g in self.data]
            for g in sub:
                g.global_feats = g.global_feats[:, keep]
            scale_idx = [i for i, n in enumerate(names) if n not in BINARY_FEATS]
            return Dataset(data=sub, feature_names=names, scale_idx=scale_idx)
        binary = [f for f in (self.binary_feats or []) if f in names]
        return Dataset(data=self.data[names], y=self.y, binary_feats=binary, feature_names=names)
