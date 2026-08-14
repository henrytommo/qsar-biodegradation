"""SMILES -> PyTorch Geometric graph conversion, with tabular
QSAR descriptors (same 41 columns as de.data) attached to each graph as global features.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data

from de.data import BINARY_FEATS, FEATURE_DICT, FEATURE_NAMES
from de.preprocessing import SCALERS

SMILES_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "smiles_data.csv"

# smiles_data.csv uses the original bracket notation (e.g. "F01[N-N]"); align to the
# notation used everywhere else in de.data / de.cv / de.preprocessing.
_RENAME_MAP = {k: k.replace("[", "(").replace("]", ")") for k in FEATURE_DICT if k != "experimental class"}
GLOBAL_FEATURE_NAMES = FEATURE_NAMES[:-1]


def smiles_to_graph(smiles, global_feats=None, y=None):
    """convert one SMILES string into a PyG data object. None for unparseable SMILES"""
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


def load_graph_dataset(path=SMILES_DATA_PATH):
    """Load data and convert each row to graph, with that row's QSAR
    descriptors as global features and Class (RB=1/NRB=0) as the label. The original
    Rows with unparseable SMILES are dropped; returns (graphs, n_skipped)."""
    df = pd.read_csv(path).rename(columns=_RENAME_MAP)
    y = np.where(df["Class"] == "RB", 1, 0)

    graphs = []
    n_skipped = 0
    for i, row in df.iterrows():
        graph = smiles_to_graph(row["Smiles"], global_feats=row[GLOBAL_FEATURE_NAMES].to_numpy(dtype=float), y=y[i])
        if graph is None:
            n_skipped += 1
            continue
        graph.status = row["Status"]
        graphs.append(graph)

    return graphs, n_skipped


_SCALE_IDX = [i for i, name in enumerate(GLOBAL_FEATURE_NAMES) if name not in BINARY_FEATS]


def holdout_split(graphs, test_size=0.2, random_state=1):
    """Stratified split, called once to keep a final test set out of CV entirely."""
    labels = [int(g.y.item()) for g in graphs]
    return train_test_split(graphs, test_size=test_size, stratify=labels, random_state=random_state)


def _global_feats_matrix(graphs):
    return torch.cat([g.global_feats for g in graphs], dim=0).numpy()


def _with_global_feats(graphs, matrix):
    scaled = [g.clone() for g in graphs]
    for g, row in zip(scaled, matrix):
        g.global_feats = torch.tensor(row, dtype=torch.float).unsqueeze(0)
    return scaled


def transform_graph_features(graphs, scaler):
    """Apply an already-fit scaler to graphs' global_feats, leaving binary_feats untouched."""
    matrix = _global_feats_matrix(graphs)
    matrix[:, _SCALE_IDX] = scaler.transform(matrix[:, _SCALE_IDX])
    return _with_global_feats(graphs, matrix)


def scale_graph_features(train_graphs, val_graphs, method="standardize"):
    """Fit a scaler on train_graphs' global_feats, apply to train/val. Returns
    (train_graphs_scaled, val_graphs_scaled, scaler); originals are left untouched."""
    scaler = SCALERS[method]()
    train_matrix = _global_feats_matrix(train_graphs)
    train_matrix[:, _SCALE_IDX] = scaler.fit_transform(train_matrix[:, _SCALE_IDX])
    train_scaled = _with_global_feats(train_graphs, train_matrix)
    return train_scaled, transform_graph_features(val_graphs, scaler), scaler


def cv_folds(graphs, method="standardize", n_splits=5, random_state=1):
    """Stratified CV folds, scaling fit per-fold (train only, no leakage). Yields
    (train_graphs, val_graphs) tuples, mirroring de.cv.cv_folds for tabular data."""
    labels = np.array([int(g.y.item()) for g in graphs])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in skf.split(graphs, labels):
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        yield scale_graph_features(train_graphs, val_graphs, method=method)[:2]
