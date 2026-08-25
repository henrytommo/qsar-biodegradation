"""External validation: score data/smiles_external_val.csv (669 molecules held out from training)
with both pipelines and report metrics against the true labels

    python -m inference.validate

First run computes the xTB features for the external SMILES (~669 x 3 SCFs, a few minutes) and
caches them; later runs are fast.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # xgboost/torch OpenMP clash -- see exp/registry.py

from pathlib import Path

import numpy as np
import pandas as pd
from exp.registry import MODELS   # imports xgboost before torch (order-safe)

import torch

torch.set_num_threads(1)

from de.dataset import Dataset
from inference.ensemble import EnsemblePredictor
from inference.predictor import Predictor
from utils.metrics import classification_metrics

EXTERNAL = Path(__file__).resolve().parents[2] / "data" / "smiles_external_val.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "inference"
METRICS = ["f1", "precision", "recall", "roc_auc", "pr_auc"]


def _labels(ds):
    return np.array([int(g.y.item()) for g in ds.data]) if ds.is_graph else ds.y


def _ids(ds):
    return [g.mol_id for g in ds.data] if ds.is_graph else list(ds.data.index)


def validate(path=EXTERNAL):
    n_total = len(pd.read_csv(path))
    pipelines = [("ensemble", EnsemblePredictor(("xgb", "logreg")), False),
                 ("gcn", Predictor("gcn"), True)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, predictor, graph in pipelines:
        ds = Dataset.from_csv(path, graph=graph, labeled=True, log=print)
        proba, preds = predictor.predict(ds, threshold=0.5)
        y = _labels(ds)
        summary[name] = (classification_metrics(y, preds, proba), len(y))
        pd.DataFrame({"mol_id": _ids(ds), "y": y, "proba": proba, "pred": preds}) \
            .to_csv(OUT_DIR / f"{name}_external.csv", index=False)

    print(f"\n--- external validation on {n_total} molecules (data/{path.name}) ---")
    for name, (m, n_scored) in summary.items():
        abstained = n_total - n_scored
        tag = f"  [{abstained} abstained]" if abstained else ""
        print(f"  {name:>9}: " + "  ".join(f"{k} {m[k]:.3f}" for k in METRICS) + tag)
    print(f"\nsaved per-molecule predictions to {OUT_DIR}")


if __name__ == "__main__":
    validate()
