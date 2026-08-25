"""inference API. Score a molecule CSV (Smiles + the 41 qsar descriptors, as in
data/smiles_data.csv) with either pipeline and get a prediction per input row.

    from inference.pipeline import predict
    predict("data/smiles_external_val.csv", pipeline="ensemble")

Molecules whose xTB features can't be computed (embedding / SCF failure) are marked "abstain"
rather than dropped, so the output row-aligns with the input."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # xgboost/torch OpenMP clash

import numpy as np
import pandas as pd
from exp.registry import MODELS   # imports xgboost before torch (order-safe)

import torch

torch.set_num_threads(1)

from de.dataset import Dataset
from inference.ensemble import EnsemblePredictor
from inference.predictor import Predictor


def _build(pipeline):
    """(predictor, is_graph) for the named pipeline."""
    if pipeline == "ensemble":
        return EnsemblePredictor(("xgb", "logreg")), False
    if pipeline == "gcn":
        return Predictor("gcn"), True
    raise ValueError(f"unknown pipeline {pipeline!r}; expected 'ensemble' or 'gcn'")


def predict(path, pipeline="ensemble", threshold=0.5, predictor=None, log=print):
    """Predict RB/NRB for every molecule in `path`. Returns a DataFrame aligned to the input rows
    with columns [Smiles, proba, prediction]; unscorable molecules get proba NaN / prediction
    'abstain'. Pass a pre-built `predictor` to avoid refitting across calls."""
    is_graph = pipeline == "gcn"
    if predictor is None:
        predictor, is_graph = _build(pipeline)

    ds = Dataset.from_csv(path, graph=is_graph, labeled=False, log=log)
    proba, label = predictor.predict(ds, threshold)
    ids = [g.mol_id for g in ds.data] if is_graph else list(ds.data.index)

    scored = pd.DataFrame({"proba": proba, "prediction": np.where(label == 1, "RB", "NRB")}, index=ids)
    out = pd.read_csv(path)[["Smiles"]].copy()
    out["proba"] = scored["proba"].reindex(out.index)
    out["prediction"] = scored["prediction"].reindex(out.index).fillna("abstain")
    return out
