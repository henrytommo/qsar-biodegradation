"""Logistic-regression plugin. Scale-sensitive, so it trials both scaling methods; no early
stopping (val_size 0, fits full train); ranks features by |coef| on scaled features."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from exp.spec import ModelSpec, tabular_predict

PARAM_DISTRIBUTIONS = {
    "C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    "l1_ratio": [0.0, 0.5, 1.0],  # 0 = pure L2, 1 = pure L1, elastic-net between
}

def _make_model(params, random_state=1):
    return LogisticRegression(
        C=params["C"],
        l1_ratio=params["l1_ratio"],
        solver="saga",  # supports l1 penalty
        class_weight="balanced",  # class imbalance
        max_iter=5000,
        random_state=random_state,
    )


def _build(params, dims, random_state):
    return _make_model(params, random_state=random_state)


def _fit(model, train, val, params):
    model.fit(train.data, train.y)   # no early stopping -> val unused, fit on full train
    return model, {}


def _ranking(model, feature_names):
    return pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)


SPEC = ModelSpec(
    name="logreg",
    param_distributions=PARAM_DISTRIBUTIONS,
    build=_build,
    fit=_fit,
    predict=tabular_predict,
    scaling_methods=("standardize", "normalize"),
    val_size=0.0,   # no early stopping: fit on the full train split
    ranking=_ranking,
)
