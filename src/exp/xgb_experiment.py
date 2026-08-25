"""
xgb
"""
import pandas as pd
from xgboost import XGBClassifier

from exp.spec import ModelSpec, tabular_predict

PARAM_DISTRIBUTIONS = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.5, 1.0],
}

def gain_ranking(model, columns):
    """Features by total_gain (desc), reindexed to columns (never-split -> 0, not missing)."""
    gain = model.get_booster().get_score(importance_type="total_gain")
    return pd.Series(gain).reindex(columns).fillna(0).sort_values(ascending=False)


def _make_model(params, random_state=1):
    return XGBClassifier(
        **{k: params[k] for k in PARAM_DISTRIBUTIONS},   # the searched hyperparameters
        n_estimators=params.get("n_estimators", 1000),
        early_stopping_rounds=params.get("early_stopping_rounds", 30),
        objective="binary:logistic", eval_metric="aucpr", random_state=random_state, n_jobs=-1,
    )


def _build(params, dims, random_state):
    return _make_model(params, random_state=random_state)   # dims unused: sklearn infers shape


def _fit(model, train, val, params):
    """Class-imbalance-weighted fit with early stopping on the val slice."""
    pos = train.y.sum()
    neg = len(train.y) - pos
    model.set_params(scale_pos_weight=neg / pos)
    model.fit(train.data, train.y, eval_set=[(val.data, val.y)], verbose=False)
    return model, {"best_iteration": model.best_iteration}


SPEC = ModelSpec(
    name="xgb",
    param_distributions=PARAM_DISTRIBUTIONS,
    build=_build,
    fit=_fit,
    predict=tabular_predict,
    finalize=lambda best: {**best["params"], "n_estimators": max(round(best["best_iteration_mean"]) + 1, 10)},
    ranking=gain_ranking,
)
