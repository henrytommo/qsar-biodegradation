"""ModelSpec: the plugin contract each model fills in (build/fit/predict over Datasets), plus
the generic routines that consume it -- cv_score (hyperparameter CV) and evaluate_final (test
fit). Tabular-vs-graph differences live inside each model's callables, not here."""
from dataclasses import dataclass
from typing import Callable

from utils.metrics import aggregate_metrics, classification_metrics


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_distributions: dict           # random_search space
    build: Callable                     # (params, dims, random_state) -> fresh model
    fit: Callable                       # (model, train_ds, val_ds, params) -> (model, aux_dict)
    predict: Callable                   # (model, ds) -> (y_true, preds, proba)
    graph: bool = False                 # True -> needs Dataset.load(graph=True) (GCN); False -> tabular
    scaling_methods: tuple = ("standardize",)   # logreg trials both; others are scale-insensitive
    val_size: float = 0.15              # internal val slice for early stopping; 0 = fit on full train
    feature_counts: tuple = (15, 20, 25, 30)   # top-N grid searched under tune --select ("keep all" added at runtime)
    finalize: Callable = lambda best: best["params"]   # CV winner dict -> final-fit params
    ranking: Callable = None            # (model, feature_names) -> pd.Series | None (feature importance)


def tabular_predict(model, ds):
    """Shared predict for sklearn-style tabular models (XGBoost, logistic regression)."""
    proba = model.predict_proba(ds.data)[:, 1]
    preds = model.predict(ds.data)
    return ds.y, preds, proba


def rank_features(ranker, scaled_train, params, random_state=1):
    """Feature order (best first) from `ranker` fit with `params` on an already-scaled train
    Dataset; works for graphs via feature_frame (ranks on global_feats)."""
    from de.dataset import Dataset

    X, y = scaled_train.feature_frame()
    rds = Dataset(data=X, y=y, feature_names=list(X.columns))
    model = ranker.build(params, rds.dims, random_state)
    model, _ = ranker.fit(model, rds, rds, params)
    return list(ranker.ranking(model, list(X.columns)).index)


def cv_score(spec, train_ds, n_splits=5, select=False, ranker=None, ranker_params=None):
    """A scorer(params, method, random_state) for random_search: CV on train_ds via the spec,
    aggregated to *_mean/*_std. select=True treats params["n_features"] as a dimension --
    features are ranked per fold (leak-free) and both folds cut to the top-N before fitting.
    A cross-model ranker uses ranker_params; self-ranking uses the sampled params."""
    def score(params, method, random_state):
        model_params = {k: v for k, v in params.items() if k != "n_features"}
        n = params.get("n_features") if select else None
        fold_metrics = []
        for tr, val in train_ds.cv_folds(method=method, n_splits=n_splits, random_state=random_state):
            if n is not None:
                names = rank_features(ranker or spec, tr, ranker_params or model_params, random_state)[:n]
                tr, val = tr.subset(names), val.subset(names)
            model = spec.build(model_params, tr.dims, random_state)
            model, aux = spec.fit(model, tr, val, model_params)
            y, preds, proba = spec.predict(model, val)
            fold_metrics.append({**classification_metrics(y, preds, proba), **aux})
        return {"params": model_params, "n_features": n, "method": method, **aggregate_metrics(fold_metrics)}
    return score


def evaluate_final(spec, train_ds, test_ds, params, method="standardize", random_state=1):
    """Refit on train_ds, evaluate once on test_ds. val_size>0 carves an internal early-stopping
    slice; val_size 0 fits on the full train. Scaler fit on train only."""
    if spec.val_size:
        tr, val = train_ds.holdout(test_size=spec.val_size, random_state=random_state)
        tr_s, val_s, scaler = tr.scale(val, method=method)
        test_s = test_ds.transform(scaler)
    else:
        tr_s, test_s, scaler = train_ds.scale(test_ds, method=method)
        val_s = None

    model = spec.build(params, train_ds.dims, random_state)
    model, _ = spec.fit(model, tr_s, val_s, params)
    y, preds, proba = spec.predict(model, test_s)
    return model, classification_metrics(y, preds, proba)
