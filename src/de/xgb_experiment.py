"""XGBoost experiment: random hyperparameter search over CV folds, then a final
fit/eval on the held-out test set. note scaling method doesn't affect results as much.
"""
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from de.cv import cv_folds
from de.preprocessing import scale_features, transform_features
from utils.metrics import aggregate_metrics, classification_metrics
from utils.search import random_search

PARAM_DISTRIBUTIONS = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.5, 1.0],
}


def _make_model(params, random_state=1):
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 1000),
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=params.get("early_stopping_rounds", 30),
        random_state=random_state,
        n_jobs=-1,
    )


def _score_params(params, method, random_state, X, y, binary_feats, n_splits=5):
    fold_metrics = []
    for fold in cv_folds(X, y, binary_feats, method=method, n_splits=n_splits, random_state=random_state):
        pos = fold.y_train.sum()
        neg = len(fold.y_train) - pos

        model = _make_model(params, random_state=random_state)
        model.set_params(scale_pos_weight=neg / pos)
        model.fit(fold.X_train, fold.y_train, eval_set=[(fold.X_val, fold.y_val)], verbose=False)

        preds = model.predict(fold.X_val)
        proba = model.predict_proba(fold.X_val)[:, 1]
        metrics = classification_metrics(fold.y_val, preds, proba)
        metrics["best_iteration"] = model.best_iteration
        fold_metrics.append(metrics)

    return {"params": params, "method": method, **aggregate_metrics(fold_metrics)}


def fit_final_model(X_train, y_train, X_test, y_test, binary_feats, params,
                     method="standardize", val_size=0.15, random_state=1):
    """Refit with a small internal val slice for early stopping, evaluate once on X_test."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )

    X_tr_scaled, X_val_scaled, scaler = scale_features(X_tr, X_val, binary_feats, method=method)
    X_test_scaled = transform_features(X_test, scaler, binary_feats)

    pos = y_tr.sum()
    neg = len(y_tr) - pos

    model = _make_model(params, random_state=random_state)
    model.set_params(scale_pos_weight=neg / pos)
    model.fit(X_tr_scaled, y_tr, eval_set=[(X_val_scaled, y_val)], verbose=False)

    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    return model, classification_metrics(y_test, preds, proba), X_test_scaled


def plot_feature_importance(model, save_path=None, importance_type="total_gain", max_features=None):
    import matplotlib.pyplot as plt
    import xgboost as xgb

    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, importance_type=importance_type,
                         max_num_features=max_features, grid=False, show_values=False)
    plt.xlabel(importance_type)
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    from pathlib import Path

    from de.cv import holdout_split
    from de.data import BINARY_FEATS, load_data

    X, y = load_data()
    X_train, X_test, y_train, y_test = holdout_split(X, y)

    print(f"Random search: {len(PARAM_DISTRIBUTIONS)} hyperparams, 5-fold CV, train shape {X_train.shape}")
    results = random_search(_score_params, PARAM_DISTRIBUTIONS, n_iter=25,
                             X=X_train, y=y_train, binary_feats=BINARY_FEATS)

    best = results[0]
    print("\nBest CV params:", best["params"])
    print(f"CV F1:      {best['f1_mean']:.3f} +/- {best['f1_std']:.3f}")
    print(f"CV ROC-AUC: {best['roc_auc_mean']:.3f} +/- {best['roc_auc_std']:.3f}")
    print(f"CV PR-AUC:  {best['pr_auc_mean']:.3f} +/- {best['pr_auc_std']:.3f}")

    n_estimators = max(int(round(best["best_iteration_mean"])) + 1, 10)
    final_params = {**best["params"], "n_estimators": n_estimators}

    model, metrics, X_test_scaled = fit_final_model(X_train, y_train, X_test, y_test, BINARY_FEATS, final_params)
    print("\nFinal test metrics:", {k: round(v, 3) for k, v in metrics.items()})

    out_dir = Path(__file__).resolve().parents[2] / "outputs"
    out_dir.mkdir(exist_ok=True)
    plot_feature_importance(model, save_path=out_dir / "xgb_feature_importance.png")
    print(f"\nSaved feature importance plot to {out_dir / 'xgb_feature_importance.png'}")
