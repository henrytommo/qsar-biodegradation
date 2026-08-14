"""Logistic regression experiment: random hyperparameter search over CV folds, trial
both scaling methods since linear models (unlike XGBoost) are scale sensitive.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression

from de.cv import cv_folds
from de.preprocessing import scale_features
from utils.metrics import aggregate_metrics, classification_metrics
from utils.search import random_search

PARAM_DISTRIBUTIONS = {
    "C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
    "l1_ratio": [0.0, 0.5, 1.0],  # 0 = pure L2, 1 = pure L1, elastic-net in between
}


def _make_model(params, random_state=1):
    return LogisticRegression(
        C=params["C"],
        l1_ratio=params["l1_ratio"],
        solver="saga", # supports l1 penalty
        class_weight="balanced", # class imbalance
        max_iter=5000,
        random_state=random_state,
    )


def _score_params(params, method, random_state, X, y, binary_feats, n_splits=5):
    fold_metrics = []
    for fold in cv_folds(X, y, binary_feats, method=method, n_splits=n_splits, random_state=random_state):
        model = _make_model(params, random_state=random_state)
        model.fit(fold.X_train, fold.y_train)

        preds = model.predict(fold.X_val)
        proba = model.predict_proba(fold.X_val)[:, 1]
        fold_metrics.append(classification_metrics(fold.y_val, preds, proba))

    return {"params": params, "method": method, **aggregate_metrics(fold_metrics)}


def fit_final_model(X_train, y_train, X_test, y_test, binary_feats, params, method="standardize",
                     random_state=1):
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test, binary_feats, method=method)

    model = _make_model(params, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    return model, classification_metrics(y_test, preds, proba), X_test_scaled


def plot_feature_importance(model, feature_names, method="standardize", save_path=None, max_features=None):
    """Coefficients are on scaled features, so magnitudes are directly comparable."""
    import matplotlib.pyplot as plt

    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
    if max_features:
        coefs = coefs.iloc[:max_features]
    colors = ["#1f77b4" if c > 0 else "#d62728" for c in coefs]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(coefs.index[::-1], coefs.values[::-1], color=colors[::-1])
    ax.set_xlabel(f"Coefficient ({method}d features)")
    ax.set_title("Logistic Regression Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    from pathlib import Path

    from de.cv import holdout_split
    from de.data import BINARY_FEATS, load_data
    from utils.logger import Logger

    log = Logger().log

    X, y = load_data()
    X_train, X_test, y_train, y_test = holdout_split(X, y)

    log(f"Random search: 5-fold CV over {list(PARAM_DISTRIBUTIONS)}, "
        f"methods=('standardize', 'normalize'), train shape {X_train.shape}")
    results = random_search(_score_params, PARAM_DISTRIBUTIONS, n_iter=20,
                             methods=("standardize", "normalize"),
                             X=X_train, y=y_train, binary_feats=BINARY_FEATS)

    best = results[0]
    log(f"\nBest CV params: {best['params']} | scaling: {best['method']}")
    log(f"CV F1:      {best['f1_mean']:.3f} +/- {best['f1_std']:.3f}")
    log(f"CV ROC-AUC: {best['roc_auc_mean']:.3f} +/- {best['roc_auc_std']:.3f}")
    log(f"CV PR-AUC:  {best['pr_auc_mean']:.3f} +/- {best['pr_auc_std']:.3f}")

    model, metrics, X_test_scaled = fit_final_model(
        X_train, y_train, X_test, y_test, BINARY_FEATS, best["params"], method=best["method"]
    )
    log(f"\nFinal test metrics: {({k: round(v, 3) for k, v in metrics.items()})}")

    if best["params"]["l1_ratio"] > 0:
        n_zero = int((model.coef_[0] == 0).sum())
        log(f"\nL1 component (l1_ratio={best['params']['l1_ratio']}) zeroed out "
            f"{n_zero}/{len(model.coef_[0])} coefficients")

    out_dir = Path(__file__).resolve().parents[2] / "outputs"
    out_dir.mkdir(exist_ok=True)
    plot_feature_importance(model, X_train.columns, method=best["method"],
                             save_path=out_dir / "logreg_feature_importance.png")
    log(f"Saved feature importance plot to {out_dir / 'logreg_feature_importance.png'}")
