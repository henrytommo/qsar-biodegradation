"""Feature analysis: multicollinearity, target correlation, and cross-model importance
(XGBoost gain + logistic-regression |coefficient|), used to recommend features to drop
before running a reduced-feature-set experiment.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from de import logreg_experiment as lrx
from de import xgb_experiment as xgbx
from de.cv import holdout_split
from de.data import BINARY_FEATS, load_data
from utils.logger import Logger

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis"
HIGH_CORR_THRESHOLD = 0.85
SPARSE_THRESHOLD = 0.05  # nonzero fraction below which a feature counts as rare
WEAK_RANK_QUANTILE = 0.7  # bottom 30% of features by avg_rank counts as "weak"

# hyperparameters already found best via random_search in xgb_experiment.py / logreg_experiment.py
XGB_PARAMS = {"max_depth": 5, "learning_rate": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
              "min_child_weight": 3, "gamma": 0.1}
LR_PARAMS = {"C": 10, "l1_ratio": 0.5}
LR_METHOD = "normalize"


def multicollinearity(X, cont_feats, threshold=HIGH_CORR_THRESHOLD):
    corr = X[cont_feats].corr()
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
    pairs.columns = ["feat_a", "feat_b", "corr"]
    pairs["abs_corr"] = pairs["corr"].abs()
    return corr, pairs[pairs["abs_corr"] > threshold].sort_values("abs_corr", ascending=False)


def target_correlation(X, y):
    return X.apply(lambda col: np.corrcoef(col, y)[0, 1])


def model_importance_summary(X, y, binary_feats):
    X_train, X_test, y_train, y_test = holdout_split(X, y)

    xgb_model, _, _ = xgbx.fit_final_model(X_train, y_train, X_test, y_test, binary_feats, XGB_PARAMS)
    lr_model, _, _ = lrx.fit_final_model(X_train, y_train, X_test, y_test, binary_feats, LR_PARAMS, method=LR_METHOD)

    xgb_gain = pd.Series(xgb_model.get_booster().get_score(importance_type="total_gain")).reindex(X.columns).fillna(0)
    lr_coef = pd.Series(np.abs(lr_model.coef_[0]), index=X_train.columns)

    summary = pd.DataFrame({
        "xgb_gain": xgb_gain,
        "xgb_rank": xgb_gain.rank(ascending=False),
        "lr_abs_coef": lr_coef,
        "lr_rank": lr_coef.rank(ascending=False),
        "target_corr": target_correlation(X, y),
    })
    summary["avg_rank"] = (summary["xgb_rank"] + summary["lr_rank"]) / 2
    return summary.sort_values("avg_rank", ascending=False)


def recommend_drops(X, summary, high_corr, sparse_threshold=SPARSE_THRESHOLD, weak_rank_quantile=WEAK_RANK_QUANTILE):
    """Derive drop candidates from computed stats: within each high-correlation pair, drop
    the weaker-ranked (avg_rank) member; separately flag rare features (< sparse_threshold
    nonzero) whose avg_rank falls in the bottom (1 - weak_rank_quantile) share of all features."""
    drops = {}

    # tie-break on xgb_gain, not avg_rank: L1 can zero one member of a correlated pair
    # would bias avg_rank against whichever feature the LR fit happened to zero.
    for _, row in high_corr.iterrows():
        a, b = row["feat_a"], row["feat_b"]
        worse, better = (a, b) if summary.loc[a, "xgb_gain"] < summary.loc[b, "xgb_gain"] else (b, a)
        drops.setdefault(worse, f"r={row['corr']:.2f} with {better} (xgb_gain {summary.loc[worse, 'xgb_gain']:.1f} "
                                 f"vs {summary.loc[better, 'xgb_gain']:.1f} kept) -- redundant")

    nonzero_frac = (X != 0).mean()
    rank_cutoff = summary["avg_rank"].quantile(weak_rank_quantile)
    for feat in X.columns:
        if feat in drops:
            continue
        if nonzero_frac[feat] < sparse_threshold and summary.loc[feat, "avg_rank"] >= rank_cutoff:
            drops[feat] = (f"{nonzero_frac[feat] * 100:.1f}% nonzero, avg_rank {summary.loc[feat, 'avg_rank']:.1f} "
                            f"(bottom {(1 - weak_rank_quantile) * 100:.0f}%) -- rare, weak signal")

    return drops


def plot_correlation_heatmap(corr, save_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return fig


def write_report(drops, n_features, out_path):
    lines = [
        "# Feature Analysis",
        "",
        "Cross-referencing multicollinearity (`high_corr_pairs.csv`, `correlation_heatmap.png`), "
        "target correlation, and XGBoost/logistic-regression importance (`feature_summary.csv`) "
        "to identify redundant or low-signal features.",
        "",
        "## Recommended drops",
        "",
    ]
    for feat, reason in drops.items():
        lines.append(f"- **{feat}**: {reason}")
    lines.append("")
    lines.append(f"{n_features - len(drops)}/{n_features} features remain after dropping these {len(drops)}.")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    log = Logger().log
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    cont_feats = [c for c in X.columns if c not in BINARY_FEATS]

    corr, high_corr = multicollinearity(X, cont_feats)
    high_corr.to_csv(OUT_DIR / "high_corr_pairs.csv", index=False)
    log(f"Saved {len(high_corr)} feature pairs with |r| > {HIGH_CORR_THRESHOLD} to high_corr_pairs.csv")

    plot_correlation_heatmap(corr, OUT_DIR / "correlation_heatmap.png")
    log("Saved correlation_heatmap.png")

    summary = model_importance_summary(X, y, BINARY_FEATS)
    summary.to_csv(OUT_DIR / "feature_summary.csv")
    log("Saved feature_summary.csv (xgb gain, lr |coef|, target corr, ranks)")

    drops = recommend_drops(X, summary, high_corr)
    pd.Series(drops, name="reason").rename_axis("feature").to_csv(OUT_DIR / "recommended_drops.csv")
    log(f"Saved recommended_drops.csv ({len(drops)} features)")

    write_report(drops, len(X.columns), OUT_DIR / "report.md")
    log(f"Saved report.md ({len(drops)} recommended drops)")
