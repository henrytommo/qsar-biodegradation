"""
Feature analysis:
1. multicollinearity -> clusters of correlated features for manual selection
2. rare and low-importance features
3. cross-model importance (XGBoost gain + logistic-regression |coef|) Outputs ->
outputs/feature_analysis/.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # xgboost/torch OpenMP clash

from pathlib import Path

import numpy as np
import pandas as pd
from exp.registry import MODELS  # imports xgboost before torch (order-safe)

from de.data import BINARY_FEATS
from de.dataset import Dataset
from exp import config
from exp.spec import evaluate_final

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "feature_analysis"
HIGH_CORR_THRESHOLD = 0.85
SPARSE_THRESHOLD = 0.05  # nonzero fraction below which a feature counts as rare
WEAK_RANK_QUANTILE = 0.7  # bottom 30% of features by avg_rank counts as "weak"


def multicollinearity(X, cont_feats, threshold=HIGH_CORR_THRESHOLD):
    corr = X[cont_feats].corr()
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
    pairs.columns = ["feat_a", "feat_b", "corr"]
    pairs["abs_corr"] = pairs["corr"].abs()
    return corr, pairs[pairs["abs_corr"] > threshold].sort_values("abs_corr", ascending=False)


def target_correlation(X, y):
    return X.apply(lambda col: np.corrcoef(col, y)[0, 1])


def correlation_clusters(high_corr):
    """(A~B, B~C makes one cluster, not two pairs), keep one representative per
    component rather than dropping pairwise. Sorted largest-first"""
    adj = {}
    for _, row in high_corr.iterrows():
        adj.setdefault(row["feat_a"], set()).add(row["feat_b"])
        adj.setdefault(row["feat_b"], set()).add(row["feat_a"]) # both ways, undirected graph

    seen, clusters = set(), []
    for node in adj:
        if node in seen:
            continue
        stack, comp = [node], []
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.append(n)
            stack.extend(adj[n] - seen)
        clusters.append(comp)
    return sorted(clusters, key=lambda c: (-len(c), sorted(c)[0]))


def cluster_table(clusters, X, y):
    """One row per (cluster, feature) with the model-free stats for choosing which member to
    keep: coverage (nonzero fraction), spread (std) and |target correlation|. `heuristic_keep`
    marks a default pick -- highest coverage, then spread, then target corr. can be overridden.
    listed best-default-first."""
    nonzero_frac = (X != 0).mean()
    std = X.std()
    abs_tcorr = target_correlation(X, y).abs()
    key = lambda f: (nonzero_frac[f], std[f], abs_tcorr[f])

    rows = []
    for cid, members in enumerate(clusters, 1):
        keep = max(members, key=key)
        for f in sorted(members, key=key, reverse=True):
            rows.append({
                "cluster": cid, "size": len(members), "feature": f,
                "nonzero_frac": round(nonzero_frac[f], 3), "std": round(std[f], 3),
                "abs_target_corr": round(abs_tcorr[f], 3), "heuristic_keep": f == keep,
            })
    return pd.DataFrame(rows)


def model_importance_summary(dataset):
    """Fit XGBoost and logistic regression once on the holdout, then cross-reference their
    feature rankings (gain vs |coef|) alongside target correlation."""
    train, test = dataset.holdout()
    xgb, logreg = MODELS["xgb"], MODELS["logreg"]
    xp, xm, _ = config.load("xgb")
    lp, lm, _ = config.load("logreg")

    xgb_model, _ = evaluate_final(xgb, train, test, xp, method=xm)
    lr_model, _ = evaluate_final(logreg, train, test, lp, method=lm)

    names = dataset.feature_names
    xgb_gain = xgb.ranking(xgb_model, names)
    lr_coef = logreg.ranking(lr_model, names).reindex(names)

    summary = pd.DataFrame({
        "xgb_gain": xgb_gain,
        "xgb_rank": xgb_gain.rank(ascending=False),
        "lr_abs_coef": lr_coef,
        "lr_rank": lr_coef.rank(ascending=False),
        "target_corr": target_correlation(dataset.data, dataset.y),
    })
    summary["avg_rank"] = (summary["xgb_rank"] + summary["lr_rank"]) / 2
    return summary.sort_values("avg_rank", ascending=False)


def flag_rare_features(X, summary, sparse_threshold=SPARSE_THRESHOLD, weak_rank_quantile=WEAK_RANK_QUANTILE):
    """Automatic drop candidates, independent of correlation: features that are both rare
    (< sparse_threshold nonzero) and weak (avg_rank in the bottom (1 - weak_rank_quantile))."""
    nonzero_frac = (X != 0).mean()
    rank_cutoff = summary["avg_rank"].quantile(weak_rank_quantile)

    drops = {}
    for feat in X.columns:
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


def write_report(ct, rare, n_features, out_path):
    n_clusters = ct["cluster"].nunique() if not ct.empty else 0
    lines = [
        "# Feature Analysis",
        "",
        "Cross-referencing multicollinearity (`correlation_clusters.csv`, "
        "`correlation_heatmap.png`), coverage/spread, target correlation, and "
        "XGBoost/logistic-regression importance (`feature_summary.csv`).",
        "",
        "## Correlated clusters (manual selection)",
        "",
        f"{n_clusters} clusters of features with |r| > {HIGH_CORR_THRESHOLD} (connected "
        "components). Keep **one** representative per cluster; see `correlation_clusters.csv` "
        "for coverage/spread/target-corr per member. The **bold** member is the model-free "
        "`heuristic_keep` default -- override with domain knowledge.",
        "",
    ]
    for cid, grp in ct.groupby("cluster"):
        keep = grp.loc[grp["heuristic_keep"], "feature"].iloc[0]
        members = ", ".join(f"**{f}**" if f == keep else f for f in grp["feature"])
        lines.append(f"- **Cluster {cid}**: {members}")

    lines += [
        "",
        "## Auto-flagged rare features",
        "",
        "Rare *and* low-importance (independent of correlation) -- safe automatic drops:",
        "",
    ]
    for feat, reason in rare.items():
        lines.append(f"- **{feat}**: {reason}")

    lines += [
        "",
        f"{n_features} features total: {len(rare)} auto-flagged, "
        f"{len(clusters)} correlated clusters awaiting manual selection.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.load()  # tabular, incl. the GFN1-xTB homo_lumo_gap, matching the models
    X, y = dataset.data, dataset.y
    cont_feats = [c for c in X.columns if c not in BINARY_FEATS]

    corr, high_corr = multicollinearity(X, cont_feats)
    high_corr.to_csv(OUT_DIR / "high_corr_pairs.csv", index=False)
    print(f"Saved {len(high_corr)} feature pairs with |r| > {HIGH_CORR_THRESHOLD} to high_corr_pairs.csv")

    plot_correlation_heatmap(corr, OUT_DIR / "correlation_heatmap.png")
    print("Saved correlation_heatmap.png")

    clusters = correlation_clusters(high_corr)
    ct = cluster_table(clusters, X, y)
    ct.to_csv(OUT_DIR / "correlation_clusters.csv", index=False)
    print(f"Saved correlation_clusters.csv ({len(clusters)} clusters for manual selection)")

    summary = model_importance_summary(dataset)
    summary.to_csv(OUT_DIR / "feature_summary.csv")
    print("Saved feature_summary.csv (xgb gain, lr |coef|, target corr, ranks)")

    rare = flag_rare_features(X, summary)
    pd.Series(rare, name="reason").rename_axis("feature").to_csv(OUT_DIR / "recommended_drops.csv")
    print(f"Saved recommended_drops.csv ({len(rare)} rare/weak features)")

    write_report(ct, rare, len(X.columns), OUT_DIR / "report.md")
    print(f"Saved report.md ({len(clusters)} clusters, {len(rare)} auto-flagged)")
