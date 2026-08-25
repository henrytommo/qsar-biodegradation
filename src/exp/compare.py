"""Model comparison: fix each model at its config params/features and vary only the holdout
split across N seeds, then compare test-metric distributions. Models sharing a data shape
(tabular xgb/logreg) share each split, so their F1 gap is a matched-pair diff; a graph model
(gcn) is compared by distribution only.

    python -m exp.compare               # all models, 10 seeds
    python -m exp.compare xgb logreg --seeds 20
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from exp.registry import MODELS  # imports xgboost before torch (order-safe)

import torch

torch.set_num_threads(1)

from de.dataset import Dataset
from exp import config
from exp.spec import evaluate_final

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "model_comparison"
METRICS = ["f1", "precision", "recall", "roc_auc", "pr_auc"]


def compare(names, seeds=range(1, 11)):
    """Evaluate each model at its config params + selected features across the seeds."""
    specs = [MODELS[n] for n in names]
    tab = Dataset.load(log=lambda *_: None)
    graph = Dataset.load(graph=True, log=lambda *_: None) if any(s.graph for s in specs) else None

    # per model: params, scaling method, and the (optionally feature-subset) dataset
    plan = {}
    for spec in specs:
        params, method, features = config.load(spec.name)
        base = graph if spec.graph else tab
        plan[spec.name] = (params, method, base.subset(features) if features else base)

    records = []
    for seed in seeds:
        for spec in specs:
            params, method, ds = plan[spec.name]
            train, test = ds.holdout(random_state=seed)
            _, metrics = evaluate_final(spec, train, test, params, method=method, random_state=seed)
            records.append({"seed": seed, "model": spec.name, **metrics})
        print(f"seed {seed:>2} done")
    return pd.DataFrame(records)


def summarise(df):
    """Per-model mean +/- std, then matched-pair F1 diffs between models sharing a data shape."""
    print(f"\n--- test metrics across {df['seed'].nunique()} holdout seeds (mean +/- std) ---")
    for name in df["model"].unique():
        g = df[df["model"] == name]
        print(f"  {name:>6}:  " + "   ".join(f"{k} {g[k].mean():.3f}+/-{g[k].std():.3f}" for k in METRICS))

    wide = df.pivot(index="seed", columns="model", values="f1")
    for a, b in combinations(df["model"].unique(), 2):
        if MODELS[a].graph != MODELS[b].graph:
            continue
        d = (wide[a] - wide[b]).dropna()
        print(f"\nPaired F1 ({a} - {b}) [matched split]: mean {d.mean():+.3f} +/- {d.std():.3f}, "
              f"{a} wins {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-seed model comparison at tuned params.")
    p.add_argument("models", nargs="*", default=list(MODELS), choices=list(MODELS),
                   help="models to compare (default: all registered)")
    p.add_argument("--seeds", type=int, default=10, help="number of holdout seeds")
    args = p.parse_args()

    df = compare(args.models, seeds=range(1, args.seeds + 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "model_stability.csv"
    df.to_csv(out, index=False)
    summarise(df)
    print(f"\nSaved per-seed results to {out}")
