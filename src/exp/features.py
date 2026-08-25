"""Feature-count validation: rank features leak-free per seed (xgb gain by default), then
retrain on the top-N for a sweep of N. One seed = a sweep; several = a stability check. For a
graph target, ranking uses a tabular load (names index the graph's global_feats).

    python -m exp.features xgb --ns 15 30 42 --seeds 10
    python -m exp.features gcn --ns 15 30 42
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

import pandas as pd
from exp.registry import MODELS  # imports xgboost before torch (order-safe)

import torch

torch.set_num_threads(1)

from de.dataset import Dataset
from exp import config
from exp.spec import evaluate_final

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "feature_sweep"


def _rank_on_train(ranker, train, params, random_state):
    """Feature order from the ranker fit on the train split only (no test leakage)."""
    tr, val = train.holdout(test_size=0.15, random_state=random_state)
    tr_s, val_s, _ = tr.scale(val)
    model, _ = ranker.fit(ranker.build(params, train.dims, random_state), tr_s, val_s, params)
    return list(ranker.ranking(model, train.feature_names).index)


def validate_feature_counts(spec, dataset, ns, params, method, seeds=(1,), ranker=None,
                            rank_dataset=None, ranker_params=None):
    """Per seed: rank on the train split, evaluate the top-N for each N. Returns a tidy
    (seed, n, *metrics) DataFrame."""
    ranker = ranker or MODELS["xgb"]
    rank_dataset = rank_dataset or dataset

    records = []
    for seed in seeds:
        order = _rank_on_train(ranker, rank_dataset.holdout(random_state=seed)[0], ranker_params, seed)
        train, test = dataset.holdout(random_state=seed)
        for n in ns:
            names = order[:n]
            _, metrics = evaluate_final(spec, train.subset(names), test.subset(names),
                                        params, method=method, random_state=seed)
            records.append({"seed": seed, "n": n, **metrics})
            print(f"seed {seed:>2}  top {n:>2}:  " + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
    return pd.DataFrame(records)


def summarise(df):
    """Per-N mean +/- std; with >1 seed, the paired (N vs largest-N) F1 diff."""
    seeds = df["seed"].nunique()
    print(f"\n--- test metrics across {seeds} seed(s) (mean +/- std) ---")
    for n, grp in df.groupby("n"):
        print(f"  top {n:>2}:  " + "  ".join(
            f"{m} {grp[m].mean():.3f}+/-{grp[m].std():.3f}" for m in ("f1", "roc_auc", "pr_auc")))
    if seeds > 1:
        full = df["n"].max()
        wide = df.pivot(index="seed", columns="n", values="f1")
        for n in sorted(c for c in wide.columns if c != full):
            diff = wide[n] - wide[full]
            print(f"\nPaired F1 (top {n} - top {full}) per seed: mean {diff.mean():+.3f} "
                  f"+/- {diff.std():.3f}, top-{n} wins {(diff > 0).sum()}/{seeds} splits")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Feature-count validation for any registered model.")
    p.add_argument("model", nargs="?", default="xgb", choices=list(MODELS))
    p.add_argument("--ns", type=int, nargs="+", default=[15, 20, 25, 30], help="feature counts to test")
    p.add_argument("--seeds", type=int, default=1, help="number of holdout seeds (1 = plain sweep)")
    args = p.parse_args()

    spec = MODELS[args.model]
    params, method, _ = config.load(args.model)  # eval params, held fixed while N varies
    ranker_params = config.load("xgb")[0]        # ranker (xgb) params
    dataset = Dataset.load(graph=spec.graph)
    rank_dataset = dataset if not spec.graph else Dataset.load(graph=False)

    df = validate_feature_counts(spec, dataset, args.ns, params, method,
                                 seeds=range(1, args.seeds + 1), rank_dataset=rank_dataset,
                                 ranker_params=ranker_params)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.model}_feature_counts.csv"
    df.to_csv(out, index=False)
    summarise(df)
    print(f"\nSaved per-seed results to {out}")
