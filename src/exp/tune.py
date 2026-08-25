"""Tune hyperparameters; --select also searches the feature count N (top-N of a leak-free
per-fold ranking) jointly. Prints a config/best_params.yml block; --write saves it.

    python -m exp.tune gcn --select --write
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # xgboost/torch OpenMP clash -- see registry.py

import argparse

from exp.registry import MODELS  # imports xgboost before torch (order-safe)

import torch

torch.set_num_threads(1)

from de.dataset import Dataset
from exp import config
from exp.spec import cv_score, evaluate_final, rank_features
from utils.search import random_search

# cap N features in the feature importance list
CAP_FEATURES = 20


def _report(best, metrics):
    for m in ("f1", "precision", "recall", "roc_auc", "pr_auc"):
        print(f"CV {m:>9}: {best[f'{m}_mean']:.3f} +/- {best[f'{m}_std']:.3f}")
    print(f"\nFinal test metrics: {({k: round(v, 3) for k, v in metrics.items()})}")


def tune(spec, dataset, n_iter=25, n_splits=5, n_repeats=3, select=False, ranker=None, ranker_params=None):
    """Random search over params (+ scaling, + feature count when select), then a final fit.
    Returns (best, metrics, model, features); features is the top-N list, or None."""
    train, test = dataset.holdout()

    space = dict(spec.param_distributions)
    if select:
        total = len(dataset.feature_names)
        if CAP_FEATURES:
            space["n_features"] = [min(CAP_FEATURES, total)]   # pinned cap -- no count search
        else:
            space["n_features"] = [n for n in spec.feature_counts if n < total] + [total]

    scorer = cv_score(spec, train, n_splits=n_splits, n_repeats=n_repeats, select=select, ranker=ranker, ranker_params=ranker_params)
    best = random_search(scorer, space, n_iter=n_iter, methods=spec.scaling_methods)[0]

    features = None
    if select:
        scaled_train, _, _ = train.scale(train, method=best["method"])
        features = rank_features(ranker or spec, scaled_train, ranker_params or best["params"])[:best["n_features"]]
        train, test = train.subset(features), test.subset(features)

    scaling = f" | scaling: {best['method']}" if len(spec.scaling_methods) > 1 else ""
    nfeat = f" | features: {best['n_features']}/{len(dataset.feature_names)}" if select else ""
    print(f"\nBest CV params: {best['params']}{scaling}{nfeat}")

    model, metrics = evaluate_final(spec, train, test, spec.finalize(best), method=best["method"])
    _report(best, metrics)
    return best, metrics, model, features


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tune hyperparameters (and optionally features).")
    p.add_argument("model", nargs="?", default="xgb", choices=list(MODELS))
    p.add_argument("--select", action="store_true", help="jointly search the feature count (top-N)")
    p.add_argument("--write", action="store_true", help="write the result to config/best_params.yml")
    p.add_argument("--n-iter", type=int, default=25, help="random-search iterations")
    p.add_argument("--repeats", type=int, default=3, help="CV partitions pooled per config (robustness vs. time)")
    args = p.parse_args()

    spec = MODELS[args.model]
    # self-ranking models rank with the params being searched; gcn ranks via already-tuned xgb
    ranker, ranker_params = (None, None)
    if args.select and not spec.ranking:
        ranker, ranker_params = MODELS["xgb"], config.load("xgb")[0]

    dataset = Dataset.load(graph=spec.graph)
    print(f"Tuning {args.model}: {len(dataset.data)} samples, {len(dataset.feature_names)} features, "
          f"{args.repeats}x{5}-fold CV" + (" (+ joint feature selection)" if args.select else ""))

    best, metrics, _, features = tune(spec, dataset, n_iter=args.n_iter, n_repeats=args.repeats,
                                      select=args.select, ranker=ranker, ranker_params=ranker_params)

    entry = config.build_entry(args.model, best, features, metrics)
    print("\n# --- paste into config/best_params.yml ---")
    print(config.to_yaml(entry).rstrip())
    if args.write:
        config.write(entry)
        print(f"\nWrote {args.model} to {config.CONFIG_PATH}")
