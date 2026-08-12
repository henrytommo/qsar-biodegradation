import random


def random_search(scorer, param_distributions, n_iter=25, methods=("standardize",), random_state=1, **fixed_kwargs):
    """Sample n_iter random hyperparameter combos per method, score each via
    scorer(params, method, random_state, **fixed_kwargs) -> dict with an 'f1_mean' key,
    return results sorted best-F1-first."""
    rng = random.Random(random_state)
    results = []
    for method in methods:
        for _ in range(n_iter):
            params = {k: rng.choice(v) for k, v in param_distributions.items()}
            results.append(scorer(params, method, random_state, **fixed_kwargs))
    return sorted(results, key=lambda r: r["f1_mean"], reverse=True)
