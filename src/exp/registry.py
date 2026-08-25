"""The model registry: maps a name to its ModelSpec so drivers and analysis scripts can run
any model by name without importing its module directly. Import order matters -- xgboost
must load before torch (pulled in via de.dataset / the GCN plugin) or their OpenMP runtimes
segfault; importing the XGBoost plugin first enforces that. Set OMP_NUM_THREADS=1 in the
entrypoint before importing this (see exp/tune.py).
"""
from exp.xgb_experiment import SPEC as XGB        # imports xgboost -- keep first
from exp.logreg_experiment import SPEC as LOGREG
from exp.gcn_experiment import SPEC as GCN        # imports torch

MODELS = {spec.name: spec for spec in (XGB, LOGREG, GCN)}
