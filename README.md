# qsar-biodegradation

Predicting biodegradability (RB vs NRB) of chemicals from molecular descriptors and
structure. Binary classification on the ~1,050-molecule QSAR dataset from
[Mansouri et al., 2013](https://doi.org/10.1021/ci4000213), comparing three model classes:
xgb, logistic regression, and a graph neural network with a shared experiment pipeline.

Mostly an exploration project to use GNNs on a molecular dataset - has the beginnings of a 
reusable pipeline for my own ease when comparing to other models.

## Results

Current models are tuned at a **20-feature cap** (see *Features* below).
Comparison across the three models (over 10 sets of test data)

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| xgb | 0.767 ± 0.020 | 0.772 ± 0.045 | 0.766 ± 0.042 | 0.910 ± 0.013 | 0.863 ± 0.028 |
| logreg | 0.789 ± 0.028 | 0.733 ± 0.036 | 0.856 ± 0.040 | 0.923 ± 0.016 | 0.870 ± 0.027 |
| gcn | 0.769 ± 0.031 | 0.707 ± 0.045 | 0.846 ± 0.040 | 0.905 ± 0.024 | 0.836 ± 0.048 |


**External validation** on a 670-molecule held-out set (via `src/inference`): ensemble
(xgb+logreg) F1 0.769 / ROC-AUC 0.921; gcn F1 0.785 / ROC-AUC 0.926 — both match internal CV,
so the models generalise with no overfitting.

**Headline finding:** the task is saturated at F1 ≈ 0.78–0.79 / ROC-AUC ≈ 0.92. Adding
descriptors (RDKit, xTB electronic) or more features past ~15–20 does not move the ceiling;
it's data/label-limited, not feature-limited. At a 20-feature budget **logistic regression is
the model to beat** — cheapest, most stable, best aggregate.

## Research highlights

The full running log is in **[notes.md](notes.md)**:

- **More features =/= better** 
  Added dft feats including HOMO-LUMO gap (using dxtb) and 6 more electronic descriptors: IP/EA/Fukui/WBO/dispersion.
  Added 2d RDKit feats: logP/MolWt/TPSA, bringing total tabular features to 53. Although these features ranked well,
  they didn't improve metrics that much, and there was little change in performance capping at 20 feats (note xgb feature
  importance used for gcn, given ease of computation over the many folds)
- **Errors partly shared.**
  Across matched splits (i.e. same seed), 34% of misclassifications are made
  by just one model; 97 molecules (9.2%) are wrong for all three, split across
  both classes.
- **xgb + log reg ensemble**
  xgb + log reg (avg of probability preds to classify) is best out of all possible ensemble combos.
  However, GCN is more stable over the splits, and generalised better with the external data set.
- **out-of-sample testing**
  On a 670-molecule external set (same source) the pipelines match internal CV
  (no overfitting); the GCN is better than ensemble on the new data.

## Features

53 candidate features assembled per molecule, capped to the top 20 (leak-free per-fold ranking)
in the final configs:

- **41 qsar descriptors** — precomputed QSAR descriptors from the source dataset.
- **3 RDKit descriptors** — `MolLogP`, `MolWt`, `TPSA`
- **9 xTB electronic descriptors** — via dxtb GFN1-xTB
  (RDKit ETKDGv3 + MMFF for 3D): HOMO-LUMO gap, IP, EA, electrophilicity, Fukui±_max,
  Wiberg bond orders (total/max), dispersion energy. Cached in `data/xtb_features.csv` as these take longer to calc

Missing values are **dropped, never imputed** (already cleaned dataset; 1,055 → 1,050 after dropping
embedding/SCF failures). 

## Layout

```
src/
  de/          data engineering: dataset assembly, qsar + RDKit + xTB featurisation
  exp/         experiment pipeline: ModelSpec plugins, CV tuning, model comparison
  inference/   fit-on-startup inference pipelines (ensemble + gcn) and external validation
  utils/       metrics, random search
config/
  best_params.yml   tuned params + selected features per model (source of truth)
data/          biodeg.csv, smiles_data.csv, xtb_features.csv (cache), external val set
outputs/       comparison + inference results
notes.md       running research log
```

Models are registered in `src/exp/registry.py` (`xgb`, `logreg`, `gcn`), each a `ModelSpec`
plugin that fills in build/fit/predict over a shared `Dataset`. Generic drivers (`cv_score`,
`evaluate_final`, `fit_final`) consume the spec, so tabular-vs-graph differences stay inside
each plugin.

## Usage

Run everything from `src/` with it on the path:

```bash
cd src

# Tune a model (+ joint feature selection) and write to config/best_params.yml
python -m exp.tune xgb --select --write

# 10-seed holdout comparison at the tuned configs
python -m exp.compare --seeds 10

# External validation of both inference pipelines
python -m inference.validate
```

Programmatic inference (input is a CSV in the `smiles_data.csv` format — SMILES + the 41 qsar
columns; the qsar descriptors are **not** computable from a bare SMILES in this repo):

```python
from inference.pipeline import predict
df = predict("data/my_molecules.csv", pipeline="ensemble")   # or pipeline="gcn"
```

**Feature cap:** `CAP_FEATURES` in `src/exp/tune.py` pins the selected feature count under
`--select` (currently 20). Set it to `None` to restore the full count search.

## Setup

Python 3.14. Install dependencies:

```bash
pip install -r requirements.txt
```

**IMPORTANT**
> **Import order matters:** xgboost must load before torch or their OpenMP runtimes clash. All
> entrypoints set `OMP_NUM_THREADS=1` and import `exp.registry` (xgboost) before torch/dxtb.


**AI Use**
used Claude code for helping generate pipeline code (and readme!). notes.md documents process and findings.
