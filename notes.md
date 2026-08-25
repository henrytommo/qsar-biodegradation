# QSAR

## Overview
Predicting whether or not a chemical is able to be broken down by microbes (target feats ready biodegradability or not, RB vs NRB)

## Chem
- Depends on the microbes being used. but expect amides, esters, carbon organic bonds etc to be easier to be broken down. halogens or bulky rings probs slow this down, expect those features to be good indicators.
- likewise, smaller molecules easier to break down as there is less of them to biodegrade
- there will be physical propoerties too that will affect this

## considerations
- 41 feats, maybe not all are important. imagine some multicollinearity
- not a huge amount of data, so do some k-fold splits. make sure to do scaling within each split so it doesn't leak into other splits. also stratify as this is a class imbalanced dataset. then train on the full dataset

## feature selection
- looking at multicollinearity, correlation to target var, and feature importance of xgboost vs logistic regression. note - l1 norm means some features go to zero (if this is due to correlation with another pair, keep note of which has higher xgb feature importance)
- next step is a nn, so multicollinearity is less important than for the logistic regression model. however keep this analysis for now as when it comes to training the nn, may want to reduce # of features to avoid overfitting - the dataset size is still on the smaller side.

currently:

| Model | Test F1 | Test ROC-AUC | Hyperparameters |
| --- | --- | --- | --- |
| XGBoost | 0.824 | 0.946 | max_depth=5, learning_rate=0.2, subsample=0.6, colsample_bytree=0.8, min_child_weight=3, gamma=0.1 (chosen via random search, 5-fold CV F1 0.811 ± 0.030) |
| Logistic Regression | 0.785 | 0.946 | C=10, l1_ratio=0.5 (elastic-net), scaling=normalize, class_weight="balanced" (chosen via random search, 5-fold CV F1 0.800 ± 0.027) |

## GCN
- So far, we have used the qsar features. Now to try using the actual molecular structures - using SMILES representation (from Kamel Mansouri, Tine Ringsted, Davide Ballabio, Roberto Todeschini, Viviana Consonni; Quantitative Structure–Activity Relationship Models for Ready Biodegradability of Chemicals. J. Chem. Inf. Model. 22 April 2013; 53 (4): 867–878. https://doi.org/10.1021/ci4000213)
- use molecular structure PLUS qsar feats as global features

| Metric | CV (5-fold) | Held-out test |
| --- | --- | --- |
| F1 | 0.809 ± 0.039 | 0.797 |
| ROC-AUC | 0.931 ± 0.019 | 0.944 |
| PR-AUC | 0.874 ± 0.045 | 0.914 |

Best params: hidden_dim=32, n_gcn_layers=3, dropout=0.3, lr=0.01, weight_decay=0.001, batch_size=64.

- pretty similar to the xgb and lr experiments - so likely we need to go back to the features to see if there are any improvements to make there.

not a huge amount of data so diong over 10 random samples of the test data:
| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.790 ± 0.041 | 0.769 ± 0.058 | 0.815 ± 0.057 | 0.923 ± 0.022 | 0.879 ± 0.036 |
| LogReg | 0.799 ± 0.029 | 0.742 ± 0.035 | 0.868 ± 0.041 | 0.923 ± 0.017 | 0.868 ± 0.033 |
| GCN | 0.790 ± 0.031 | 0.719 ± 0.071 | 0.885 ± 0.050 | 0.924 ± 0.015 | 0.864 ± 0.027 |


## features, again
- now that we have chemical structures, we can check for presence of certain functional groups.
- additionally: this is a biodegradability prediction by microbes. the SMILES data contains stereoisomeric compounds, and chiral molecules will interact differently with different enzymes. this won't be picked up by the way im currently doing my gcn, so we can add stereisomer flags as an extra feature
    - or not, only about 1% out of 1000 molecules have chiral centres or stereo double bonds. tried out a GINEConv but worse results. pros won;t include this.
- next, adding a feature corresponding to homo-lumo gap to hopefully try and give an idication of how hard to break down these molecules. dxtb looks like the easiest, quickest estimate, so use that (along with rdkit to estimate 3d structure)


adding homo lumo gap and doing some feature removal - during cv search, search over feature ranking as well as hyperparams. set up a reusable experiment pipeline that saves best configs to yaml file - allows for manual selection of feats if warranted.
homo lumo gap: using dxtb python package - semi empirical. not too computationally expensive, but added a cache to avoid re-calculating on the same feats.
pretty similar across all metrics, now to add some more features now that pipeline is set up.

| Model | Config | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| xgb | 25 feats | 0.773 ± 0.015 | 0.734 | 0.820 | 0.912 | 0.866 |
| logreg | 25 feats | 0.792 ± 0.026 | 0.730 | 0.868 | 0.922 | 0.867 |
| gcn | 42 feats | 0.802 ± 0.025 | 0.747 | 0.869 | 0.928 | 0.871 |

## experimentation pipeline
missing feats? it's a curated dataset so have kind of skipped the cleaning step. but for homo-lumo gap failures, this happens 1/1055 times, so just drop this as will be hard to compute.

generate report for rare/low importance features

3 sweeps: feature selection, then hyperparam tuning, then model comparison.

getting an experiment process i was happy with has taken a few iterations (and there will probs be more as i go through), but this is a good place to start. next some more features.


## aside - adding more partitions
in tune, update so we take a 5 fold partition over three random seeds (rather than 5 over 1), so we get a better representation.


## rdkit features
there are loads of rdkit descriptors that can be added. will start with:
- octanol-water partition coef (Crippen.MolLogP): high val means hydrophobic, may affect biodegradation.
- molecular weight (Descriptors.MolWt): a larger molecule will be slower to cross bacterial membranes. however we have a few feats already that denote number of atoms and mass weighted/size correlated counts. so molecular weight will likely have some collinearity with these.
- topological polar surface area (Descriptors.TPSA): indication of hydrogen bonding ability - affects bioavailability. again, we have nHDon (h bond donors) as a feat already, so will see some multicollinearity.

checked overlap first: none of the three are directly present (no lipophilicity descriptor at all - MolWt/TPSA have related feats). added all three -> 45 feats total. cheap 2D descriptors so no cache (unlike the xTB gap). MolLogP ranks high in the tree ranking (used by xgb + gcn) but logreg doesn't pick it up (a linear model gets less from logP); TPSA selected by all three; MolWt selected by logreg + gcn (ranks lower, the flagged collinearity).

logreg improved a touch (it's the model that took MolWt+TPSA), xgb/gcn flat within noise. not a breakthrough.


## repeated CV outcome (3x partitions)
implemented as `tune --repeats N` (default 3): pool N 5-fold partitions per config instead of 1, so the winner is robust to fold assignment (3x cost). result: xgb + logreg configs unchanged (their single-split winners were already robust), gcn moved to a gentler config (h64/lr1e-3 vs h128/lr1e-2). old gcn config claimed test f1 0.816 at tune time but delivered 0.783 across seeds (0.033 mirage); new one claims 0.795, delivers 0.794. gap basically gone. tune numbers now mean something.

final 10-seed comparison at the 45-feat, 3x-repeated-CV configs:

| Model | Config | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| xgb | 45 feats | 0.766 ± 0.024 | 0.716 ± 0.034 | 0.825 ± 0.041 | 0.916 ± 0.019 | 0.872 ± 0.027 |
| logreg | 30 feats | 0.804 ± 0.026 | 0.741 ± 0.037 | 0.882 ± 0.036 | 0.923 ± 0.019 | 0.864 ± 0.039 |
| gcn | 45 feats | 0.794 ± 0.029 | 0.742 ± 0.054 | 0.858 ± 0.031 | 0.925 ± 0.015 | 0.870 ± 0.030 |

still tied within noise on every column. logreg best recall (0.882), gcn best precision/ROC-AUC/PR-AUC (better-calibrated ranking, more conservative), xgb trails.


## using dxtb for more dft feats

- ionisation potential, electron affinity: indicator for single electron transfer (ofen first step of breakdown)
- fukui function: nucleophilic and electrophilic attack
- wiberg bond order - bond strengths
- dispersion_energy

now up to 53 feats - so many

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| xgb | 0.779 ± 0.033 | 0.749 ± 0.028 | 0.813 ± 0.047 | 0.916 ± 0.019 | 0.876 ± 0.030 |
| logreg | 0.788 ± 0.019 | 0.729 ± 0.027 | 0.861 ± 0.037 | 0.924 ± 0.016 | 0.869 ± 0.033 |
| gcn | 0.789 ± 0.025 | 0.740 ± 0.050 | 0.848 ± 0.039 | 0.921 ± 0.020 | 0.862 ± 0.037 |
 still getting similar results, although wbo_max ranked 3rd by xgb.


## do the models fail on the same molecules? (error analysis)
ran all 3 models over 10 seeds where every seed shares ONE stratified test split, so each (molecule, seed) gets 3 matched right/wrong calls + probas.

errors are only partly shared. of 469 misclassified (mol, seed) instances: 43% wrong by all 3, 23% by 2, but 34% wrong by just ONE model. pairwise error corr (phi): xgb-logreg 0.61, xgb-gcn 0.65, logreg-gcn 0.77. so logreg and gcn make more of the same mistakes.

hard core: 97 molecules (9.2%) wrong by all 3 in >=50% of their appearances. label mix RB 34 / NRB 63 = 35% RB, same as the dataset - not one-class problem.


## ensembling
soft-vote (average the probas). beats every single model on every metric, matched splits:

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| xgb | 0.779 ± 0.033 | 0.749 | 0.813 | 0.916 | 0.876 |
| logreg | 0.788 ± 0.019 | 0.729 | 0.861 | 0.924 | 0.869 |
| gcn | 0.789 ± 0.025 | 0.740 | 0.848 | 0.921 | 0.862 |
| xgb+logreg | 0.797 ± 0.028 | 0.759 | 0.841 | 0.928 | 0.887 |
| all three | 0.799 ± 0.012 | 0.748 | 0.858 | 0.929 | 0.888 |

gcn is droppable: xgb+logreg - all three = -0.001 ± 0.019 f1 (tied 5/10 seeds), and the 2-way has the best precision of anything tested (0.759). gcn adds ~nothing to the mean.
BUT the only thing the 3 model ensemble does is less seed-to-seed variance (std 0.012 vs 0.028) - might generalise better.


## inference + external validation
built two inference pipelines in src/inference: `ensemble` (xgb+logreg soft-vote) and `gcn`. fit-on-startup (no saved model files) - each fits once on the full training set at its config via a new `exp.fit_final` (the exact fit the reported metrics came from, factored out of evaluate_final), then scores new molecules by `Dataset.from_csv(...).subset(feats).transform(scaler)`. public api: `predict(csv, pipeline=...)` (row-aligned, abstains where xtb featurisation fails); `python -m inference.validate` runs the external set.

inference input is a csv in the smiles_data format (smiles + the 41 qsar cols + optional label), not a bare smiles. external-validation set (670 molecules, 191 RB / 28.5%).

external validation (670 molecules, 0 abstained - external set computed no errors):

| Pipeline | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| ensemble (xgb+logreg) | 0.769 | 0.763 | 0.775 | 0.921 | 0.814 |
| gcn | 0.785 | 0.796 | 0.775 | 0.926 | 0.825 |

both generalise - external f1 ~0.77-0.79 and roc-auc ~0.92 basically match internal 10-seed cv, so no overfitting, models hold up on new molecs. lower pr-auc (~0.82) is just the lower RB prevalence out here (28.5% vs 34% in train).

on the external set gcn slightly beats the ensemble on EVERY metric (f1 0.785 vs 0.769, precision 0.796 vs 0.763). small gap, but the looks like gcn generalises a bit better to new chemistry than internal cv suggested (probs learning something from the 3d structure graphs)


## capping features at 20
53 features is way too many - adding new feats doesn't have a corresponding improvement, so will try capping. reduces complexity and maybe overfitting. note that gcn feat importances are taken from xgb feature importances - could maybe look at GNNExplainer in pytorch geometric BUT it's not as quick as xgb feat importance and would need to be run for each fold.. also better for looking at explainability for one molecule rather than an overall feature importance.
added a CAP_FEATURES const to exp/tune.py: when set (=20), --select pins n_features to it instead of searching the count grid, so the written config keeps exactly 20 feats. re-tuned all 3 with --select --write (order xgb -> logreg -> gcn, so gcn ranks against the freshly-tuned xgb). config/best_params.yml now holds 20 feats per model.

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| xgb | 0.767 ± 0.020 | 0.772 ± 0.045 | 0.766 ± 0.042 | 0.910 ± 0.013 | 0.863 ± 0.028 |
| logreg | 0.789 ± 0.028 | 0.733 ± 0.036 | 0.856 ± 0.040 | 0.923 ± 0.016 | 0.870 ± 0.027 |
| gcn | 0.769 ± 0.031 | 0.707 ± 0.045 | 0.846 ± 0.040 | 0.905 ± 0.024 | 0.836 ± 0.048 |


basically no loss from dropping 53 -> 20 feats.

logreg flipped to normalize scaling and both linear+graph went full l1 (l1_ratio 1.0). xgb & gcn share an identical 20-feat list (gcn ranks via xgb); ip / wbo_max / fukui±_max / homo_lumo_gap all survive.






