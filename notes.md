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

checked overlap first: none of the three are directly present (no lipophilicity descriptor at all - MolWt/TPSA have related-but-distinct feats as expected). added all three -> 45 feats total. cheap 2D descriptors so no cache (unlike the xTB gap). MolLogP ranks high in the tree ranking (used by xgb + gcn) but logreg doesn't pick it up (a linear model gets less from logP); TPSA selected by all three; MolWt selected by logreg + gcn (ranks lower, the flagged collinearity).

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



