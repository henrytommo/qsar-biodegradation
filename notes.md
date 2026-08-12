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

┌────────────────┬────────┬───────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐
│     Model      │ Test   │   Test    │                                       Hyperparameters                                        │
│                │   F1   │  ROC-AUC  │                                                                                              │
├────────────────┼────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ XGBoost        │ 0.824  │ 0.946     │ max_depth=5, learning_rate=0.2, subsample=0.6, colsample_bytree=0.8, min_child_weight=3,     │
│                │        │           │ gamma=0.1 (chosen via random search, 5-fold CV F1 0.811 ± 0.030)                             │
├────────────────┼────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ Logistic       │ 0.785  │ 0.946     │ C=10, l1_ratio=0.5 (elastic-net), scaling=normalize, class_weight="balanced" (chosen via     │
│ Regression     │        │           │ random search, 5-fold CV F1 0.800 ± 0.027)                                                   |
└────────────────┴────────┴───────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘