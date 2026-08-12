from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from de.preprocessing import scale_features


@dataclass
class Fold:
    fold: int
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    scaler: object


def holdout_split(X, y, test_size=0.2, random_state=1):
    """stratified train/test split"""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def cv_folds(X, y, binary_feats, method="standardize", n_splits=5, random_state=1):
    """reuse the same (X, y, method, n_splits, random_state) across models being compared
    so every model sees identical folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val, binary_feats, method)

        yield Fold(
            fold=i,
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            y_train=y_train,
            y_val=y_val,
            scaler=scaler,
        )
