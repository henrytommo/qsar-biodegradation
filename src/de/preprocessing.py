from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALERS = {
    "normalize": MinMaxScaler,
    "standardize": StandardScaler,
}


def transform_features(X, scaler, binary_feats):
    """scaler, exclude binary_feats (categorical)"""
    scale_feats = [c for c in X.columns if c not in binary_feats]
    X_scaled = X.copy()
    X_scaled[scale_feats] = scaler.transform(X[scale_feats])
    return X_scaled


def scale_features(X_train, X_val, binary_feats, method="standardize"):
    if method not in SCALERS:
        raise ValueError(f"method must be one of {list(SCALERS)}, got {method!r}")

    scale_feats = [c for c in X_train.columns if c not in binary_feats]
    scaler = SCALERS[method]()

    X_train_scaled = X_train.copy()
    X_train_scaled[scale_feats] = scaler.fit_transform(X_train[scale_feats])

    return X_train_scaled, transform_features(X_val, scaler, binary_feats), scaler
