import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y_true, preds, proba):
    return {
        "f1": f1_score(y_true, preds),
        # zero_division=0: all-negative predictions score 0 precision instead of warning
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def aggregate_metrics(metric_dicts):
    out = {}
    for key in metric_dicts[0]:
        values = [d[key] for d in metric_dicts]
        out[f"{key}_mean"] = np.mean(values)
        out[f"{key}_std"] = np.std(values)
    return out
