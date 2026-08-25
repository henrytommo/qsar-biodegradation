"""Soft-vote ensemble over tabular models: average their P(RB). Default xgb+logreg - error
analysis found this matches the 3-way ensemble's mean while dropping the GCN's graph pipeline."""
import numpy as np

from de.dataset import Dataset
from inference.predictor import Predictor


class EnsemblePredictor:
    """Average the member probabilities. Members share one loaded training Dataset (fit once each);
    each subsets it to its own config features, so differing feature sets are handled per member."""

    def __init__(self, names=("xgb", "logreg"), weights=None, train_ds=None):
        base = train_ds if train_ds is not None else Dataset.load(log=lambda *a: None)
        self.members = [Predictor(n, train_ds=base) for n in names]
        self.weights = weights   # None -> equal weight

    def predict_proba(self, ds):
        probas = np.array([m.predict_proba(ds) for m in self.members])
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, ds, threshold=0.5):
        proba = self.predict_proba(ds)
        return proba, (proba >= threshold).astype(int)
