"""Fit-on-startup single-model predictor. Fits one model + scaler once on the full training set
at its tuned config (params / feature subset / scaling), then scores new molecules. Reuses
exp.fit_final (the exact fitting the reported metrics used) and Dataset.subset/transform, so a
deployed predictor is the same model that was evaluated."""
from exp.registry import MODELS   # imports xgboost before torch (order-safe)

from de.dataset import Dataset
from exp import config
from exp.spec import fit_final


class Predictor:
    """One model fit once on the full training data at its config. `.predict_proba(ds)` scores a
    featurised inference Dataset (subset to the model's features, scaled with the train scaler)."""

    def __init__(self, name, train_ds=None, random_state=1):
        self.name = name
        self.spec = MODELS[name]
        self.params, self.method, self.features = config.load(name)
        base = train_ds if train_ds is not None else Dataset.load(graph=self.spec.graph, log=lambda *a: None)
        train = base.subset(self.features) if self.features else base
        self.model, self.scaler, _ = fit_final(self.spec, train, self.params, self.method, random_state)

    def predict_proba(self, ds):
        """P(RB) per molecule in ds, aligned to ds's row order."""
        sub = ds.subset(self.features) if self.features else ds
        _, _, proba = self.spec.predict(self.model, sub.transform(self.scaler))
        return proba

    def predict(self, ds, threshold=0.5):
        """(proba, label) with label = 1 (RB) where proba >= threshold."""
        proba = self.predict_proba(ds)
        return proba, (proba >= threshold).astype(int)
