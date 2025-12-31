
# app/models/weighted_ensemble.py
# ============================================================
# EXACT copy of the training-time WeightedEnsemble (from your code),
# with minimal additions so that it's importable and pickle-safe.
# ============================================================

from typing import Dict, Any
import numpy as np

class WeightedEnsemble:
    def __init__(self, models_dict: Dict[str, Any], scaler: Any, weights: Dict[str, float] = None):
        """
        models_dict: dict of sub-models (e.g. {'lr': lr_model, 'rf': rf_model, ...})
        scaler: StandardScaler fitted on training features (used internally for LR only)
        weights: dict of per-model weights; if None, defaults to 1.0 for each
        """
        self.models_dict = models_dict
        self.scaler = scaler
        self.weights = weights or {}
        # Optional: expose classes_ to play nice with sklearn-like API
        self._classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, X):
        """
        Returns array of shape [n_samples, 2] for classes [0, 1].
        For 'lr' model, uses scaled features internally.
        Other models use raw features (same as your training code).
        """
        # Scale once for LR internally (avoids external/double scaling)
        X_scaled = self.scaler.transform(X)

        probas = []
        wts = []
        for name, model in self.models_dict.items():
            if name == 'lr':
                p = model.predict_proba(X_scaled)[:, 1]
            else:
                p = model.predict_proba(X)[:, 1]
            probas.append(p)
            wts.append(self.weights.get(name, 1.0))

        probas = np.vstack(probas)  # [n_models, n_samples]
        wts = np.array(wts, dtype=float)
        wts = wts / wts.sum() if wts.sum() != 0 else np.ones_like(wts) / len(wts)

        ens = np.average(probas, axis=0, weights=wts)  # [n_samples]
        return np.column_stack([1.0 - ens, ens])

    def predict(self, X, thr: float = 0.5):
        """
        Thresholded prediction on class-1 probability.
        """
        return (self.predict_proba(X)[:, 1] > thr).astype(int)

    @property
    def classes_(self):
        return self._classes_
