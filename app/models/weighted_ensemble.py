
# app/models/weighted_ensemble.py
# ============================================================
# EXACT copy of the training-time WeightedEnsemble (from your code),
# with minimal additions so that it's importable and pickle-safe.
# ============================================================

from typing import Dict, Any
import numpy as np
import sys
import types

# Import BaseEstimator and ClassifierMixin from sklearn
from sklearn.base import BaseEstimator, ClassifierMixin

# Monkey patch CatBoostClassifier to add __sklearn_tags__ if missing
try:
    import catboost
    if hasattr(catboost, 'CatBoostClassifier'):
        from sklearn.utils._tags import Tags, TargetTags
        if not hasattr(catboost.CatBoostClassifier, '__sklearn_tags__'):
            def __sklearn_tags__(self):
                return Tags(
                    estimator_type="classifier",
                    target_tags=TargetTags(required=True, one_d_labels=False, two_d_labels=False, positive_only=False, multi_output=False, single_output=True)
                )
            catboost.CatBoostClassifier.__sklearn_tags__ = __sklearn_tags__
            print("✓ Added __sklearn_tags__ to CatBoostClassifier")
except ImportError:
    pass

# Try to import xgboost for XGBBoosterWrapper
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None


class XGBBoosterWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for XGBoost Booster objects to provide sklearn-compatible predict_proba.
    Required for loading pickled models that contain XGBoost boosters.
    Inherits from BaseEstimator and ClassifierMixin for sklearn 1.6+ compatibility.
    """
    # Class attribute for sklearn estimator type detection
    _estimator_type = "classifier"
    
    def __init__(self, booster=None, n_classes=2, **kwargs):
        self.booster = booster
        self.n_classes = n_classes
        self._classes_ = np.array(list(range(n_classes)), dtype=int)
        self._estimator_type = "classifier"  # Explicitly set estimator type
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __sklearn_tags__(self):
        """Return sklearn tags for this estimator"""
        from sklearn.utils._tags import Tags, TargetTags
        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True, one_d_labels=False, two_d_labels=False, positive_only=False, multi_output=False, single_output=True)
        )
    
    def fit(self, X, y):
        """Dummy fit method for sklearn compatibility"""
        return self
    
    def _get_booster(self):
        candidates = ['booster', '_booster', 'model', '_model', 'xgb_model', 'estimator', '_Booster']
        for attr in candidates:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None:
                    return val
        if hasattr(self, 'predict'):
            return self
        return None
    
    def fit(self, X, y=None):
        """Dummy fit for sklearn compatibility."""
        return self
    
    def predict_proba(self, X):
        booster = self._get_booster()
        if booster is None:
            raise ValueError(f"Booster not found. Available: {list(self.__dict__.keys())}")
        
        if hasattr(booster, 'predict_proba') and booster is not self:
            return booster.predict_proba(X)
        
        try:
            if HAS_XGB and not isinstance(X, xgb.DMatrix):
                dmatrix = xgb.DMatrix(X)
            else:
                dmatrix = X
            preds = booster.predict(dmatrix)
            if len(preds.shape) == 1:
                return np.column_stack([1.0 - preds, preds])
            return preds
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")
    
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)
    
    @property
    def classes_(self):
        if hasattr(self, '_classes_'):
            return self._classes_
        return np.array([0, 1], dtype=int)
    
    def __sklearn_is_fitted__(self):
        """Check if the estimator is fitted."""
        return hasattr(self, 'booster') and self.booster is not None
    
    def __sklearn_tags__(self):
        """Return sklearn tags for compatibility with sklearn 1.6+."""
        # Import Tags to manually construct with correct estimator_type
        from sklearn.utils._tags import Tags, TargetTags, InputTags
        
        # Create a Tags object with classifier as the estimator type
        # This ensures sklearn's is_classifier() will work correctly
        tags = Tags(
            estimator_type="classifier",
            target_tags=TargetTags(
                required=False,
                one_d_labels=False,
                two_d_labels=False,
                positive_only=False,
                multi_output=False,
                single_output=True,
            ),
            transformer_tags=None,
            classifier_tags=None,
            regressor_tags=None,
            array_api_support=False,
            no_validation=False,
            non_deterministic=False,
            requires_fit=True,
            _skip_test=False,
            input_tags=InputTags(
                one_d_array=False,
                two_d_array=True,
                three_d_array=False,
                sparse=False,
                categorical=False,
                string=False,
                dict=False,
                positive_only=False,
                allow_nan=False,
                pairwise=False,
            ),
        )
        return tags

class WeightedEnsemble(BaseEstimator, ClassifierMixin):
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
        self._estimator_type = "classifier"  # Explicitly set estimator type
    
    def __sklearn_tags__(self):
        """Return sklearn tags for this estimator"""
        from sklearn.utils._tags import Tags, TargetTags
        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True, one_d_labels=False, two_d_labels=False, positive_only=False, multi_output=False, single_output=True)
        )
    
    def fit(self, X, y):
        """Dummy fit method for sklearn compatibility"""
        return self

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


# ============================================================
# Register classes in __main__ for pickle compatibility
# ============================================================
try:
    if '__main__' not in sys.modules:
        sys.modules['__main__'] = types.ModuleType('__main__')
    sys.modules['__main__'].WeightedEnsemble = WeightedEnsemble
    sys.modules['__main__'].XGBBoosterWrapper = XGBBoosterWrapper
except Exception as e:
    print(f"Warning: Could not register classes in __main__: {e}")
