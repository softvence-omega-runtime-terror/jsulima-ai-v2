"""Simple model load test"""
import sys
import warnings
import traceback
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("Loading...")
from app.models.weighted_ensemble import WeightedEnsemble, XGBBoosterWrapper
import joblib
import numpy as np

m = joblib.load('app/models/saved/ensemble_model_regularized.pkl')
print(f"Model: {type(m).__name__}")
print(f"Has scaler: {hasattr(m, 'scaler')}")
print(f"Scaler type: {type(m.scaler).__name__ if hasattr(m, 'scaler') else 'N/A'}")

if hasattr(m, 'models_dict'):
    for name, submodel in m.models_dict.items():
        print(f"\n{name}: {type(submodel).__name__}")
        # Check for CalibratedClassifierCV which might have issues
        if hasattr(submodel, 'calibrated_classifiers_'):
            print(f"  calibrated_classifiers_: {len(submodel.calibrated_classifiers_)}")
            for i, cc in enumerate(submodel.calibrated_classifiers_):
                print(f"    [{i}]: {type(cc).__name__}")
                if hasattr(cc, 'estimator'):
                    print(f"         estimator: {type(cc.estimator).__name__}")
                if hasattr(cc, 'base_estimator'):
                    print(f"         base_estimator: {type(cc.base_estimator).__name__}")
