"""Simple model load test"""
import sys
import warnings
warnings.filterwarnings('ignore')

# Setup encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("Loading classes...")
from app.models.weighted_ensemble import WeightedEnsemble, XGBBoosterWrapper
import joblib

print("Loading model...")
m = joblib.load('app/models/saved/ensemble_model_regularized.pkl')
print(f"Model type: {type(m).__name__}")

if hasattr(m, 'models_dict'):
    print(f"Sub-models: {list(m.models_dict.keys())}")
    
    for name, submodel in m.models_dict.items():
        print(f"\nTesting {name} ({type(submodel).__name__})...")
        try:
            import numpy as np
            import pandas as pd
            X_test = np.zeros((1, 12))  # 12 features
            
            if hasattr(submodel, 'predict_proba'):
                p = submodel.predict_proba(X_test)
                print(f"  OK: {p}")
            else:
                print(f"  No predict_proba method")
        except Exception as e:
            print(f"  FAILED: {str(e)[:100]}")
