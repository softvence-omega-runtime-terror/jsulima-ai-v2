"""Test predict_proba with proper error handling"""
import sys
import warnings
import traceback
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("Loading...")
from app.models.weighted_ensemble import WeightedEnsemble, XGBBoosterWrapper
import joblib
import numpy as np
import pandas as pd
import pickle

# Load model
m = joblib.load('app/models/saved/ensemble_model_regularized.pkl')
print(f"Model: {type(m).__name__}")

# Load feature columns
with open('app/models/saved/nbafeature_columns.pkl', 'rb') as f:
    feat_cols = pickle.load(f)
print(f"Features: {feat_cols}")

# Create test data
X = pd.DataFrame([{col: 0.5 for col in feat_cols}])
print(f"X shape: {X.shape}")

# Test scaler
print(f"\nScaler type: {type(m.scaler).__name__}")
try:
    X_scaled = m.scaler.transform(X)
    print(f"Scaled OK: {X_scaled.shape}")
except Exception as e:
    print(f"Scaler FAILED: {e}")

# Test each sub-model
print("\n--- Testing sub-models ---")
for name, submodel in m.models_dict.items():
    print(f"\n{name}:")
    try:
        p = submodel.predict_proba(X)
        print(f"  OK: {p[0]}")
    except Exception as e:
        print(f"  FAILED: {str(e)[:200]}")
        traceback.print_exc()

print("\n--- Full predict_proba ---")
try:
    proba = m.predict_proba(X)
    print(f"OK: {proba}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
