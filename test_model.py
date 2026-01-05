"""Test script to debug model loading"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    print("Step 1: Import classes...")
    from app.models.weighted_ensemble import WeightedEnsemble, XGBBoosterWrapper
    print("  OK")
    
    print("Step 2: Load model...")
    import joblib
    m = joblib.load('app/models/saved/ensemble_model_regularized.pkl')
    print(f"  Loaded: {type(m).__name__}")
    
    print("Step 3: Check model structure...")
    if hasattr(m, 'models_dict'):
        print(f"  models_dict keys: {list(m.models_dict.keys())}")
        for name, submodel in m.models_dict.items():
            print(f"    {name}: {type(submodel).__name__}")
    
    print("Step 4: Load feature columns...")
    import pickle
    with open('app/models/saved/nbafeature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print(f"  Features: {feature_columns}")
    
    print("Step 5: Try prediction with dummy data...")
    import numpy as np
    import pandas as pd
    X = pd.DataFrame([{col: 0.0 for col in feature_columns}])
    print(f"  Feature shape: {X.shape}")
    
    proba = m.predict_proba(X)
    print(f"  Prediction OK: {proba}")
    
except Exception as e:
    import traceback
    print(f"\nERROR at step: {str(e)}")
    traceback.print_exc()
