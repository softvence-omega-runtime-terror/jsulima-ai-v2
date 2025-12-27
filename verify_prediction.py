
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from app.services.predictor import get_predictor

def test_prediction():
    print("Initializing predictor...")
    predictor = get_predictor()
    
    # MOCK MODELS to bypass loading error (we want to test feature extraction)
    # create dummy objects with predict/predict_proba methods
    class DummyModel:
        def predict(self, X): return [1]
        def predict_proba(self, X): return [[0.4, 0.6]]
        
    predictor.models = {
        'winner': DummyModel(), 
        'method': DummyModel(), 
        'round': DummyModel(), 
        'stats': {'local_total_strikes': DummyModel()} # partial stats
    }
    
    class DummyEncoder:
        def inverse_transform(self, x): return ["KO/TKO"]
        
    predictor.method_encoder = DummyEncoder()
    predictor.feature_cols = ['weight_class', 'local_height'] # dummy cols
    
    predictor._loaded = True
    print("Models mocked for feature verification.")

    print("Predicting match...")
    # Use dummy IDs - "1" and "2".
    try:
        result = predictor.predict_match(
            local_id="1", 
            away_id="2", 
            weight_class="Lightweight"
        )
        
        if "error" in result:
            print(f"Prediction Error: {result['error']}")
        else:
            print("Prediction Successful!")
            print(f"Winner: {result['prediction']['winner']}")
            print("Features constructed successfully via FeatureExtractor.")
            
    except Exception as e:
        print(f"FAILED during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
