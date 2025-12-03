
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.predictor import UFCPredictor

def test_prediction():
    print("Testing UFC Predictor with raw stats...")
    
    # Sample raw stats (no engineered features)
    raw_stats = {
        "local_strikes_total_head": 85,
        "local_strikes_total_body": 12,
        "local_strikes_total_leg": 8,
        "local_strikes_power_head": 35,
        "local_strikes_power_body": 8,
        "local_strikes_power_leg": 6,
        "local_takedowns_att": 3,
        "local_takedowns_landed": 2,
        "local_submissions": 0,
        "local_knockdowns": 1,
        "local_control_time": "2:00",  # String format
        
        "away_strikes_total_head": 70,
        "away_strikes_total_body": 10,
        "away_strikes_total_leg": 6,
        "away_strikes_power_head": 25,
        "away_strikes_power_body": 7,
        "away_strikes_power_leg": 5,
        "away_takedowns_att": 2,
        "away_takedowns_landed": 1,
        "away_submissions": 0,
        "away_knockdowns": 0,
        "away_control_time": "1:00"   # String format
    }
    
    try:
        predictor = UFCPredictor()
        result = predictor.predict_single(raw_stats)
        print("\nPrediction Result:")
        print(result)
        print("\n[SUCCESS] Prediction successful with raw stats!")
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
