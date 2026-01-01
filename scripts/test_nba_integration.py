
import sys
import time
from pathlib import Path
from pprint import pprint

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

try:
    from app.routes.NBA.nba_service import predict_upcoming_from_goalserve
except ImportError as e:
    print(f"Error importing service: {e}")
    sys.exit(1)

from unittest.mock import patch, MagicMock

# Mock data sample
MOCK_SCHEDULE = {
    "shedules": {
        "matches": [
            {
                "date": "01.01.2025",
                "match": [
                    {
                        "id": "123",
                        "hometeam": {"id": "1610612737", "name": "Atlanta Hawks"},
                        "awayteam": {"id": "1610612738", "name": "Boston Celtics"},
                        "formatted_date": "01.01.2025",
                        "status": "19:00",
                        "venue_name": "State Farm Arena"
                    }
                ]
            }
        ]
    }
}

def test_nba_prediction():
    print("--- Starting NBA Prediction Verification ---")
    
    # Mocking requests.get to avoid external dependency failure
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_SCHEDULE
        mock_get.return_value = mock_resp
        
        start_time = time.time()
        print("1. First Call (Cold Cache, Mocked API)...")
        try:
            predictions = predict_upcoming_from_goalserve(limit=2)
            elapsed = time.time() - start_time
            print(f"   Success! Time taken: {elapsed:.2f} seconds")
            print(f"   Number of predictions: {len(predictions)}")
            if predictions:
                pprint(predictions[0]['game_overview'])
                pprint(predictions[0]['prediction'])
        except Exception as e:
            print(f"   Failed: {e}")
            import traceback
            traceback.print_exc()
            return

        print("\n2. Second Call (Warm Cache, Mocked API)...")
        start_time = time.time()
        try:
            predictions_2 = predict_upcoming_from_goalserve(limit=2)
            elapsed = time.time() - start_time
            print(f"   Success! Time taken: {elapsed:.4f} seconds")
            if elapsed > 1.0:
                print("   WARNING: Cache might not be working as expected (took > 1s).")
            else:
                print("   Cache appears to be working.")
        except Exception as e:
            print(f"   Failed: {e}")

if __name__ == "__main__":
    test_nba_prediction()
