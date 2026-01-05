"""Test script to debug prediction errors"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from app.services.Basketball.basketball_predictor import basketball_predictor
    print("Predictor loaded:", basketball_predictor._loaded)
    print("Error log:", basketball_predictor._error_log)
    
    if basketball_predictor._loaded:
        result = basketball_predictor.predict_game('1066', '1067', '2026-01-05')
        print("\nResult:", result)
    else:
        print("Model not loaded!")
        print("Errors:", basketball_predictor.get_error_details())
except Exception as e:
    import traceback
    print("ERROR:", str(e))
    traceback.print_exc()
