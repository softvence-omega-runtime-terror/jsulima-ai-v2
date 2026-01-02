"""
Helper script to save your trained model in the correct format for the FastAPI app
"""

import joblib
from pathlib import Path

def save_model(model, feature_names=None):
    """
    Save your trained model for use with the FastAPI application
    
    Usage:
        from model_save_helper import save_model
        import joblib
        
        # Load your trained model from notebook
        model = joblib.load('your_model.pkl')  # or however you have it
        feature_names = ['home_team_avg_points', 'away_team_avg_points', ...]
        
        save_model(model, feature_names)
    """
    
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "lineup_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names if provided
    if feature_names:
        features_path = models_dir / "feature_names.pkl"
        joblib.dump(feature_names, features_path)
        print(f"Feature names saved to {features_path}")

if __name__ == "__main__":
    print("Use this module to save your trained model:")
    print("from app.model_save_helper import save_model")
    print("save_model(your_model, your_feature_names)")
