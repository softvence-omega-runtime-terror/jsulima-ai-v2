import joblib
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages loading and using the trained lineup prediction model
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.preprocessing_info = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.model_path = Path(__file__).parent.parent.parent / "models" / "lineup_model.pkl"
        self.features_path = Path(__file__).parent.parent.parent / "models" / "feature_names.pkl"
        self.preprocessing_path = Path(__file__).parent.parent.parent / "models" / "preprocessing_info.pkl"
    
    def load_model(self) -> bool:
        """
        Load the trained model from disk
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
                
                # Load feature names if available
                if self.features_path.exists():
                    self.feature_names = joblib.load(self.features_path)
                    logger.info(f"Feature names loaded from {self.features_path}")
                
                # Load preprocessing info if available
                if self.preprocessing_path.exists():
                    self.preprocessing_info = joblib.load(self.preprocessing_path)
                    self.accuracy = self.preprocessing_info.get('accuracy')
                    self.precision = self.preprocessing_info.get('precision')
                    self.recall = self.preprocessing_info.get('recall')
                    self.f1_score = self.preprocessing_info.get('f1_score')
                    logger.info(f"Preprocessing info loaded. Accuracy: {self.accuracy:.4f}")
                
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                logger.info("Please save your trained model using: joblib.dump(model, 'models/lineup_model.pkl')")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, features_df: pd.DataFrame) -> list:
        """
        Make predictions using the loaded model
        
        Args:
            features_df: DataFrame with features for prediction
            
        Returns:
            list: Predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        try:
            predictions = self.model.predict(features_df)
            return predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_proba(self, features_df: pd.DataFrame) -> Optional[list]:
        """
        Get prediction probabilities (if supported by model)
        
        Args:
            features_df: DataFrame with features for prediction
            
        Returns:
            list: Prediction probabilities or None if not supported
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        try:
            proba = self.model.predict_proba(features_df)
            # Return probability of class 1 (starter)
            return proba[:, 1].tolist() if len(proba.shape) > 1 else proba.tolist()
        except Exception as e:
            logger.error(f"Error getting probabilities: {str(e)}")
            return None
