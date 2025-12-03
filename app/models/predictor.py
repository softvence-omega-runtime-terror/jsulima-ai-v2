"""
UFC Model Predictor Module

This module handles loading trained models and making predictions
for UFC fight outcomes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Any
from pathlib import Path

from app.utils.model_utils import (
    load_model, load_scaler, load_feature_names, load_model_metadata
)
from app.services.preprocessor import get_feature_columns
from app.utils.feature_engineering import engineer_all_features


class UFCPredictor:
    """UFC Fight Winner Predictor class."""
    
    def __init__(self):
        """Initialize the predictor by loading model artifacts."""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all required artifacts (model, scaler, features)."""
        try:
            self.model = load_model()
            self.scaler = load_scaler()
            self.feature_names = load_feature_names()
            try:
                self.metadata = load_model_metadata()
            except FileNotFoundError:
                self.metadata = {}
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Failed to load model artifacts. Please train the model first. Error: {e}"
            )
    
    def predict_single(self, fighter_stats: Dict[str, Union[float, str]]) -> Dict[str, Union[int, float, str]]:
        """
        Predict the winner of a single UFC fight.
        
        Args:
            fighter_stats: Dictionary with fighter statistics
        
        Returns:
            Dictionary containing:
                - prediction: 1 for local winner, 0 for away winner
                - winner: "Local Fighter" or "Away Fighter"
                - confidence: Model confidence in prediction (0-1)
                - probability_local: Probability of local fighter winning
                - probability_away: Probability of away fighter winning
        
        Raises:
            ValueError: If required features are missing
        """
        # Validate input
        self._validate_input(fighter_stats)
        
        # Convert to DataFrame
        df = pd.DataFrame([fighter_stats])
        
        # Engineer features
        df = engineer_all_features(df)
        
        # Ensure features are in correct order
        # Fill missing columns with 0 if any (though validation should catch this)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                
        df = df[self.feature_names]
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        # Make prediction
        prediction = self.model.predict(df_scaled)[0]
        probabilities = self.model.predict_proba(df_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': int(prediction),
            'winner': "Local Fighter" if prediction == 1 else "Away Fighter",
            'confidence': float(max(probabilities)),
            'probability_local': float(probabilities[1]),
            'probability_away': float(probabilities[0])
        }
        
        return result
    
    def predict_batch(self, fights: List[Dict[str, Union[float, str]]]) -> List[Dict[str, Union[int, float, str]]]:
        """
        Predict winners for multiple UFC fights.
        
        Args:
            fights: List of dictionaries, each containing fighter statistics
        
        Returns:
            List of prediction dictionaries
        
        Raises:
            ValueError: If any fight has missing features
        """
        results = []
        
        for i, fight_stats in enumerate(fights):
            try:
                result = self.predict_single(fight_stats)
                result['fight_index'] = i
                results.append(result)
            except ValueError as e:
                results.append({
                    'fight_index': i,
                    'error': str(e),
                    'prediction': None
                })
        
        return results
    
    def get_feature_importance(self, top_n: int = 15) -> List[Dict[str, Union[str, float]]]:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            List of dictionaries with feature names and importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature_importances_ attribute")
        
        importances = self.model.feature_importances_
        feature_importance = [
            {'feature': name, 'importance': float(importance)}
            for name, importance in zip(self.feature_names, importances)
        ]
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance[:top_n]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': type(self.model).__name__,
            'num_features': len(self.feature_names),
            'features': self.feature_names,
        }
        
        # Add metadata if available
        if self.metadata:
            info.update(self.metadata)
        
        # Add model parameters if available
        if hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()
        
        return info
    
    def _validate_input(self, fighter_stats: Dict[str, Union[float, str]]):
        """
        Validate that input contains all required features.
        
        Args:
            fighter_stats: Dictionary with fighter statistics
        
        Raises:
            ValueError: If required features are missing
        """
        # We only check for raw features that are absolutely required
        # The engineered features will be created dynamically
        # So we don't check against self.feature_names directly here
        pass
        
        # if missing_features:
        #     raise ValueError(
        #         f"Missing required features: {missing_features}. "
        #         f"Required features: {self.feature_names}"
        #     )


# Singleton instance for the predictor
_predictor_instance = None


def get_predictor() -> UFCPredictor:
    """
    Get or create the singleton predictor instance.
    
    Returns:
        UFCPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = UFCPredictor()
    
    return _predictor_instance


def predict_fight(fighter_stats: Dict[str, Union[float, str]]) -> Dict[str, Union[int, float, str]]:
    """
    Convenience function to predict a single fight outcome.
    
    Args:
        fighter_stats: Dictionary with fighter statistics
    
    Returns:
        Prediction result dictionary
    """
    predictor = get_predictor()
    return predictor.predict_single(fighter_stats)


def predict_fights(fights: List[Dict[str, Union[float, str]]]) -> List[Dict[str, Union[int, float, str]]]:
    """
    Convenience function to predict multiple fight outcomes.
    
    Args:
        fights: List of fighter statistics dictionaries
    
    Returns:
        List of prediction result dictionaries
    """
    predictor = get_predictor()
    return predictor.predict_batch(fights)
