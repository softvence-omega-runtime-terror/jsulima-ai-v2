"""
Model Persistence Utilities

This module provides functions for saving and loading ML models and preprocessing objects.
"""

import pickle
from pathlib import Path
from typing import Any, List
from sklearn.preprocessing import StandardScaler


def get_artifacts_dir() -> Path:
    """
    Get the artifacts directory path, creating it if it doesn't exist.
    
    Returns:
        Path to artifacts directory
    """
    project_root = Path(__file__).parent.parent.parent
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def save_model(model: Any, filename: str = "ufc_winner_prediction_model.pkl") -> Path:
    """
    Save a trained model to disk using pickle.
    
    Args:
        model: Trained scikit-learn model
        filename: Name of the file to save
    
    Returns:
        Path where model was saved
    """
    artifacts_dir = get_artifacts_dir()
    model_path = artifacts_dir / filename
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path


def load_model(filename: str = "ufc_winner_prediction_model.pkl") -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filename: Name of the model file to load
    
    Returns:
        Loaded scikit-learn model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    artifacts_dir = get_artifacts_dir()
    model_path = artifacts_dir / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def save_scaler(scaler: StandardScaler, filename: str = "scaler.pkl") -> Path:
    """
    Save a StandardScaler to disk.
    
    Args:
        scaler: Fitted StandardScaler object
        filename: Name of the file to save
    
    Returns:
        Path where scaler was saved
    """
    artifacts_dir = get_artifacts_dir()
    scaler_path = artifacts_dir / filename
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return scaler_path


def load_scaler(filename: str = "scaler.pkl") -> StandardScaler:
    """
    Load a StandardScaler from disk.
    
    Args:
        filename: Name of the scaler file to load
    
    Returns:
        Loaded StandardScaler object
    
    Raises:
        FileNotFoundError: If scaler file doesn't exist
    """
    artifacts_dir = get_artifacts_dir()
    scaler_path = artifacts_dir / filename
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return scaler


def save_feature_names(features: List[str], filename: str = "feature_names.pkl") -> Path:
    """
    Save feature names to disk.
    
    Args:
        features: List of feature names
        filename: Name of the file to save
    
    Returns:
        Path where feature names were saved
    """
    artifacts_dir = get_artifacts_dir()
    features_path = artifacts_dir / filename
    
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    
    return features_path


def load_feature_names(filename: str = "feature_names.pkl") -> List[str]:
    """
    Load feature names from disk.
    
    Args:
        filename: Name of the feature names file to load
    
    Returns:
        List of feature names
    
    Raises:
        FileNotFoundError: If feature names file doesn't exist
    """
    artifacts_dir = get_artifacts_dir()
    features_path = artifacts_dir / filename
    
    if not features_path.exists():
        raise FileNotFoundError(f"Feature names file not found: {features_path}")
    
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    return features


def save_model_metadata(metadata: dict, filename: str = "model_metadata.pkl") -> Path:
    """
    Save model metadata (metrics, parameters, etc.) to disk.
    
    Args:
        metadata: Dictionary with model metadata
        filename: Name of the file to save
    
    Returns:
        Path where metadata was saved
    """
    artifacts_dir = get_artifacts_dir()
    metadata_path = artifacts_dir / filename
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata_path


def load_model_metadata(filename: str = "model_metadata.pkl") -> dict:
    """
    Load model metadata from disk.
    
    Args:
        filename: Name of the metadata file to load
    
    Returns:
        Dictionary with model metadata
    
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    artifacts_dir = get_artifacts_dir()
    metadata_path = artifacts_dir / filename
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return metadata
