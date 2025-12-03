"""
UFC Data Preprocessor Module

This module handles data cleaning, feature engineering, and preparation
for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from typing import Tuple, List, Optional
import warnings

from app.utils.feature_engineering import engineer_all_features

warnings.filterwarnings('ignore')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the UFC data by removing incomplete records and handling missing values.
    
    Args:
        df: Raw UFC fight data
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove rows with missing fight statistics (incomplete records)
    df_clean = df_clean.dropna(
        subset=['local_strikes_total_head', 'away_strikes_total_head']
    )
    
    # Fill missing values in numerical columns with 0
    numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df_clean[col] = df_clean[col].fillna(0)
    
    return df_clean


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features by selecting relevant columns and creating target variable.
    
    Args:
        df: DataFrame with engineered features
    
    Returns:
        DataFrame with selected features and target
    """
    df_prepared = df.copy()
    
    # Create target variable: 1 if local_winner, 0 if away_winner
    df_prepared['target'] = (df_prepared['local_winner'] == True).astype(int)
    
    # Define columns to exclude from features
    exclude_cols = [
        'source_date', 'category', 'category_date', 'category_id', 'match_id',
        'date', 'time', 'status', 'localteam_name', 'local_id', 'awayteam_name', 'away_id',
        'local_winner', 'away_winner', 'win_type', 'win_round', 'win_minute',
        'won_by_ko_type', 'won_by_ko_target', 'win_sub_type', 'win_points_score',
        'local_control_time', 'away_control_time', 'target'
    ]
    
    # Select feature columns
    feature_cols = [col for col in df_prepared.columns if col not in exclude_cols]
    
    return df_prepared[feature_cols + ['target']]


def get_feature_columns() -> List[str]:
    """
    Get the list of feature column names used in the model.
    
    Returns:
        List of feature column names
    """
    return [
        'local_strikes_total_head', 'local_strikes_total_body', 'local_strikes_total_leg',
        'local_strikes_power_head', 'local_strikes_power_body', 'local_strikes_power_leg',
        'local_takedowns_att', 'local_takedowns_landed', 'local_submissions', 'local_knockdowns',
        'away_strikes_total_head', 'away_strikes_total_body', 'away_strikes_total_leg',
        'away_strikes_power_head', 'away_strikes_power_body', 'away_strikes_power_leg',
        'away_takedowns_att', 'away_takedowns_landed', 'away_submissions', 'away_knockdowns',
        'local_total_strikes', 'away_total_strikes',
        'local_power_strikes', 'away_power_strikes',
        'local_strike_accuracy', 'away_strike_accuracy',
        'local_head_strike_ratio', 'away_head_strike_ratio',
        'local_takedown_defense', 'away_takedown_defense',
        'local_control_time_seconds', 'away_control_time_seconds',
        'local_grappling_score', 'away_grappling_score',
        'local_damage_score', 'away_damage_score',
        'local_control_index', 'away_control_index'
    ]


def split_and_scale_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Split data into train/test sets and apply standard scaling.
    
    Args:
        df: DataFrame with features and target
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Apply variance threshold to remove low-variance features
    variance_threshold = VarianceThreshold(threshold=0.01)
    X_var = variance_threshold.fit_transform(X)
    
    # Get selected feature names
    selected_feature_indices = variance_threshold.get_support(indices=True)
    selected_features = [X.columns[i] for i in selected_feature_indices]
    X = X[selected_features]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Apply standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)
    
    # Reset indices
    X_train_scaled = X_train_scaled.reset_index(drop=True)
    X_test_scaled = X_test_scaled.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Complete preprocessing pipeline: clean, engineer features, prepare, and split.
    
    Args:
        df: Raw UFC fight data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Step 1: Clean data
    df_clean = clean_data(df)
    
    # Step 2: Engineer features
    df_engineered = engineer_all_features(df_clean)
    
    # Step 3: Prepare features
    df_prepared = prepare_features(df_engineered)
    
    # Step 4: Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
        df_prepared, test_size, random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler
