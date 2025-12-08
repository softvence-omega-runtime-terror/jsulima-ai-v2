"""
UFC Data Loader Module

This module handles loading and validation of UFC fight data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_ufc_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load UFC fight data from CSV file.
    
    Args:
        data_path: Path to the UFC data CSV file. 
                   Defaults to './app/data/ufc_2010_2023.csv'
    
    Returns:
        DataFrame containing UFC fight data
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the data file is empty or invalid
    """
    if data_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "app" / "data" / "ufc_2018_2024.csv"
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    if df.empty:
        raise ValueError("Loaded data is empty")
    
    return df


def validate_data_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has all required columns for UFC prediction.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if schema is valid
    
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = [
        # Match metadata
        'date', 'localteam_name', 'awayteam_name',
        'local_winner', 'away_winner',
        
        # Strike statistics
        'local_strikes_total_head', 'local_strikes_total_body', 'local_strikes_total_leg',
        'local_strikes_power_head', 'local_strikes_power_body', 'local_strikes_power_leg',
        'away_strikes_total_head', 'away_strikes_total_body', 'away_strikes_total_leg',
        'away_strikes_power_head', 'away_strikes_power_body', 'away_strikes_power_leg',
        
        # Grappling statistics
        'local_takedowns_att', 'local_takedowns_landed', 'local_submissions',
        'away_takedowns_att', 'away_takedowns_landed', 'away_submissions',
        
        # Other statistics
        'local_control_time', 'away_control_time',
        'local_knockdowns', 'away_knockdowns'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get summary information about the UFC data.
    
    Args:
        df: UFC fight data DataFrame
    
    Returns:
        Dictionary with data statistics
    """
    return {
        'total_fights': len(df),
        'date_range': {
            'start': df['date'].min() if 'date' in df.columns else None,
            'end': df['date'].max() if 'date' in df.columns else None
        },
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'shape': df.shape
    }

import pandas as pd
from .UFC.player_service import PlayerService

class DataLoader:

    def __init__(self):
        self.df = pd.read_csv("app/data/ufc_2018_2024.csv")
        self.player_service = PlayerService(self.df)

    def get_player_service(self):
        return self.player_service
