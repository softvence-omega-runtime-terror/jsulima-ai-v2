"""
Data preprocessing module for matching input data to model features
"""
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles preprocessing of match schedule data into model-compatible format
    """
    
    def __init__(self):
        self.feature_names = None
        self.categorical_cols = None
        self.preprocessing_info = None
        self._load_preprocessing_info()
    
    def _load_preprocessing_info(self):
        """Load preprocessing metadata"""
        try:
            preprocessing_path = Path(__file__).parent.parent.parent / "models" / "preprocessing_info.pkl"
            if preprocessing_path.exists():
                self.preprocessing_info = joblib.load(preprocessing_path)
                self.feature_names = self.preprocessing_info.get('feature_names', [])
                self.categorical_cols = self.preprocessing_info.get('categorical_cols', [])
                logger.info("Preprocessing info loaded successfully")
            else:
                logger.warning(f"Preprocessing info not found at {preprocessing_path}")
        except Exception as e:
            logger.error(f"Error loading preprocessing info: {str(e)}")
    
    def preprocess_match_data(self, 
                            players_data: List[Dict[str, Any]],
                            team_stats: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess player data into model-compatible format
        
        Args:
            players_data: List of player statistics dictionaries
            team_stats: Team-level statistics
            
        Returns:
            Tuple of (preprocessed_df, player_names)
        """
        try:
            logger.debug(f"Received {len(players_data)} players for preprocessing")
            
            if not players_data or len(players_data) == 0:
                raise ValueError("No player data provided")
            
            # Create DataFrame from players data
            df = pd.DataFrame(players_data)
            logger.debug(f"Created DataFrame with shape: {df.shape}")
            
            if len(df) == 0:
                raise ValueError("DataFrame is empty after creation")
            
            # Store player names for later mapping
            player_names = df['player_name'].tolist() if 'player_name' in df.columns else []
            logger.debug(f"Extracted {len(player_names)} player names")
            
            # Ensure required columns exist
            required_cols = ['minutes', 'points', 'assists', 'is_home', 'oncourt', 
                           'team_name', 'rebounds', 'steals', 'blocks']
            
            for col in required_cols:
                if col not in df.columns:
                    if col in ['is_home', 'oncourt']:
                        df[col] = 0
                    else:
                        df[col] = 0
            
            # Convert data types
            numeric_cols = ['minutes', 'points', 'assists', 'rebounds', 'steals', 'blocks']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Convert boolean columns to int
            df['is_home'] = df['is_home'].astype(int)
            df['oncourt'] = df['oncourt'].astype(int)
            
            logger.debug(f"After type conversion, DataFrame shape: {df.shape}")
            
            # Engineer features
            df['points_per_minute'] = df.apply(
                lambda row: row['points'] / row['minutes'] if row['minutes'] > 0 else 0,
                axis=1
            )
            df['assists_per_minute'] = df.apply(
                lambda row: row['assists'] / row['minutes'] if row['minutes'] > 0 else 0,
                axis=1
            )
            
            # Create player_status based on points_per_minute (use 75th percentile as threshold)
            # For prediction, we'll estimate this based on performance
            threshold = df['points_per_minute'].quantile(0.75) if len(df) > 0 else 0.5
            df['player_status'] = df['points_per_minute'].apply(
                lambda x: 'Good' if x >= threshold else 'Average'
            )
            
            # Select and reorder features to match training
            feature_df = df[['minutes', 'points', 'assists', 'is_home', 'oncourt',
                           'team_name', 'rebounds', 'steals', 'blocks',
                           'points_per_minute', 'assists_per_minute', 'player_status']].copy()
            
            logger.debug(f"Before one-hot encoding, shape: {feature_df.shape}")
            
            # Apply one-hot encoding to categorical columns
            feature_df = pd.get_dummies(feature_df, columns=['team_name', 'player_status'], drop_first=True)
            
            logger.debug(f"After one-hot encoding, shape: {feature_df.shape}, columns: {list(feature_df.columns)}")
            
            # Reindex to match training features
            if self.feature_names:
                feature_df = feature_df.reindex(columns=self.feature_names, fill_value=0)
            
            logger.debug(f"Final preprocessed shape: {feature_df.shape}")
            
            if len(feature_df) == 0:
                raise ValueError("DataFrame became empty after preprocessing")
            
            return feature_df, player_names
        
        except Exception as e:
            logger.error(f"Error preprocessing match data: {str(e)}", exc_info=True)
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names used in the model"""
        return self.feature_names if self.feature_names else []
