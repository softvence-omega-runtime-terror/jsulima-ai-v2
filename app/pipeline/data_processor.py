"""
Data Processor Module
Cleans and processes raw fight data for model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import re


DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class DataProcessor:
    """Processes raw fight data into clean DataFrame"""
    
    # Weight class encoding
    WEIGHT_CLASSES = {
        "Strawweight": 1,
        "Women's Strawweight": 1,
        "Flyweight": 2,
        "Women's Flyweight": 2,
        "Bantamweight": 3,
        "Women's Bantamweight": 3,
        "Featherweight": 4,
        "Women's Featherweight": 4,
        "Lightweight": 5,
        "Welterweight": 6,
        "Middleweight": 7,
        "Light Heavyweight": 8,
        "Heavyweight": 9,
        "Catch Weight": 5,  # Default to lightweight
    }
    
    # Win method encoding
    WIN_METHODS = {
        "KO": "KO",
        "TKO": "KO",
        "SUB": "SUB",
        "Submission": "SUB",
        "Points": "DEC",
        "U Dec": "DEC",
        "S Dec": "DEC",
        "M Dec": "DEC",
        "Decision": "DEC",
        "DQ": "OTHER",
        "NC": "OTHER",
        "Draw": "OTHER",
    }
    
    def __init__(self):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load raw fight data from CSV"""
        if filepath is None:
            filepath = DATA_DIR / "historical_fights.csv"
        
        df = pd.read_csv(filepath, encoding='utf-8')
        return df
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into clean format"""
        # Filter only completed fights
        df = df[df['status'] == 'Final'].copy()
        
        # Parse dates
        df['fight_date'] = pd.to_datetime(df['match_date'], format='%d.%m.%Y', errors='coerce')
        
        # Determine winner
        df['winner'] = df.apply(self._determine_winner, axis=1)
        
        # Encode win method
        df['win_method'] = df['win_type'].apply(self._encode_win_method)
        
        # Parse win round
        df['fight_round'] = pd.to_numeric(df['win_round'], errors='coerce').fillna(0).astype(int)
        
        # Encode weight class (default to 5/Lightweight if column is missing)
        if 'weight_class' in df.columns:
            df['weight_class_encoded'] = df['weight_class'].apply(
                lambda x: self.WEIGHT_CLASSES.get(x, 5) if isinstance(x, str) else 5
            )
        else:
            # Weight class not available in data, default to Lightweight (5)
            df['weight_class_encoded'] = 5
        
        # Calculate total strikes
        df['local_total_strikes'] = (
            df['local_strikes_head'].fillna(0) + 
            df['local_strikes_body'].fillna(0) + 
            df['local_strikes_legs'].fillna(0)
        )
        df['away_total_strikes'] = (
            df['away_strikes_head'].fillna(0) + 
            df['away_strikes_body'].fillna(0) + 
            df['away_strikes_legs'].fillna(0)
        )
        
        # Calculate takedown accuracy
        df['local_td_accuracy'] = df.apply(
            lambda r: r['local_takedowns_landed'] / r['local_takedowns_att'] 
            if r['local_takedowns_att'] > 0 else 0, axis=1
        )
        df['away_td_accuracy'] = df.apply(
            lambda r: r['away_takedowns_landed'] / r['away_takedowns_att'] 
            if r['away_takedowns_att'] > 0 else 0, axis=1
        )
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def _determine_winner(self, row) -> int:
        """Determine winner: 1 = localteam, 0 = awayteam, -1 = draw/NC"""
        if row.get('localteam_winner') == True or row.get('localteam_winner') == 'True':
            return 1
        elif row.get('awayteam_winner') == True or row.get('awayteam_winner') == 'True':
            return 0
        else:
            return -1
    
    def _encode_win_method(self, method: str) -> str:
        """Encode win method to standard categories"""
        if not isinstance(method, str):
            return "OTHER"
        
        for key, value in self.WIN_METHODS.items():
            if key.lower() in method.lower():
                return value
        return "OTHER"
    
    def save_processed(self, df: pd.DataFrame, filename: str = "processed_fights.csv"):
        """Save processed data"""
        output_path = PROCESSED_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        return output_path


def process_data(input_path: Optional[str] = None) -> pd.DataFrame:
    """Process raw fight data"""
    processor = DataProcessor()
    df = processor.load_raw_data(input_path)
    df = processor.process(df)
    processor.save_processed(df)
    return df


if __name__ == "__main__":
    df = process_data()
    print(f"Processed {len(df)} fights")
    print(f"Columns: {list(df.columns)}")
