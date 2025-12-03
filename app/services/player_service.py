"""
Player Service Module

This module handles fetching player statistics and information from historical UFC data.
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
from app.services.loader import load_ufc_data
from app.core.logger import get_logger

logger = get_logger(__name__)

# Cache for loaded data
_data_cache: Optional[pd.DataFrame] = None


def get_data_cache() -> pd.DataFrame:
    """
    Get or load the UFC data cache.
    
    Returns:
        DataFrame with UFC fight data
    """
    global _data_cache
    if _data_cache is None:
        logger.info("Loading UFC data for player service...")
        _data_cache = load_ufc_data()
    return _data_cache


def get_player_stats_by_id(player_id: int, before_date: Optional[str] = None) -> Dict:
    """
    Get average statistics for a player based on their historical fights.
    
    Args:
        player_id: Player ID to look up
        before_date: Optional date to filter fights before this date (YYYY-MM-DD format)
    
    Returns:
        Dictionary with average player statistics
    
    Raises:
        ValueError: If player not found or has no historical data
    """
    df = get_data_cache()
    
    # Filter fights where player was either local or away
    player_fights = df[
        (df['local_id'] == player_id) | (df['away_id'] == player_id)
    ].copy()
    
    # Filter by date if provided
    if before_date:
        try:
            before_date_obj = pd.to_datetime(before_date)
            player_fights = player_fights[pd.to_datetime(player_fights['date']) < before_date_obj]
        except Exception as e:
            logger.warning(f"Error parsing date {before_date}: {e}")
    
    if player_fights.empty:
        raise ValueError(f"Player ID {player_id} not found or has no historical data")
    
    # Separate local and away fights
    local_fights = player_fights[player_fights['local_id'] == player_id]
    away_fights = player_fights[player_fights['away_id'] == player_id]
    
    # Helper function to convert control_time string to seconds
    def time_to_seconds(time_str):
        """Convert MM:SS format to seconds."""
        try:
            if pd.isna(time_str):
                return 0
            if isinstance(time_str, (int, float)):
                return int(time_str)
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return 0
        except (ValueError, AttributeError):
            return 0
    
    # Calculate average statistics
    stats = {}
    
    # For local fights, use local stats
    if not local_fights.empty:
        local_cols = [
            'local_strikes_total_head', 'local_strikes_total_body', 'local_strikes_total_leg',
            'local_strikes_power_head', 'local_strikes_power_body', 'local_strikes_power_leg',
            'local_takedowns_att', 'local_takedowns_landed', 'local_submissions',
            'local_knockdowns', 'local_control_time'
        ]
        for col in local_cols:
            avg_key = col.replace('local_', '')
            if avg_key == 'control_time':
                # Convert to seconds, average, then convert back
                seconds_list = local_fights[col].apply(time_to_seconds)
                avg_seconds = seconds_list.mean() if not seconds_list.empty else 0
                stats[avg_key] = avg_seconds
            else:
                stats[avg_key] = local_fights[col].mean()
    
    # For away fights, use away stats
    if not away_fights.empty:
        away_cols = [
            'away_strikes_total_head', 'away_strikes_total_body', 'away_strikes_total_leg',
            'away_strikes_power_head', 'away_strikes_power_body', 'away_strikes_power_leg',
            'away_takedowns_att', 'away_takedowns_landed', 'away_submissions',
            'away_knockdowns', 'away_control_time'
        ]
        for col in away_cols:
            avg_key = col.replace('away_', '')
            if avg_key == 'control_time':
                # Convert to seconds, average, then combine with local if exists
                seconds_list = away_fights[col].apply(time_to_seconds)
                avg_seconds = seconds_list.mean() if not seconds_list.empty else 0
                if avg_key in stats:
                    # Weighted average
                    total_fights = len(local_fights) + len(away_fights)
                    stats[avg_key] = (
                        (stats[avg_key] * len(local_fights) + avg_seconds * len(away_fights))
                        / total_fights
                    )
                else:
                    stats[avg_key] = avg_seconds
            else:
                # Combine with local stats if they exist
                if avg_key in stats:
                    # Weighted average based on number of fights
                    total_fights = len(local_fights) + len(away_fights)
                    stats[avg_key] = (
                        (stats[avg_key] * len(local_fights) + away_fights[col].mean() * len(away_fights))
                        / total_fights
                    )
                else:
                    stats[avg_key] = away_fights[col].mean()
    
    # Fill NaN values and convert control_time to string format
    for key in stats:
        if pd.isna(stats[key]):
            if key == 'control_time':
                stats[key] = "0:00"
            else:
                stats[key] = 0.0
        else:
            if key == 'control_time':
                # Convert seconds to MM:SS format
                total_seconds = int(stats[key])
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                stats[key] = f"{minutes}:{seconds:02d}"
            else:
                stats[key] = float(stats[key])
    
    # Ensure control_time exists
    if 'control_time' not in stats:
        stats['control_time'] = "0:00"
    
    return stats


def get_player_name_by_id(player_id: int) -> str:
    """
    Get player name by ID.
    
    Args:
        player_id: Player ID to look up
    
    Returns:
        Player name
    
    Raises:
        ValueError: If player not found
    """
    df = get_data_cache()
    
    # Try to find player as local
    local_match = df[df['local_id'] == player_id]
    if not local_match.empty:
        return local_match.iloc[0]['localteam_name']
    
    # Try to find player as away
    away_match = df[df['away_id'] == player_id]
    if not away_match.empty:
        return away_match.iloc[0]['awayteam_name']
    
    raise ValueError(f"Player ID {player_id} not found")


def get_fighter_stats_for_prediction(
    player_id_1: int,
    player_id_2: int,
    date: Optional[str] = None
) -> Dict:
    """
    Get fighter statistics for both players formatted for prediction.
    
    Args:
        player_id_1: First player ID (will be local fighter)
        player_id_2: Second player ID (will be away fighter)
        date: Optional date to filter historical data
    
    Returns:
        Dictionary with fighter stats formatted for prediction API
    """
    # Get stats for both players
    player1_stats = get_player_stats_by_id(player_id_1, before_date=date)
    player2_stats = get_player_stats_by_id(player_id_2, before_date=date)
    
    # Format for prediction (player1 as local, player2 as away)
    fighter_stats = {
        # Local fighter (player1)
        'local_strikes_total_head': player1_stats.get('strikes_total_head', 0),
        'local_strikes_total_body': player1_stats.get('strikes_total_body', 0),
        'local_strikes_total_leg': player1_stats.get('strikes_total_leg', 0),
        'local_strikes_power_head': player1_stats.get('strikes_power_head', 0),
        'local_strikes_power_body': player1_stats.get('strikes_power_body', 0),
        'local_strikes_power_leg': player1_stats.get('strikes_power_leg', 0),
        'local_takedowns_att': player1_stats.get('takedowns_att', 0),
        'local_takedowns_landed': player1_stats.get('takedowns_landed', 0),
        'local_submissions': player1_stats.get('submissions', 0),
        'local_knockdowns': player1_stats.get('knockdowns', 0),
        'local_control_time': player1_stats.get('control_time', '0:00'),
        
        # Away fighter (player2)
        'away_strikes_total_head': player2_stats.get('strikes_total_head', 0),
        'away_strikes_total_body': player2_stats.get('strikes_total_body', 0),
        'away_strikes_total_leg': player2_stats.get('strikes_total_leg', 0),
        'away_strikes_power_head': player2_stats.get('strikes_power_head', 0),
        'away_strikes_power_body': player2_stats.get('strikes_power_body', 0),
        'away_strikes_power_leg': player2_stats.get('strikes_power_leg', 0),
        'away_takedowns_att': player2_stats.get('takedowns_att', 0),
        'away_takedowns_landed': player2_stats.get('takedowns_landed', 0),
        'away_submissions': player2_stats.get('submissions', 0),
        'away_knockdowns': player2_stats.get('knockdowns', 0),
        'away_control_time': player2_stats.get('control_time', '0:00'),
    }
    
    return fighter_stats

