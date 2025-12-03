"""
UFC Feature Engineering Module

This module contains functions for calculating derived features
from raw UFC fight statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union


def calculate_total_strikes(row: pd.Series, fighter: str) -> float:
    """
    Calculate total strikes (head + body + leg).
    
    Args:
        row: DataFrame row with fight statistics
        fighter: 'local' or 'away'
    
    Returns:
        Total strikes count
    """
    head = row.get(f'{fighter}_strikes_total_head', 0)
    body = row.get(f'{fighter}_strikes_total_body', 0)
    leg = row.get(f'{fighter}_strikes_total_leg', 0)
    
    return float(head + body + leg)


def calculate_power_strikes(row: pd.Series, fighter: str) -> float:
    """
    Calculate total power strikes (head + body + leg).
    
    Args:
        row: DataFrame row with fight statistics
        fighter: 'local' or 'away'
    
    Returns:
        Total power strikes count
    """
    head = row.get(f'{fighter}_strikes_power_head', 0)
    body = row.get(f'{fighter}_strikes_power_body', 0)
    leg = row.get(f'{fighter}_strikes_power_leg', 0)
    
    return float(head + body + leg)


def calculate_strike_accuracy(total_strikes: float, power_strikes: float) -> float:
    """
    Calculate strike accuracy as percentage of power strikes to total strikes.
    
    Args:
        total_strikes: Total number of strikes
        power_strikes: Number of power strikes
    
    Returns:
        Strike accuracy percentage (0-100)
    """
    if total_strikes > 0:
        return (power_strikes / total_strikes) * 100
    return 0.0


def calculate_head_strike_ratio(head_strikes: float, total_strikes: float) -> float:
    """
    Calculate percentage of strikes aimed at head.
    
    Args:
        head_strikes: Number of head strikes
        total_strikes: Total number of strikes
    
    Returns:
        Head strike ratio percentage (0-100)
    """
    if total_strikes > 0:
        return (head_strikes / total_strikes) * 100
    return 0.0


def calculate_takedown_defense(opponent_att: float, opponent_landed: float) -> float:
    """
    Calculate takedown defense percentage.
    
    Args:
        opponent_att: Opponent's takedown attempts
        opponent_landed: Opponent's successful takedowns
    
    Returns:
        Takedown defense percentage (0-100)
    """
    if opponent_att > 0:
        return ((opponent_att - opponent_landed) / opponent_att) * 100
    return 0.0


def convert_time_to_seconds(time_str: str) -> int:
    """
    Convert time string in MM:SS format to total seconds.
    
    Args:
        time_str: Time string in 'MM:SS' format
    
    Returns:
        Total seconds
    """
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        return 0
    except (ValueError, AttributeError):
        return 0


def calculate_grappling_score(takedowns_landed: float, submissions: float) -> float:
    """
    Calculate grappling effectiveness score.
    Formula: (takedowns_landed * 2) + (submissions * 3)
    
    Args:
        takedowns_landed: Number of successful takedowns
        submissions: Number of submission attempts
    
    Returns:
        Grappling score
    """
    return (takedowns_landed * 2) + (submissions * 3)


def calculate_damage_score(total_strikes: float, knockdowns: float) -> float:
    """
    Calculate overall damage score.
    Formula: (total_strikes * 0.5) + (knockdowns * 5)
    
    Args:
        total_strikes: Total number of strikes
        knockdowns: Number of knockdowns
    
    Returns:
        Damage score
    """
    return (total_strikes * 0.5) + (knockdowns * 5)


def calculate_control_index(
    total_strikes: float,
    takedowns_landed: float,
    submissions: float,
    control_time_seconds: int
) -> float:
    """
    Calculate fight control index (composite offensive metric).
    Formula: total_strikes + (takedowns_landed * 3) + (submissions * 5) + (control_time / 60)
    
    Args:
        total_strikes: Total number of strikes
        takedowns_landed: Number of successful takedowns
        submissions: Number of submission attempts
        control_time_seconds: Control time in seconds
    
    Returns:
        Control index score
    """
    return (
        total_strikes +
        (takedowns_landed * 3) +
        (submissions * 5) +
        (control_time_seconds / 60)
    )


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the DataFrame.
    
    Args:
        df: Raw UFC fight data
    
    Returns:
        DataFrame with all engineered features added
    """
    df = df.copy()
    
    # Calculate total and power strikes
    df['local_total_strikes'] = df.apply(
        lambda row: calculate_total_strikes(row, 'local'), axis=1
    )
    df['away_total_strikes'] = df.apply(
        lambda row: calculate_total_strikes(row, 'away'), axis=1
    )
    df['local_power_strikes'] = df.apply(
        lambda row: calculate_power_strikes(row, 'local'), axis=1
    )
    df['away_power_strikes'] = df.apply(
        lambda row: calculate_power_strikes(row, 'away'), axis=1
    )
    
    # Calculate strike accuracy
    df['local_strike_accuracy'] = df.apply(
        lambda row: calculate_strike_accuracy(
            row['local_total_strikes'], row['local_power_strikes']
        ), axis=1
    )
    df['away_strike_accuracy'] = df.apply(
        lambda row: calculate_strike_accuracy(
            row['away_total_strikes'], row['away_power_strikes']
        ), axis=1
    )
    
    # Calculate head strike ratio
    df['local_head_strike_ratio'] = df.apply(
        lambda row: calculate_head_strike_ratio(
            row.get('local_strikes_total_head', 0), row['local_total_strikes']
        ), axis=1
    )
    df['away_head_strike_ratio'] = df.apply(
        lambda row: calculate_head_strike_ratio(
            row.get('away_strikes_total_head', 0), row['away_total_strikes']
        ), axis=1
    )
    
    # Calculate takedown defense
    df['local_takedown_defense'] = df.apply(
        lambda row: calculate_takedown_defense(
            row.get('away_takedowns_att', 0), row.get('away_takedowns_landed', 0)
        ), axis=1
    )
    df['away_takedown_defense'] = df.apply(
        lambda row: calculate_takedown_defense(
            row.get('local_takedowns_att', 0), row.get('local_takedowns_landed', 0)
        ), axis=1
    )
    
    # Convert control time to seconds
    df['local_control_time_seconds'] = df['local_control_time'].apply(
        convert_time_to_seconds
    )
    df['away_control_time_seconds'] = df['away_control_time'].apply(
        convert_time_to_seconds
    )
    
    # Calculate grappling scores
    df['local_grappling_score'] = df.apply(
        lambda row: calculate_grappling_score(
            row.get('local_takedowns_landed', 0), row.get('local_submissions', 0)
        ), axis=1
    )
    df['away_grappling_score'] = df.apply(
        lambda row: calculate_grappling_score(
            row.get('away_takedowns_landed', 0), row.get('away_submissions', 0)
        ), axis=1
    )
    
    # Calculate damage scores
    df['local_damage_score'] = df.apply(
        lambda row: calculate_damage_score(
            row['local_total_strikes'], row.get('local_knockdowns', 0)
        ), axis=1
    )
    df['away_damage_score'] = df.apply(
        lambda row: calculate_damage_score(
            row['away_total_strikes'], row.get('away_knockdowns', 0)
        ), axis=1
    )
    
    # Calculate control indices
    df['local_control_index'] = df.apply(
        lambda row: calculate_control_index(
            row['local_total_strikes'],
            row.get('local_takedowns_landed', 0),
            row.get('local_submissions', 0),
            row['local_control_time_seconds']
        ), axis=1
    )
    df['away_control_index'] = df.apply(
        lambda row: calculate_control_index(
            row['away_total_strikes'],
            row.get('away_takedowns_landed', 0),
            row.get('away_submissions', 0),
            row['away_control_time_seconds']
        ), axis=1
    )
    
    return df
