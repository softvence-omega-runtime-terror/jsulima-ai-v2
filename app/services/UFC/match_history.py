import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
HISTORICAL_FIGHTS_FILE = DATA_DIR / "historical_fights.csv"

def get_head_to_head(local_team_id: str, away_team_id: str) -> Dict[str, Any]:
    """
    Get head-to-head match history between two fighters.
    Stats (wins, losses, draws) are relative to local_team_id.
    """
    if not HISTORICAL_FIGHTS_FILE.exists():
        return {"error": "Historical fights data not found"}

    try:
        df = pd.read_csv(HISTORICAL_FIGHTS_FILE)
    except Exception as e:
        return {"error": f"Failed to read historical data: {str(e)}"}

    # Convert IDs to string for comparison
    local_id = str(local_team_id)
    away_id = str(away_team_id)

    # Filter matches where both fighters participated
    # Case 1: Fighter A is local, Fighter B is away
    # Case 2: Fighter A is away, Fighter B is local
    mask = (
        ((df['localteam_id'].astype(str) == local_id) & (df['awayteam_id'].astype(str) == away_id)) |
        ((df['localteam_id'].astype(str) == away_id) & (df['awayteam_id'].astype(str) == local_id))
    )
    
    matches_df = df[mask].copy()
    
    if matches_df.empty:
        return {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "matches": []
        }

    wins = 0
    losses = 0
    draws = 0
    matches_list = []

    for _, row in matches_df.iterrows():
        # Determine if the requested 'local_team_id' was the local or away fighter in THIS specific historical match
        is_requested_local_actually_local = str(row['localteam_id']) == local_id
        
        # Determine winner relative to the requested local_team_id
        winner_val = row.get('localteam_winner')
        # Handle different types of winner indicators (bool or string)
        is_local_winner = winner_val == True or str(winner_val).lower() == 'true'
        
        away_winner_val = row.get('awayteam_winner')
        is_away_winner = away_winner_val == True or str(away_winner_val).lower() == 'true'

        if is_requested_local_actually_local:
            # Requested fighter was local in this match
            if is_local_winner:
                wins += 1
            elif is_away_winner:
                losses += 1
            else:
                draws += 1
                
            home_data = {
                "id": int(row['localteam_id']),
                "name": row['localteam_name'],
                "total_strike_head": int(row.get('local_strikes_head', 0)),
                "total_strike_body": int(row.get('local_strikes_body', 0)),
                "total_strike_legs": int(row.get('local_strikes_legs', 0)),
            }
            away_data = {
                "id": int(row['awayteam_id']),
                "name": row['awayteam_name'],
                "total_strike_head": int(row.get('away_strikes_head', 0)),
                "total_strike_body": int(row.get('away_strikes_body', 0)),
                "total_strike_legs": int(row.get('away_strikes_legs', 0)),
            }
        else:
            # Requested fighter was away in this match
            if is_away_winner:
                wins += 1
            elif is_local_winner:
                losses += 1
            else:
                draws += 1
                
            # Swap data so the requested "local_team_id" corresponds to home_team in the response matches list?
            # User example shows "home_team" and "away_team" in the list.
            # Usually we keep the original roles or normalize. 
            # The example shows names like "Arizona Cardinals" and "Atlanta Falcons".
            # Let's normalize so home_team IS the local_team_id requested.
            home_data = {
                "id": int(row['awayteam_id']),
                "name": row['awayteam_name'],
                "total_strike_head": int(row.get('away_strikes_head', 0)),
                "total_strike_body": int(row.get('away_strikes_body', 0)),
                "total_strike_legs": int(row.get('away_strikes_legs', 0)),
            }
            away_data = {
                "id": int(row['localteam_id']),
                "name": row['localteam_name'],
                "total_strike_head": int(row.get('local_strikes_head', 0)),
                "total_strike_body": int(row.get('local_strikes_body', 0)),
                "total_strike_legs": int(row.get('local_strikes_legs', 0)),
            }

        matches_list.append({
            "match_id": int(row['match_id']),
            "date": row['match_date'],
            "home_team": home_data,
            "away_team": away_data
        })

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "matches": matches_list
    }
