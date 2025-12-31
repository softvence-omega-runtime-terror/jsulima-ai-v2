import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "NBA"
NBA_GAMES_FILE = DATA_DIR / "nba_games_dataset.csv"

def get_head_to_head(home_team_name: str, away_team_name: str) -> Dict[str, Any]:
    """
    Get head-to-head match history between two NBA teams.
    Stats (wins, losses) are relative to home_team_name.
    """
    if not NBA_GAMES_FILE.exists():
        return {"error": "NBA games data not found"}

    try:
        df = pd.read_csv(NBA_GAMES_FILE)
    except Exception as e:
        return {"error": f"Failed to read NBA games data: {str(e)}"}

    # Filter matches where both teams participated
    # Case 1: Team A is home, Team B is away
    # Case 2: Team A is away, Team B is home
    mask = (
        ((df['home_name'] == home_team_name) & (df['away_name'] == away_team_name)) |
        ((df['home_name'] == away_team_name) & (df['away_name'] == home_team_name))
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
        # Determine if the requested 'home_team_name' was the home or away team in this specific historical match
        is_requested_home_actually_home = row['home_name'] == home_team_name
        
        # Get scores
        home_score = int(row.get('home_totalscore', 0)) if pd.notna(row.get('home_totalscore')) else 0
        away_score = int(row.get('away_totalscore', 0)) if pd.notna(row.get('away_totalscore')) else 0
        
        # Determine winner
        if home_score > away_score:
            is_home_winner = True
            is_away_winner = False
        elif away_score > home_score:
            is_home_winner = False
            is_away_winner = True
        else:
            is_home_winner = False
            is_away_winner = False

        if is_requested_home_actually_home:
            # Requested team was home in this match
            if is_home_winner:
                wins += 1
            elif is_away_winner:
                losses += 1
            else:
                draws += 1
        else:
            # Requested team was away in this match
            if is_away_winner:
                wins += 1
            elif is_home_winner:
                losses += 1
            else:
                draws += 1

        # Format date as DD.MM.YYYY
        date_str = ""
        raw_date = row.get('date')
        if pd.notna(raw_date):
            try:
                date_str = pd.to_datetime(raw_date).strftime('%d.%m.%Y')
            except:
                date_str = str(raw_date)

        matches_list.append({
            "date": date_str,
            "home_team": row['home_name'],
            "away_team": row['away_name'],
            "home_score": home_score,
            "away_score": away_score
        })

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_matches": len(matches_list),
        "matches": matches_list
    }


