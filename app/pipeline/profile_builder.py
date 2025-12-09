"""
Fighter Profile Builder
Aggregates comprehensive statistics for all fighters
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class FighterProfileBuilder:
    """Builds detailed statistical profiles for fighters"""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "id": "",
            "name": "",
            "record": {"wins": 0, "losses": 0, "draws": 0, "nc": 0},
            "win_methods": {"KO": 0, "SUB": 0, "DEC": 0, "OTHER": 0},
            "loss_methods": {"KO": 0, "SUB": 0, "DEC": 0, "OTHER": 0},
            "striking": {
                "total_landed": 0, "total_absorbed": 0,
                "head_landed": 0, "head_absorbed": 0,
                "body_landed": 0, "body_absorbed": 0,
                "legs_landed": 0, "legs_absorbed": 0,
                "power_head_landed": 0, "power_body_landed": 0, "power_legs_landed": 0,
            },
            "grappling": {
                "takedowns_landed": 0, "takedowns_attempts": 0, "takedowns_absorbed": 0,
                "submissions_attempts": 0,
                "control_time_seconds": 0,
            },
            "fight_time_seconds": 0,
            "total_fights": 0,
            "pvp_history": {}, # Opponent ID -> Result
            "last_fight_date": None,
        })
        self.fighter_names = {}

    def load_data(self) -> pd.DataFrame:
        """Load processed fight data"""
        return pd.read_csv(PROCESSED_DIR / "processed_fights.csv")

    def build_profiles(self, df: pd.DataFrame):
        """Process all fights and build profiles"""
        # Sort by date
        df['fight_date'] = pd.to_datetime(df['fight_date'])
        df = df.sort_values('fight_date')
        
        for _, row in df.iterrows():
            self._process_fight(row)
            
        # Calculate averages and percentages
        self._finalize_profiles()
        
    def _process_fight(self, row: pd.Series):
        """Process a single fight row"""
        local_id = str(row['localteam_id'])
        away_id = str(row['awayteam_id'])
        
        if local_id == 'nan' or away_id == 'nan':
            return

        # Store names
        self.fighter_names[local_id] = row['localteam_name']
        self.fighter_names[away_id] = row['awayteam_name']
        
        # Update Local Fighter
        self._update_fighter_stats(local_id, row, is_local=True, opponent_id=away_id)
        
        # Update Away Fighter
        self._update_fighter_stats(away_id, row, is_local=False, opponent_id=local_id)

    def _update_fighter_stats(self, fighter_id: str, row: pd.Series, is_local: bool, opponent_id: str):
        """Update stats for a single fighter from a fight"""
        profile = self.profiles[fighter_id]
        profile["id"] = fighter_id
        profile["name"] = self.fighter_names.get(fighter_id, "Unknown")
        profile["total_fights"] += 1
        profile["last_fight_date"] = str(row['fight_date'].date())
        
        # Prefixes
        prefix = "local_" if is_local else "away_"
        opp_prefix = "away_" if is_local else "local_"
        
        # Result
        winner = row['winner']
        method = row['win_method']
        
        is_winner = (is_local and winner == 1) or (not is_local and winner == 0)
        is_draw = winner == -1
        
        if is_draw:
            profile["record"]["draws"] += 1
            result_str = "DRAW"
        elif is_winner:
            profile["record"]["wins"] += 1
            profile["win_methods"][method] = profile["win_methods"].get(method, 0) + 1
            result_str = f"WIN ({method})"
        else:
            profile["record"]["losses"] += 1
            profile["loss_methods"][method] = profile["loss_methods"].get(method, 0) + 1
            result_str = f"LOSS ({method})"
            
        # PvP History
        profile["pvp_history"][opponent_id] = {
            "date": str(row['fight_date'].date()),
            "result": result_str,
            "method": method,
            "round": int(row['fight_round'])
        }
        
        # Striking Stats
        s = profile["striking"]
        s["total_landed"] += row.get(f"{prefix}total_strikes", 0)
        s["total_absorbed"] += row.get(f"{opp_prefix}total_strikes", 0)
        s["head_landed"] += row.get(f"{prefix}strikes_head", 0)
        s["head_absorbed"] += row.get(f"{opp_prefix}strikes_head", 0)
        s["body_landed"] += row.get(f"{prefix}strikes_body", 0)
        s["body_absorbed"] += row.get(f"{opp_prefix}strikes_body", 0)
        s["legs_landed"] += row.get(f"{prefix}strikes_legs", 0)
        s["legs_absorbed"] += row.get(f"{opp_prefix}strikes_legs", 0)
        
        # Power Strikes
        s["power_head_landed"] += row.get(f"{prefix}power_head", 0)
        s["power_body_landed"] += row.get(f"{prefix}power_body", 0)
        s["power_legs_landed"] += row.get(f"{prefix}power_legs", 0)
        
        # Grappling Stats
        g = profile["grappling"]
        g["takedowns_landed"] += row.get(f"{prefix}takedowns_landed", 0)
        g["takedowns_attempts"] += row.get(f"{prefix}takedowns_att", 0)
        g["takedowns_absorbed"] += row.get(f"{opp_prefix}takedowns_landed", 0)
        # Note: submissions usually tracked as total attempts in processed data?
        # Checking data_fetcher, it saves 'submissions' (total).
        # But processed_fights might not have it explicitly exposed in the same way?
        # Let's check if 'local_submissions' exists. If not, default 0.
        # Actually data_fetcher saves it as f"{prefix}submissions".
        # data_processor doesn't explicitly rename it, so it should be there.
        g["submissions_attempts"] += row.get(f"{prefix}submissions", 0)
        
        # Time
        # control_time is in seconds in processed data? 
        # data_fetcher parses "M:SS" to seconds.
        g["control_time_seconds"] += row.get(f"{prefix}control_time", 0)
        
        # Fight time estimate (5 mins per round)
        profile["fight_time_seconds"] += row.get("fight_round", 0) * 300

    def _finalize_profiles(self):
        """Calculate derived stats (percentages, averages)"""
        for pid, p in self.profiles.items():
            fights = max(p["total_fights"], 1)
            wins = max(p["record"]["wins"], 1)
            
            # Win Percentage
            p["record"]["win_pct"] = round(p["record"]["wins"] / fights * 100, 1)
            
            # Method Percentages
            for m in list(p["win_methods"].keys()):
                p["win_methods"][f"{m}_pct"] = round(p["win_methods"][m] / wins * 100, 1)
            
            # Striking Averages
            s = p["striking"]
            s["slpm"] = round(s["total_landed"] / max(p["fight_time_seconds"] / 60, 1), 2)
            s["sapm"] = round(s["total_absorbed"] / max(p["fight_time_seconds"] / 60, 1), 2)
            s["avg_head_landed"] = round(s["head_landed"] / fights, 1)
            s["avg_body_landed"] = round(s["body_landed"] / fights, 1)
            s["avg_legs_landed"] = round(s["legs_landed"] / fights, 1)
            
            # Grappling Averages
            g = p["grappling"]
            g["td_avg_15m"] = round(g["takedowns_landed"] / max(p["fight_time_seconds"] / 900, 1), 2)
            g["sub_avg_15m"] = round(g["submissions_attempts"] / max(p["fight_time_seconds"] / 900, 1), 2)
            g["td_accuracy"] = round(g["takedowns_landed"] / max(g["takedowns_attempts"], 1) * 100, 1)

        # Merge with physical attributes from raw profiles
        self._merge_physical_attributes()

    def _merge_physical_attributes(self):
        """Merge physical attributes from raw fighter profiles"""
        raw_profiles_path = Path(__file__).parent.parent / "data" / "raw" / "fighter_profiles.json"
        if not raw_profiles_path.exists():
            print("Warning: Raw fighter profiles not found. Skipping physical attributes merge.")
            return

        try:
            with open(raw_profiles_path, 'r', encoding='utf-8') as f:
                raw_profiles = json.load(f)
            
            print(f"Loaded {len(raw_profiles)} raw profiles for merging")
            
            for pid, profile in self.profiles.items():
                if pid in raw_profiles:
                    raw = raw_profiles[pid]
                    profile["height"] = raw.get("height", "")
                    profile["weight"] = raw.get("weight", "")
                    profile["reach"] = raw.get("reach", "")
                    profile["stance"] = raw.get("stance", "")
                    profile["age"] = raw.get("age", "")
                    profile["birth_date"] = raw.get("birth_date", "")
                    profile["nickname"] = raw.get("nickname", "")
                    profile["team"] = raw.get("team", "")
        except Exception as e:
            print(f"Error merging physical attributes: {e}")

    def save_profiles(self, filename: str = "fighter_profiles.json"):
        """Save profiles to JSON"""
        output_path = PROCESSED_DIR / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, indent=2)
        print(f"Saved {len(self.profiles)} fighter profiles to {output_path}")


if __name__ == "__main__":
    builder = FighterProfileBuilder()
    df = builder.load_data()
    builder.build_profiles(df)
    builder.save_profiles()
