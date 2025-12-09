"""
Fighter Profiles Module
Fetches and stores fighter profiles locally for frontend display
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from app.data.data_fetcher import FighterFetcher


OUTPUT_DIR = Path(__file__).parent / "raw"
PROFILES_FILE = OUTPUT_DIR / "fighter_profiles.json"
FIGHTER_IDS_FILE = OUTPUT_DIR / "fighter_ids.json"


class FighterProfileCollector:
    """Collects and stores fighter profiles"""
    
    def __init__(self):
        self.fetcher = FighterFetcher(delay=0.3)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_fighter_ids(self) -> List[str]:
        """Load fighter IDs from data collection"""
        if FIGHTER_IDS_FILE.exists():
            with open(FIGHTER_IDS_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load existing profiles"""
        if PROFILES_FILE.exists():
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_profiles(self, profiles: Dict[str, Dict[str, Any]]):
        """Save profiles to JSON"""
        with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
    
    def collect_profiles(self, fighter_ids: Optional[List[str]] = None):
        """
        Collect fighter profiles
        
        Args:
            fighter_ids: List of fighter IDs to collect. If None, uses saved IDs.
        """
        if fighter_ids is None:
            fighter_ids = self.load_fighter_ids()
            
            # If no saved IDs, try to fetch from API
            if not fighter_ids:
                print("No saved fighter IDs found. Fetching from API...")
                try:
                    from app.data.data_fetcher import fetch_all_fighters
                    all_fighters = fetch_all_fighters()
                    fighter_ids = [f['fighter_id'] for f in all_fighters if f.get('fighter_id')]
                    
                    # Save these IDs for future use
                    with open(FIGHTER_IDS_FILE, 'w') as f:
                        json.dump(fighter_ids, f)
                    print(f"Fetched and saved {len(fighter_ids)} fighter IDs from API")
                except Exception as e:
                    print(f"Error fetching fighter list: {e}")
        
        if not fighter_ids:
            print("No fighter IDs found. Run data_collector.py first or ensure API is accessible.")
            return
        
        profiles = self.load_profiles()
        new_count = 0
        
        print(f"Collecting profiles for {len(fighter_ids)} fighters...")
        print(f"Existing profiles: {len(profiles)}")
        
        for i, fighter_id in enumerate(fighter_ids):
            # Skip if already collected
            if fighter_id in profiles:
                continue
            
            profile = self.fetcher.fetch_fighter(fighter_id)
            if profile:
                profiles[fighter_id] = profile
                new_count += 1
                
                if new_count % 10 == 0:
                    print(f"Progress: {i+1}/{len(fighter_ids)} - New: {new_count}")
                    self.save_profiles(profiles)
        
        self.save_profiles(profiles)
        print(f"Done! Total profiles: {len(profiles)} (New: {new_count})")
    
    def get_profile(self, fighter_id: str) -> Optional[Dict[str, Any]]:
        """Get a single fighter profile"""
        profiles = self.load_profiles()
        return profiles.get(fighter_id)
    
    def search_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search profiles by name (case-insensitive)"""
        profiles = self.load_profiles()
        name_lower = name.lower()
        return [
            p for p in profiles.values()
            if name_lower in p.get("name", "").lower()
        ]


def get_fighter_profile(fighter_id: str) -> Optional[Dict[str, Any]]:
    """Get fighter profile by ID - looks up from profiles or historical data"""
    collector = FighterProfileCollector()
    profile = collector.get_profile(fighter_id)
    
    if profile:
        return profile
    
    # Fallback: Try to get fighter name from historical data
    try:
        import pandas as pd
        fights_file = OUTPUT_DIR / "historical_fights.csv"
        if fights_file.exists():
            df = pd.read_csv(fights_file)
            
            # Look in localteam
            local_match = df[df['localteam_id'] == int(fighter_id)]
            if not local_match.empty:
                return {
                    'fighter_id': fighter_id,
                    'name': local_match.iloc[0]['localteam_name']
                }
            
            # Look in awayteam
            away_match = df[df['awayteam_id'] == int(fighter_id)]
            if not away_match.empty:
                return {
                    'fighter_id': fighter_id,
                    'name': away_match.iloc[0]['awayteam_name']
                }
    except Exception:
        pass
    
    return None


def search_fighters(name: str) -> List[Dict[str, Any]]:
    """Search fighters by name"""
    return FighterProfileCollector().search_by_name(name)


def collect_all_profiles():
    """Collect all fighter profiles (run after data_collector.py)"""
    collector = FighterProfileCollector()
    collector.collect_profiles()


if __name__ == "__main__":
    collect_all_profiles()
