"""
Fighter Profile Fetcher
Fetches/updates fighter profiles from GoalServe API

Usage:
    python -m app.data.profile_fetcher [--full] [--missing-only]
"""

import os
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.getenv("GOALSERVE_API_KEY")
BASE_URL = os.getenv("GOALSERVE_BASE_URL")
REQUEST_DELAY = float(os.getenv("API_REQUEST_DELAY", "0.3"))

# Paths
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)

PROFILES_FILE = RAW_DIR / "fighter_profiles.json"
FIGHTER_IDS_FILE = RAW_DIR / "fighter_ids.json"


class FighterProfileFetcher:
    """Fetches fighter profiles from GoalServe API"""
    
    def __init__(self, delay: float = REQUEST_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.base_url = f"{BASE_URL}/{API_KEY}/mma"
    
    def fetch_fighters_list(self) -> List[Dict[str, str]]:
        """Fetch list of all fighters from API"""
        print("Fetching fighters list from API...")
        url = f"{self.base_url}/fighters"
        
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            fighters = []
            for category in root.findall(".//category"):
                for fighter in category.findall("fighter"):
                    fighters.append({
                        "fighter_id": fighter.get("id", ""),
                        "name": fighter.get("name", ""),
                    })
            
            print(f"Found {len(fighters)} fighters in API")
            return fighters
        except Exception as e:
            print(f"Error fetching fighters list: {e}")
            return []
    
    def fetch_fighter_profile(self, fighter_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single fighter profile"""
        url = f"{self.base_url}/fighter"
        params = {"profile": fighter_id}
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(self.delay)
            
            root = ET.fromstring(response.content)
            fighter = root.find(".//fighter")
            
            if fighter is None:
                return None
            
            # Parse profile
            profile = {
                "fighter_id": fighter.get("id", ""),
                "name": fighter.get("name", ""),
                "gender": self._get_text(fighter, "gender"),
                "birth_date": self._get_text(fighter, "birth_date"),
                "age": self._get_text(fighter, "age"),
                "height": self._get_text(fighter, "heigth"),  # Note: API typo
                "weight": self._get_text(fighter, "weigth"),  # Note: API typo
                "weight_class": self._get_text(fighter, "weightclass"),
                "team": self._get_text(fighter, "team"),
                "nickname": self._get_text(fighter, "nickname"),
                "stance": self._get_text(fighter, "stance"),
                "reach": self._get_text(fighter, "reach"),
            }
            
            # Parse records
            records = fighter.find("records")
            if records is not None:
                for record in records.findall("record"):
                    if record.get("win") is not None:
                        profile["wins"] = self._safe_int(record.get("win", "0"))
                        profile["losses"] = self._safe_int(record.get("loss", "0"))
                        profile["draws"] = self._safe_int(record.get("draw", "0"))
                    if record.get("total_wins") is not None:
                        profile["total_wins"] = self._safe_int(record.get("total_wins", "0"))
                        profile["ko_wins"] = self._safe_int(record.get("ko_wins", "0"))
                    if record.get("ko_loss") is not None:
                        profile["ko_losses"] = self._safe_int(record.get("ko_loss", "0"))
                        profile["sub_wins"] = self._safe_int(record.get("sub", "0"))
            
            return profile
        except Exception as e:
            print(f"Error fetching profile {fighter_id}: {e}")
            return None
    
    @staticmethod
    def _get_text(element: ET.Element, tag: str) -> str:
        """Get text content of child element"""
        child = element.find(tag)
        return child.text.strip() if child is not None and child.text else ""
    
    @staticmethod
    def _safe_int(value: str) -> int:
        """Safely convert string to int"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def load_existing_profiles(self) -> Dict[str, Dict]:
        """Load existing profiles"""
        if PROFILES_FILE.exists():
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_fighter_ids(self) -> List[str]:
        """Load fighter IDs"""
        if FIGHTER_IDS_FILE.exists():
            with open(FIGHTER_IDS_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def save_profiles(self, profiles: Dict[str, Dict]):
        """Save profiles to JSON"""
        with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
    
    def save_fighter_ids(self, ids: List[str]):
        """Save fighter IDs"""
        with open(FIGHTER_IDS_FILE, 'w') as f:
            json.dump(ids, f)
    
    def fetch_all_profiles(self, missing_only: bool = True):
        """Fetch all fighter profiles"""
        print("=" * 60)
        print("FIGHTER PROFILE FETCHER")
        print("=" * 60)
        
        # Load existing
        profiles = self.load_existing_profiles()
        fighter_ids = self.load_fighter_ids()
        
        print(f"Existing profiles: {len(profiles)}")
        print(f"Known fighter IDs: {len(fighter_ids)}")
        
        # Optionally refresh fighter list from API
        if not fighter_ids or not missing_only:
            api_fighters = self.fetch_fighters_list()
            if api_fighters:
                new_ids = [f["fighter_id"] for f in api_fighters if f.get("fighter_id")]
                # Merge with existing
                all_ids = list(set(fighter_ids + new_ids))
                fighter_ids = all_ids
                self.save_fighter_ids(fighter_ids)
                print(f"Updated fighter IDs: {len(fighter_ids)}")
        
        # Determine which to fetch
        if missing_only:
            to_fetch = [fid for fid in fighter_ids if fid not in profiles]
            print(f"Fetching {len(to_fetch)} missing profiles...")
        else:
            to_fetch = fighter_ids
            print(f"Fetching all {len(to_fetch)} profiles...")
        
        # Fetch profiles
        new_count = 0
        for i, fid in enumerate(to_fetch):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(to_fetch)} - New: {new_count}")
                self.save_profiles(profiles)  # Save periodically
            
            profile = self.fetch_fighter_profile(fid)
            if profile:
                profiles[fid] = profile
                new_count += 1
        
        # Final save
        self.save_profiles(profiles)
        
        print("=" * 60)
        print(f"COMPLETE! Total profiles: {len(profiles)} (New: {new_count})")
        print(f"Saved to: {PROFILES_FILE}")
        print("=" * 60)
        
        return profiles


def fetch_profiles(missing_only: bool = True):
    """Main entry point"""
    fetcher = FighterProfileFetcher()
    return fetcher.fetch_all_profiles(missing_only=missing_only)


if __name__ == "__main__":
    import sys
    
    missing_only = "--full" not in sys.argv
    fetch_profiles(missing_only=missing_only)
