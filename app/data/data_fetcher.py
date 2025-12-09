"""
Data Fetcher Module
Fetches data from GoalServe MMA APIs
"""
import os
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MMADataFetcher:
    """Base class for fetching MMA data from GoalServe API"""
    
    def __init__(self, delay: float = None):
        """
        Initialize the fetcher
        
        Args:
            delay: Delay between API requests in seconds (to avoid rate limiting)
        """
        # Load configuration from environment
        self.api_key = os.getenv("GOALSERVE_API_KEY", "48cbeb0a39014dc2d6db08dd947404e4")
        self.base_url = os.getenv("GOALSERVE_BASE_URL", "http://www.goalserve.com/getfeed")
        self.delay = delay if delay is not None else float(os.getenv("API_REQUEST_DELAY", "0.5"))
        self.session = requests.Session()
    
    @property
    def BASE_URL(self):
        """Backward compatible property for BASE_URL"""
        return f"{self.base_url}/{self.api_key}/mma"
    
    def _fetch(self, endpoint: str, params: Optional[Dict] = None) -> Optional[ET.Element]:
        """
        Fetch data from API endpoint
        
        Args:
            endpoint: API endpoint (schedule, live, fighters, fighter)
            params: Optional query parameters
            
        Returns:
            XML Element root or None if error
        """
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(self.delay)
            return ET.fromstring(response.content)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None


class ScheduleFetcher(MMADataFetcher):
    """Fetches upcoming fight schedule"""
    
    def fetch_schedule(self) -> List[Dict[str, Any]]:
        """
        Fetch all scheduled fights
        
        Returns:
            List of fight dictionaries with event and match info
        """
        root = self._fetch("schedule")
        if root is None:
            return []
        
        fights = []
        for category in root.findall(".//category"):
            event_name = category.get("name", "")
            event_date = category.get("date", "")
            event_id = category.get("id", "")
            
            for match in category.findall("match"):
                localteam = match.find("localteam")
                awayteam = match.find("awayteam")
                
                fight = {
                    "event_name": event_name,
                    "event_date": event_date,
                    "event_id": event_id,
                    "match_id": match.get("id", ""),
                    "match_date": match.get("date", ""),
                    "match_time": match.get("time", ""),
                    "status": match.get("status", ""),
                    "ismain": match.get("ismain", "False") == "True",
                    "weight_class": match.get("type", ""),
                    "localteam_name": localteam.get("name", "") if localteam is not None else "",
                    "localteam_id": localteam.get("id", "") if localteam is not None else "",
                    "awayteam_name": awayteam.get("name", "") if awayteam is not None else "",
                    "awayteam_id": awayteam.get("id", "") if awayteam is not None else "",
                }
                fights.append(fight)
        
        return fights


class LiveFetcher(MMADataFetcher):
    """Fetches live and historical fight data with stats"""
    
    def fetch_live(self) -> List[Dict[str, Any]]:
        """Fetch current live fights"""
        return self._parse_fights(self._fetch("live"))
    
    def fetch_by_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Fetch fights for a specific date
        
        Args:
            date: Date in format dd.MM.yyyy
            
        Returns:
            List of fight dictionaries with full stats
        """
        return self._parse_fights(self._fetch("live", params={"date": date}))
    
    def _parse_fights(self, root: Optional[ET.Element]) -> List[Dict[str, Any]]:
        """Parse fight data from XML"""
        if root is None:
            return []
        
        fights = []
        for category in root.findall(".//category"):
            event_name = category.get("name", "")
            event_date = category.get("date", "")
            event_id = category.get("id", "")
            
            for match in category.findall("match"):
                fight = self._parse_match(match, event_name, event_date, event_id)
                if fight:
                    fights.append(fight)
        
        return fights
    
    def _parse_match(self, match: ET.Element, event_name: str, 
                     event_date: str, event_id: str) -> Optional[Dict[str, Any]]:
        """Parse single match element"""
        localteam = match.find("localteam")
        awayteam = match.find("awayteam")
        win_result = match.find("win_result")
        stats = match.find("stats")
        
        if localteam is None or awayteam is None:
            return None
        
        # Base fight info
        fight = {
            "event_name": event_name,
            "event_date": event_date,
            "event_id": event_id,
            "match_id": match.get("id", ""),
            "match_date": match.get("date", ""),
            "match_time": match.get("time", ""),
            "status": match.get("status", ""),
            "localteam_name": localteam.get("name", ""),
            "localteam_id": localteam.get("id", ""),
            "localteam_winner": localteam.get("winner", "False") == "True",
            "awayteam_name": awayteam.get("name", ""),
            "awayteam_id": awayteam.get("id", ""),
            "awayteam_winner": awayteam.get("winner", "False") == "True",
        }
        
        # Parse win result
        if win_result is not None:
            won_by = win_result.find("won_by")
            if won_by is not None:
                fight["win_type"] = won_by.get("type", "")
                fight["win_round"] = won_by.get("round", "")
                fight["win_minute"] = won_by.get("minute", "")
                
                ko = won_by.find("ko")
                if ko is not None:
                    fight["ko_type"] = ko.get("type", "")
                    fight["ko_target"] = ko.get("target", "")
                
                sub = won_by.find("sub")
                if sub is not None:
                    fight["sub_type"] = sub.get("type", "")
                
                points = won_by.find("points")
                if points is not None:
                    fight["points_score"] = points.get("score", "")
        
        # Parse fight statistics
        if stats is not None:
            for team, prefix in [(stats.find("localteam"), "local_"), 
                                 (stats.find("awayteam"), "away_")]:
                if team is not None:
                    # Strikes total
                    strikes_total = team.find("strikes_total")
                    if strikes_total is not None:
                        fight[f"{prefix}strikes_head"] = self._safe_int(strikes_total.get("head", "0"))
                        fight[f"{prefix}strikes_body"] = self._safe_int(strikes_total.get("body", "0"))
                        fight[f"{prefix}strikes_legs"] = self._safe_int(strikes_total.get("legs", "0"))
                    
                    # Power strikes
                    strikes_power = team.find("strikes_power")
                    if strikes_power is not None:
                        fight[f"{prefix}power_head"] = self._safe_int(strikes_power.get("head", "0"))
                        fight[f"{prefix}power_body"] = self._safe_int(strikes_power.get("body", "0"))
                        fight[f"{prefix}power_legs"] = self._safe_int(strikes_power.get("legs", "0"))
                    
                    # Takedowns
                    takedowns = team.find("takedowns")
                    if takedowns is not None:
                        fight[f"{prefix}takedowns_att"] = self._safe_int(takedowns.get("att", "0"))
                        fight[f"{prefix}takedowns_landed"] = self._safe_int(takedowns.get("landed", "0"))
                    
                    # Submissions
                    submissions = team.find("submissions")
                    if submissions is not None:
                        fight[f"{prefix}submissions"] = self._safe_int(submissions.get("total", "0"))
                    
                    # Control time
                    control_time = team.find("control_time")
                    if control_time is not None:
                        fight[f"{prefix}control_time"] = self._parse_time(control_time.get("total", "0:00"))
                    
                    # Knockdowns
                    knockdowns = team.find("knockdowns")
                    if knockdowns is not None:
                        fight[f"{prefix}knockdowns"] = self._safe_int(knockdowns.get("total", "0"))
        
        return fight
    
    @staticmethod
    def _safe_int(value: str) -> int:
        """Safely convert string to int"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    @staticmethod
    def _parse_time(time_str: str) -> int:
        """Parse time string (M:SS) to seconds"""
        try:
            parts = time_str.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return 0
        except:
            return 0


class FighterFetcher(MMADataFetcher):
    """Fetches fighter profile data"""
    
    def fetch_fighter(self, fighter_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch single fighter profile
        
        Args:
            fighter_id: Fighter ID from the API
            
        Returns:
            Fighter profile dictionary or None
        """
        root = self._fetch("fighter", params={"profile": fighter_id})
        if root is None:
            return None
        
        fighter = root.find(".//fighter")
        if fighter is None:
            return None
        
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
                # First record: win/loss/draw
                if record.get("win") is not None:
                    profile["wins"] = self._safe_int(record.get("win", "0"))
                    profile["losses"] = self._safe_int(record.get("loss", "0"))
                    profile["draws"] = self._safe_int(record.get("draw", "0"))
                
                # Second record: total wins and KO wins
                if record.get("total_wins") is not None:
                    profile["total_wins"] = self._safe_int(record.get("total_wins", "0"))
                    profile["ko_wins"] = self._safe_int(record.get("ko_wins", "0"))
                
                # Third record: KO losses and submissions
                if record.get("ko_loss") is not None:
                    profile["ko_losses"] = self._safe_int(record.get("ko_loss", "0"))
                    profile["sub_wins"] = self._safe_int(record.get("sub", "0"))
        
        return profile
    
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


class FightersListFetcher(MMADataFetcher):
    """Fetches list of all fighters"""
    
    def fetch_fighters_list(self) -> List[Dict[str, Any]]:
        """
        Fetch all fighters
        
        Returns:
            List of fighter dictionaries (id, name)
        """
        root = self._fetch("fighters")
        if root is None:
            return []
        
        fighters = []
        for category in root.findall(".//category"):
            for fighter in category.findall("fighter"):
                f_data = {
                    "fighter_id": fighter.get("id", ""),
                    "name": fighter.get("name", ""),
                }
                fighters.append(f_data)
        
        return fighters


# Convenience functions
def fetch_schedule() -> List[Dict[str, Any]]:
    """Fetch current fight schedule"""
    return ScheduleFetcher().fetch_schedule()


def fetch_fights_by_date(date: str) -> List[Dict[str, Any]]:
    """Fetch fights for a specific date (format: dd.MM.yyyy)"""
    return LiveFetcher().fetch_by_date(date)


def fetch_fighter_profile(fighter_id: str) -> Optional[Dict[str, Any]]:
    """Fetch fighter profile by ID"""
    return FighterFetcher().fetch_fighter(fighter_id)


def fetch_all_fighters() -> List[Dict[str, Any]]:
    """Fetch list of all fighters"""
    return FightersListFetcher().fetch_fighters_list()


if __name__ == "__main__":
    # Test the fetchers
    print("Testing Schedule Fetcher...")
    schedule = fetch_schedule()
    print(f"Found {len(schedule)} scheduled fights")
    if schedule:
        print(f"Sample: {schedule[0]}")
    
    print("\nTesting Live Fetcher...")
    fights = fetch_fights_by_date("22.11.2025")
    print(f"Found {len(fights)} fights")
    if fights:
        print(f"Sample: {fights[0]}")
