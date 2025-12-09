r'''
Historical Data Collector - STANDALONE SCRIPT
This script collects historical UFC fight data from 2010 to current.
RUN THIS MANUALLY - it may take several hours to complete.

Usage:
    cd c:\Users\NiloySannyal\UFC
    python -m app.data.data_collector

Progress is saved automatically - you can stop and resume at any time.
'''

import os
import csv
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from pathlib import Path

from app.data.data_fetcher import LiveFetcher


# Configuration
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime.now()
OUTPUT_DIR = Path(__file__).parent / "raw"
FIGHTS_FILE = OUTPUT_DIR / "historical_fights.csv"
PROGRESS_FILE = OUTPUT_DIR / "collection_progress.json"
FIGHTER_IDS_FILE = OUTPUT_DIR / "fighter_ids.json"

# CSV columns for fight data
FIGHT_COLUMNS = [
    "event_name", "event_date", "event_id", "match_id", "match_date", "match_time",
    "status", "localteam_name", "localteam_id", "localteam_winner",
    "awayteam_name", "awayteam_id", "awayteam_winner",
    "win_type", "win_round", "win_minute", "ko_type", "ko_target", "sub_type", "points_score",
    "local_strikes_head", "local_strikes_body", "local_strikes_legs",
    "local_power_head", "local_power_body", "local_power_legs",
    "local_takedowns_att", "local_takedowns_landed", "local_submissions",
    "local_control_time", "local_knockdowns",
    "away_strikes_head", "away_strikes_body", "away_strikes_legs",
    "away_power_head", "away_power_body", "away_power_legs",
    "away_takedowns_att", "away_takedowns_landed", "away_submissions",
    "away_control_time", "away_knockdowns"
]


class HistoricalDataCollector:
    """Collects historical fight data with progress tracking"""
    
    def __init__(self):
        self.fetcher = LiveFetcher(delay=0.3)  # 300ms delay between requests
        self.fights_collected = 0
        self.fighter_ids: Set[str] = set()
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_progress(self) -> datetime:
        """Load last processed date from progress file"""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
                return datetime.strptime(data["last_date"], "%d.%m.%Y")
        return START_DATE
    
    def save_progress(self, date: datetime):
        """Save current progress"""
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({"last_date": date.strftime("%d.%m.%Y")}, f)
    
    def load_fighter_ids(self) -> Set[str]:
        """Load collected fighter IDs"""
        if FIGHTER_IDS_FILE.exists():
            with open(FIGHTER_IDS_FILE, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_fighter_ids(self):
        """Save collected fighter IDs"""
        with open(FIGHTER_IDS_FILE, 'w') as f:
            json.dump(list(self.fighter_ids), f)
    
    def init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not FIGHTS_FILE.exists():
            with open(FIGHTS_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=FIGHT_COLUMNS)
                writer.writeheader()
    
    def append_fights(self, fights: List[Dict[str, Any]]):
        """Append fights to CSV file"""
        with open(FIGHTS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIGHT_COLUMNS)
            for fight in fights:
                # Only include completed fights with stats
                if fight.get("status") == "Final":
                    # Extract fighter IDs
                    if fight.get("localteam_id"):
                        self.fighter_ids.add(fight["localteam_id"])
                    if fight.get("awayteam_id"):
                        self.fighter_ids.add(fight["awayteam_id"])
                    
                    # Write fight data
                    row = {col: fight.get(col, "") for col in FIGHT_COLUMNS}
                    writer.writerow(row)
                    self.fights_collected += 1
    
    def collect(self):
        """Main collection loop"""
        print("=" * 60)
        print("UFC Historical Data Collector")
        print("=" * 60)
        
        # Load progress
        start = self.load_progress()
        self.fighter_ids = self.load_fighter_ids()
        self.init_csv()
        
        # Count existing fights
        if FIGHTS_FILE.exists():
            with open(FIGHTS_FILE, 'r', encoding='utf-8') as f:
                self.fights_collected = sum(1 for _ in f) - 1  # Exclude header
        
        print(f"Starting from: {start.strftime('%d.%m.%Y')}")
        print(f"Ending at: {END_DATE.strftime('%d.%m.%Y')}")
        print(f"Fights already collected: {self.fights_collected}")
        print(f"Fighter IDs collected: {len(self.fighter_ids)}")
        print("-" * 60)
        
        current = start
        dates_processed = 0
        
        try:
            while current <= END_DATE:
                date_str = current.strftime("%d.%m.%Y")
                
                # Fetch fights for this date
                fights = self.fetcher.fetch_by_date(date_str)
                
                if fights:
                    self.append_fights(fights)
                    print(f"[{date_str}] Found {len(fights)} fights (Total: {self.fights_collected})")
                
                # Save progress every 7 days
                dates_processed += 1
                if dates_processed % 7 == 0:
                    self.save_progress(current)
                    self.save_fighter_ids()
                
                # Move to next day
                current += timedelta(days=1)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving progress...")
        
        finally:
            # Save final progress
            self.save_progress(current)
            self.save_fighter_ids()
            
            print("\n" + "=" * 60)
            print("Collection Complete!")
            print(f"Total fights collected: {self.fights_collected}")
            print(f"Total fighter IDs: {len(self.fighter_ids)}")
            print(f"Data saved to: {FIGHTS_FILE}")
            print(f"Fighter IDs saved to: {FIGHTER_IDS_FILE}")
            print("=" * 60)


def main():
    """Main entry point"""
    collector = HistoricalDataCollector()
    collector.collect()


if __name__ == "__main__":
    main()
