"""
Enhanced Fighter Profile Builder
Creates comprehensive fighter profiles by merging:
1. Physical attributes from GoalServe API
2. Statistical data calculated from historical fights

Usage:
    python -m app.pipeline.enhanced_profile_builder
"""

import os
import re
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Output files
ENHANCED_PROFILES_FILE = PROCESSED_DIR / "enhanced_fighter_profiles.json"
PHYSICAL_ATTRIBUTES_FILE = PROCESSED_DIR / "fighter_physical_attributes.json"


class PhysicalAttributeParser:
    """Parses and normalizes physical attributes from API data"""
    
    # Weight class medians for imputation (in inches for height, inches for reach)
    WEIGHT_CLASS_MEDIANS = {
        "Strawweight": {"height": 64, "reach": 64},
        "Flyweight": {"height": 66, "reach": 66},
        "Bantamweight": {"height": 67, "reach": 68},
        "Featherweight": {"height": 69, "reach": 70},
        "Lightweight": {"height": 70, "reach": 72},
        "Welterweight": {"height": 72, "reach": 74},
        "Middleweight": {"height": 73, "reach": 75},
        "Light Heavyweight": {"height": 74, "reach": 76},
        "Heavyweight": {"height": 76, "reach": 78},
    }
    DEFAULT_HEIGHT = 70
    DEFAULT_REACH = 72
    DEFAULT_AGE = 32
    
    @staticmethod
    def parse_height(height_str: str) -> Optional[float]:
        """Parse height string (e.g., '5'9"' or '5&#x27; 9&quot;') to inches"""
        if not height_str:
            return None
        
        # Clean HTML entities
        height_str = height_str.replace("&#x27;", "'").replace("&quot;", '"')
        height_str = height_str.replace("&amp;", "&").replace("&#39;", "'")
        
        # Pattern: 5'9" or 5' 9" or 5'9
        match = re.search(r"(\d+)['\s]+(\d+)", height_str)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return feet * 12 + inches
        
        # Pattern: just inches like "69"
        match = re.search(r"^(\d+)$", height_str.strip())
        if match:
            return float(match.group(1))
        
        return None
    
    @staticmethod
    def parse_reach(reach_str: str) -> Optional[float]:
        """Parse reach string (e.g., '74"' or '74&quot;') to inches"""
        if not reach_str:
            return None
        
        # Clean HTML entities
        reach_str = reach_str.replace("&quot;", '"').replace("&#x27;", "'")
        reach_str = reach_str.replace("&amp;", "&")
        
        # Pattern: number followed by optional "
        match = re.search(r"(\d+(?:\.\d+)?)", reach_str)
        if match:
            return float(match.group(1))
        
        return None
    
    @staticmethod
    def parse_weight(weight_str: str) -> Optional[float]:
        """Parse weight string (e.g., '156 lbs') to pounds"""
        if not weight_str:
            return None
        
        match = re.search(r"(\d+(?:\.\d+)?)", weight_str)
        if match:
            return float(match.group(1))
        
        return None
    
    @staticmethod
    def parse_age(age_str: str, birth_date_str: str = None) -> Optional[int]:
        """Parse age from age string or calculate from birth date"""
        if age_str and age_str.strip():
            try:
                return int(age_str)
            except:
                pass
        
        if birth_date_str:
            try:
                # Try common formats
                for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"]:
                    try:
                        birth = datetime.strptime(birth_date_str, fmt)
                        today = datetime.now()
                        age = today.year - birth.year
                        if today.month < birth.month or (today.month == birth.month and today.day < birth.day):
                            age -= 1
                        return age
                    except:
                        continue
            except:
                pass
        
        return None
    
    @staticmethod
    def normalize_stance(stance_str: str) -> str:
        """Normalize stance to standard values"""
        if not stance_str:
            return "Orthodox"  # Default
        
        stance_lower = stance_str.lower().strip()
        if "south" in stance_lower:
            return "Southpaw"
        elif "switch" in stance_lower:
            return "Switch"
        else:
            return "Orthodox"
    
    @classmethod
    def get_imputed_height(cls, weight_class: str) -> float:
        """Get imputed height based on weight class"""
        wc = weight_class or ""
        for wc_name, medians in cls.WEIGHT_CLASS_MEDIANS.items():
            if wc_name.lower() in wc.lower():
                return medians["height"]
        return cls.DEFAULT_HEIGHT
    
    @classmethod
    def get_imputed_reach(cls, weight_class: str, height: float = None) -> float:
        """Get imputed reach (reach ≈ height for most fighters)"""
        if height:
            return height  # Reasonable approximation
        
        wc = weight_class or ""
        for wc_name, medians in cls.WEIGHT_CLASS_MEDIANS.items():
            if wc_name.lower() in wc.lower():
                return medians["reach"]
        return cls.DEFAULT_REACH


class FighterStatsCalculator:
    """Calculates comprehensive fight statistics from historical data"""
    
    def __init__(self, fights_df: pd.DataFrame):
        self.fights = fights_df.copy()
        self._preprocess_fights()
    
    def _preprocess_fights(self):
        """Preprocess fights dataframe"""
        # Ensure date column
        if 'match_date' in self.fights.columns:
            self.fights['fight_date'] = pd.to_datetime(
                self.fights['match_date'], format='%d.%m.%Y', errors='coerce'
            )
        
        # Filter completed fights
        self.fights = self.fights[self.fights['status'] == 'Final'].copy()
        
        # Sort by date
        self.fights = self.fights.sort_values('fight_date')
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = self.fights.select_dtypes(include=[np.number]).columns
        self.fights[numeric_cols] = self.fights[numeric_cols].fillna(0)
    
    def calculate_fighter_stats(self, fighter_id: str) -> Dict[str, Any]:
        """Calculate all statistics for a fighter"""
        fid = str(fighter_id)
        
        # Get all fights for this fighter (as local or away)
        local_fights = self.fights[self.fights['localteam_id'].astype(str) == fid]
        away_fights = self.fights[self.fights['awayteam_id'].astype(str) == fid]
        
        if len(local_fights) == 0 and len(away_fights) == 0:
            return self._empty_stats()
        
        # Combine fight data from both perspectives
        all_fight_stats = []
        
        # Process local fights
        for _, fight in local_fights.iterrows():
            stats = self._extract_fighter_stats(fight, is_local=True)
            all_fight_stats.append(stats)
        
        # Process away fights
        for _, fight in away_fights.iterrows():
            stats = self._extract_fighter_stats(fight, is_local=False)
            all_fight_stats.append(stats)
        
        # Sort by date
        all_fight_stats.sort(key=lambda x: x['date'])
        
        return self._aggregate_stats(all_fight_stats, fid)
    
    def _extract_fighter_stats(self, fight: pd.Series, is_local: bool) -> Dict:
        """Extract stats for one fighter from a fight"""
        prefix = "local_" if is_local else "away_"
        opp_prefix = "away_" if is_local else "local_"
        
        # Determine win/loss
        local_winner = fight.get('localteam_winner', False)
        if isinstance(local_winner, str):
            local_winner = local_winner.lower() == 'true'
        
        won = local_winner if is_local else not local_winner
        
        # Win method
        win_type = str(fight.get('win_type', '')).upper()
        win_round = int(fight.get('win_round', 0) or 0)
        
        # Normalize win type
        if 'KO' in win_type or 'TKO' in win_type:
            method = 'KO'
        elif 'SUB' in win_type:
            method = 'SUB'
        elif 'DEC' in win_type or 'POINTS' in win_type or win_type in ['U DEC', 'S DEC', 'M DEC']:
            method = 'DEC'
        else:
            method = 'OTHER'
        
        return {
            'date': fight.get('fight_date', pd.NaT),
            'won': won,
            'method': method,
            'round': win_round,
            
            # Strikes
            'strikes_head': int(fight.get(f'{prefix}strikes_head', 0) or 0),
            'strikes_body': int(fight.get(f'{prefix}strikes_body', 0) or 0),
            'strikes_legs': int(fight.get(f'{prefix}strikes_legs', 0) or 0),
            
            # Power strikes
            'power_head': int(fight.get(f'{prefix}power_head', 0) or 0),
            'power_body': int(fight.get(f'{prefix}power_body', 0) or 0),
            'power_legs': int(fight.get(f'{prefix}power_legs', 0) or 0),
            
            # Strikes absorbed
            'strikes_head_absorbed': int(fight.get(f'{opp_prefix}strikes_head', 0) or 0),
            'strikes_body_absorbed': int(fight.get(f'{opp_prefix}strikes_body', 0) or 0),
            'strikes_legs_absorbed': int(fight.get(f'{opp_prefix}strikes_legs', 0) or 0),
            
            # Takedowns
            'td_landed': int(fight.get(f'{prefix}takedowns_landed', 0) or 0),
            'td_attempted': int(fight.get(f'{prefix}takedowns_att', 0) or 0),
            'td_absorbed': int(fight.get(f'{opp_prefix}takedowns_landed', 0) or 0),
            
            # Submissions
            'submissions': int(fight.get(f'{prefix}submissions', 0) or 0),
            
            # Control time
            'control_time': int(fight.get(f'{prefix}control_time', 0) or 0),
            
            # Knockdowns
            'knockdowns': int(fight.get(f'{prefix}knockdowns', 0) or 0),
            'knockdowns_absorbed': int(fight.get(f'{opp_prefix}knockdowns', 0) or 0),
        }
    
    def _aggregate_stats(self, fight_stats: List[Dict], fighter_id: str) -> Dict[str, Any]:
        """Aggregate fight statistics into profile format"""
        n_fights = len(fight_stats)
        if n_fights == 0:
            return self._empty_stats()
        
        # Record
        wins = sum(1 for f in fight_stats if f['won'])
        losses = n_fights - wins
        
        # Win methods
        ko_wins = sum(1 for f in fight_stats if f['won'] and f['method'] == 'KO')
        sub_wins = sum(1 for f in fight_stats if f['won'] and f['method'] == 'SUB')
        dec_wins = sum(1 for f in fight_stats if f['won'] and f['method'] == 'DEC')
        
        # Loss methods
        ko_losses = sum(1 for f in fight_stats if not f['won'] and f['method'] == 'KO')
        sub_losses = sum(1 for f in fight_stats if not f['won'] and f['method'] == 'SUB')
        dec_losses = sum(1 for f in fight_stats if not f['won'] and f['method'] == 'DEC')
        
        # First round finishes
        r1_finishes = sum(1 for f in fight_stats if f['won'] and f['method'] in ['KO', 'SUB'] and f['round'] == 1)
        
        # Calculate averages
        def avg(key): 
            return sum(f[key] for f in fight_stats) / n_fights
        
        def total(key): 
            return sum(f[key] for f in fight_stats)
        
        # Total strikes
        total_strikes = [f['strikes_head'] + f['strikes_body'] + f['strikes_legs'] for f in fight_stats]
        total_absorbed = [f['strikes_head_absorbed'] + f['strikes_body_absorbed'] + f['strikes_legs_absorbed'] for f in fight_stats]
        
        # Estimated fight time (5 min per round, approximate)
        total_fight_time_mins = sum(f['round'] * 5 for f in fight_stats if f['round'] > 0)
        if total_fight_time_mins == 0:
            total_fight_time_mins = n_fights * 15  # Default 3 rounds
        
        # Strike ratios
        all_head = total('strikes_head')
        all_body = total('strikes_body')
        all_legs = total('strikes_legs')
        all_strikes = all_head + all_body + all_legs
        
        head_ratio = all_head / all_strikes if all_strikes > 0 else 0.5
        body_ratio = all_body / all_strikes if all_strikes > 0 else 0.25
        legs_ratio = all_legs / all_strikes if all_strikes > 0 else 0.25
        
        # Takedown accuracy
        td_landed_total = total('td_landed')
        td_attempted_total = total('td_attempted')
        td_acc = td_landed_total / td_attempted_total if td_attempted_total > 0 else 0.0
        
        # Takedown defense
        td_absorbed_total = total('td_absorbed')
        td_def_attempts = td_absorbed_total + sum(1 for _ in fight_stats)  # Rough estimate
        td_def = 1 - (td_absorbed_total / max(td_def_attempts, 1))
        
        # Recent form (last 3 fights)
        last_3 = fight_stats[-3:] if len(fight_stats) >= 3 else fight_stats
        l3_wins = sum(1 for f in last_3 if f['won'])
        l3_strikes = sum(f['strikes_head'] + f['strikes_body'] + f['strikes_legs'] for f in last_3) / len(last_3) if last_3 else 0
        l3_td = sum(f['td_landed'] for f in last_3) / len(last_3) if last_3 else 0
        l3_finishes = sum(1 for f in last_3 if f['won'] and f['method'] in ['KO', 'SUB'])
        
        # Current streak
        streak = 0
        for f in reversed(fight_stats):
            if f['won']:
                if streak >= 0:
                    streak += 1
                else:
                    break
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    break
        
        # Form score (weighted recent results)
        weights = [0.5, 0.3, 0.2][:len(last_3)]
        form_score = sum(
            (1 if last_3[-(i+1)]['won'] else 0) * weights[i] 
            for i in range(len(weights))
        ) / sum(weights) if weights else 0
        
        # Last fight date
        last_fight_date = fight_stats[-1]['date'] if fight_stats else None
        if last_fight_date and pd.notna(last_fight_date):
            last_fight_date = str(last_fight_date.date()) if hasattr(last_fight_date, 'date') else str(last_fight_date)[:10]
        else:
            last_fight_date = None
        
        return {
            "record": {
                "wins": wins,
                "losses": losses,
                "draws": 0,
                "ko_wins": ko_wins,
                "sub_wins": sub_wins,
                "dec_wins": dec_wins,
                "ko_losses": ko_losses,
                "sub_losses": sub_losses,
                "dec_losses": dec_losses,
            },
            "strikes": {
                # Per fight averages
                "total_landed_avg": round(sum(total_strikes) / n_fights, 1),
                "total_absorbed_avg": round(sum(total_absorbed) / n_fights, 1),
                "head_landed_avg": round(avg('strikes_head'), 1),
                "body_landed_avg": round(avg('strikes_body'), 1),
                "legs_landed_avg": round(avg('strikes_legs'), 1),
                "power_head_landed_avg": round(avg('power_head'), 1),
                "power_body_landed_avg": round(avg('power_body'), 1),
                "power_legs_landed_avg": round(avg('power_legs'), 1),
                "head_absorbed_avg": round(avg('strikes_head_absorbed'), 1),
                "body_absorbed_avg": round(avg('strikes_body_absorbed'), 1),
                "legs_absorbed_avg": round(avg('strikes_legs_absorbed'), 1),
                # Rate-based
                "slpm": round(sum(total_strikes) / total_fight_time_mins, 2) if total_fight_time_mins > 0 else 0,
                "sapm": round(sum(total_absorbed) / total_fight_time_mins, 2) if total_fight_time_mins > 0 else 0,
                # Ratios
                "head_ratio": round(head_ratio, 2),
                "body_ratio": round(body_ratio, 2),
                "legs_ratio": round(legs_ratio, 2),
            },
            "grappling": {
                "td_landed_avg": round(avg('td_landed'), 1),
                "td_attempted_avg": round(avg('td_attempted'), 1),
                "td_absorbed_avg": round(avg('td_absorbed'), 1),
                "td_avg_15m": round(td_landed_total / (total_fight_time_mins / 15), 2) if total_fight_time_mins > 0 else 0,
                "sub_avg_15m": round(total('submissions') / (total_fight_time_mins / 15), 2) if total_fight_time_mins > 0 else 0,
                "td_acc": round(td_acc, 2),
                "td_def": round(td_def, 2),
                "ctrl_time_avg": round(avg('control_time'), 1),
                "submissions_avg": round(avg('submissions'), 1),
            },
            "knockdowns": {
                "kd_landed_avg": round(avg('knockdowns'), 2),
                "kd_absorbed_avg": round(avg('knockdowns_absorbed'), 2),
            },
            "performance": {
                "win_rate": round(wins / n_fights, 2) if n_fights > 0 else 0,
                "finish_rate": round((ko_wins + sub_wins) / wins, 2) if wins > 0 else 0,
                "ko_rate": round(ko_wins / wins, 2) if wins > 0 else 0,
                "sub_rate": round(sub_wins / wins, 2) if wins > 0 else 0,
                "dec_rate": round(dec_wins / wins, 2) if wins > 0 else 0,
                "avg_fight_time": round(total_fight_time_mins * 60 / n_fights) if n_fights > 0 else 0,
                "first_round_finish": round(r1_finishes / wins, 2) if wins > 0 else 0,
                "ko_loss_rate": round(ko_losses / losses, 2) if losses > 0 else 0,
                "sub_loss_rate": round(sub_losses / losses, 2) if losses > 0 else 0,
            },
            "recent": {
                "l3_wins": l3_wins,
                "l3_strikes_avg": round(l3_strikes, 1),
                "l3_td_avg": round(l3_td, 1),
                "l3_finish_count": l3_finishes,
                "current_streak": streak,
                "form_score": round(form_score, 2),
            },
            "total_fights": n_fights,
            "last_fight_date": last_fight_date,
        }
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty stats structure"""
        return {
            "record": {"wins": 0, "losses": 0, "draws": 0, "ko_wins": 0, "sub_wins": 0, "dec_wins": 0, "ko_losses": 0, "sub_losses": 0, "dec_losses": 0},
            "strikes": {"total_landed_avg": 0, "total_absorbed_avg": 0, "head_landed_avg": 0, "body_landed_avg": 0, "legs_landed_avg": 0, "power_head_landed_avg": 0, "power_body_landed_avg": 0, "power_legs_landed_avg": 0, "head_absorbed_avg": 0, "body_absorbed_avg": 0, "legs_absorbed_avg": 0, "slpm": 0, "sapm": 0, "head_ratio": 0.5, "body_ratio": 0.25, "legs_ratio": 0.25},
            "grappling": {"td_landed_avg": 0, "td_attempted_avg": 0, "td_absorbed_avg": 0, "td_avg_15m": 0, "sub_avg_15m": 0, "td_acc": 0, "td_def": 0.5, "ctrl_time_avg": 0, "submissions_avg": 0},
            "knockdowns": {"kd_landed_avg": 0, "kd_absorbed_avg": 0},
            "performance": {"win_rate": 0, "finish_rate": 0, "ko_rate": 0, "sub_rate": 0, "dec_rate": 0, "avg_fight_time": 0, "first_round_finish": 0, "ko_loss_rate": 0, "sub_loss_rate": 0},
            "recent": {"l3_wins": 0, "l3_strikes_avg": 0, "l3_td_avg": 0, "l3_finish_count": 0, "current_streak": 0, "form_score": 0},
            "total_fights": 0,
            "last_fight_date": None,
        }


class EnhancedProfileBuilder:
    """Main class to build enhanced fighter profiles"""
    
    def __init__(self):
        self.parser = PhysicalAttributeParser()
        self.profiles: Dict[str, Dict] = {}
        self.physical_data: Dict[str, Dict] = {}
        
    def load_raw_profiles(self) -> Dict[str, Dict]:
        """Load raw fighter profiles from API cache"""
        raw_file = RAW_DIR / "fighter_profiles.json"
        if raw_file.exists():
            with open(raw_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_historical_fights(self) -> pd.DataFrame:
        """Load historical fights data"""
        fights_file = RAW_DIR / "historical_fights.csv"
        if fights_file.exists():
            return pd.read_csv(fights_file, encoding='utf-8')
        raise FileNotFoundError(f"Historical fights not found: {fights_file}")
    
    def load_fighter_ids(self) -> List[str]:
        """Load all fighter IDs"""
        ids_file = RAW_DIR / "fighter_ids.json"
        if ids_file.exists():
            with open(ids_file, 'r') as f:
                return json.load(f)
        return []
    
    def extract_physical_attributes(self, raw_profiles: Dict) -> Dict[str, Dict]:
        """Extract and parse physical attributes from raw profiles"""
        physical_data = {}
        
        for fid, profile in raw_profiles.items():
            height = self.parser.parse_height(profile.get('height', ''))
            reach = self.parser.parse_reach(profile.get('reach', ''))
            weight = self.parser.parse_weight(profile.get('weight', ''))
            age = self.parser.parse_age(profile.get('age', ''), profile.get('birth_date', ''))
            stance = self.parser.normalize_stance(profile.get('stance', ''))
            weight_class = profile.get('weight_class', '')
            
            # Impute missing values
            if height is None:
                height = self.parser.get_imputed_height(weight_class)
            if reach is None:
                reach = self.parser.get_imputed_reach(weight_class, height)
            if age is None:
                age = self.parser.DEFAULT_AGE
            
            physical_data[fid] = {
                'height_inches': height,
                'reach_inches': reach,
                'weight_lbs': weight,
                'age': age,
                'stance': stance,
                'weight_class': weight_class,
                'team': profile.get('team', ''),
                'nickname': profile.get('nickname', ''),
                'name': profile.get('name', ''),
            }
        
        return physical_data
    
    def build_profiles(self) -> Dict[str, Dict]:
        """Build complete enhanced profiles"""
        print("=" * 60)
        print("ENHANCED FIGHTER PROFILE BUILDER")
        print("=" * 60)
        
        # Step 1: Load raw profiles
        print("\n[1/4] Loading raw fighter profiles...")
        raw_profiles = self.load_raw_profiles()
        print(f"      Loaded {len(raw_profiles)} raw profiles from API cache")
        
        # Step 2: Extract physical attributes
        print("\n[2/4] Extracting and parsing physical attributes...")
        self.physical_data = self.extract_physical_attributes(raw_profiles)
        print(f"      Parsed {len(self.physical_data)} physical profiles")
        
        # Save physical attributes separately
        with open(PHYSICAL_ATTRIBUTES_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.physical_data, f, indent=2)
        print(f"      Saved to {PHYSICAL_ATTRIBUTES_FILE}")
        
        # Step 3: Load historical fights and calculate stats
        print("\n[3/4] Calculating fight statistics from historical data...")
        fights_df = self.load_historical_fights()
        print(f"      Loaded {len(fights_df)} historical fights")
        
        stats_calc = FighterStatsCalculator(fights_df)
        
        # Get all fighter IDs
        fighter_ids = self.load_fighter_ids()
        print(f"      Processing {len(fighter_ids)} fighters...")
        
        # Step 4: Build enhanced profiles
        print("\n[4/4] Building enhanced profiles...")
        for i, fid in enumerate(fighter_ids):
            if (i + 1) % 500 == 0:
                print(f"      Progress: {i + 1}/{len(fighter_ids)}")
            
            # Get physical data
            physical = self.physical_data.get(fid, {})
            
            # Calculate fight stats
            fight_stats = stats_calc.calculate_fighter_stats(fid)
            
            # Get name from physical data or historical fights
            name = physical.get('name', '')
            if not name:
                # Try to get from historical fights
                local_match = fights_df[fights_df['localteam_id'].astype(str) == fid]
                if not local_match.empty:
                    name = local_match.iloc[0]['localteam_name']
                else:
                    away_match = fights_df[fights_df['awayteam_id'].astype(str) == fid]
                    if not away_match.empty:
                        name = away_match.iloc[0]['awayteam_name']
            
            # Calculate data quality score
            quality_score = self._calculate_quality_score(physical, fight_stats)
            
            # Build complete profile
            self.profiles[fid] = {
                "id": fid,
                "name": name,
                
                # Physical
                "height_inches": physical.get('height_inches', self.parser.DEFAULT_HEIGHT),
                "reach_inches": physical.get('reach_inches', self.parser.DEFAULT_REACH),
                "weight_lbs": physical.get('weight_lbs'),
                "age": physical.get('age', self.parser.DEFAULT_AGE),
                "stance": physical.get('stance', 'Orthodox'),
                "weight_class": physical.get('weight_class', ''),
                "team": physical.get('team', ''),
                "nickname": physical.get('nickname', ''),
                
                # Stats from historical fights
                **fight_stats,
                
                # Quality indicator
                "data_quality_score": quality_score,
            }
        
        # Save enhanced profiles
        with open(ENHANCED_PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, indent=2)
        
        print(f"\n{'=' * 60}")
        print(f"COMPLETE! Built {len(self.profiles)} enhanced profiles")
        print(f"Saved to: {ENHANCED_PROFILES_FILE}")
        print(f"{'=' * 60}")
        
        self._print_quality_report()
        
        return self.profiles
    
    def _calculate_quality_score(self, physical: Dict, stats: Dict) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Physical attributes (40% weight)
        if physical.get('height_inches'):
            score += 0.1
        max_score += 0.1
        
        if physical.get('reach_inches'):
            score += 0.1
        max_score += 0.1
        
        if physical.get('age'):
            score += 0.1
        max_score += 0.1
        
        if physical.get('stance'):
            score += 0.1
        max_score += 0.1
        
        # Fight stats (60% weight)
        if stats.get('total_fights', 0) > 0:
            score += 0.2
        max_score += 0.2
        
        if stats.get('total_fights', 0) >= 3:
            score += 0.2
        max_score += 0.2
        
        if stats.get('strikes', {}).get('total_landed_avg', 0) > 0:
            score += 0.2
        max_score += 0.2
        
        return round(score / max_score, 2) if max_score > 0 else 0
    
    def _print_quality_report(self):
        """Print quality report"""
        print("\n--- QUALITY REPORT ---")
        
        # Physical attributes coverage
        has_height = sum(1 for p in self.profiles.values() if p.get('height_inches'))
        has_reach = sum(1 for p in self.profiles.values() if p.get('reach_inches'))
        has_age = sum(1 for p in self.profiles.values() if p.get('age'))
        has_stance = sum(1 for p in self.profiles.values() if p.get('stance'))
        
        total = len(self.profiles)
        print(f"Height coverage:  {has_height}/{total} ({has_height/total*100:.1f}%)")
        print(f"Reach coverage:   {has_reach}/{total} ({has_reach/total*100:.1f}%)")
        print(f"Age coverage:     {has_age}/{total} ({has_age/total*100:.1f}%)")
        print(f"Stance coverage:  {has_stance}/{total} ({has_stance/total*100:.1f}%)")
        
        # Quality score distribution
        scores = [p.get('data_quality_score', 0) for p in self.profiles.values()]
        high_quality = sum(1 for s in scores if s >= 0.8)
        medium_quality = sum(1 for s in scores if 0.5 <= s < 0.8)
        low_quality = sum(1 for s in scores if s < 0.5)
        
        print(f"\nQuality distribution:")
        print(f"  High (>=0.8):   {high_quality}")
        print(f"  Medium (0.5-0.8): {medium_quality}")
        print(f"  Low (<0.5):     {low_quality}")
        
        # Fight stats coverage
        has_fights = sum(1 for p in self.profiles.values() if p.get('total_fights', 0) > 0)
        print(f"\nFighters with fight data: {has_fights}/{total} ({has_fights/total*100:.1f}%)")


def build_enhanced_profiles():
    """Main entry point"""
    builder = EnhancedProfileBuilder()
    return builder.build_profiles()


if __name__ == "__main__":
    build_enhanced_profiles()
