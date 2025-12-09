"""
Enhanced Feature Engineering Module v2
Creates comprehensive features from historical fight data for model training
Now includes physical attributes (height, reach, age, stance) from enhanced profiles

Features:
- ELO ratings and form indicators
- Physical differentials (height, reach, age)
- Stance matchup encoding
- Strike averages and ratios
- Grappling statistics
- Performance metrics
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import json


PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ENHANCED_PROFILES_FILE = PROCESSED_DIR / "enhanced_fighter_profiles.json"


class ELORatingSystem:
    """ELO rating system for fighters"""
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = defaultdict(lambda: initial_rating)
    
    def get_rating(self, fighter_id: str) -> float:
        """Get current ELO rating for a fighter"""
        return self.ratings[str(fighter_id)]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for fighter A against fighter B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner_id: str, loser_id: str, is_draw: bool = False):
        """Update ratings after a fight"""
        winner_id = str(winner_id)
        loser_id = str(loser_id)
        
        rating_w = self.ratings[winner_id]
        rating_l = self.ratings[loser_id]
        
        expected_w = self.expected_score(rating_w, rating_l)
        expected_l = 1 - expected_w
        
        if is_draw:
            actual_w = 0.5
            actual_l = 0.5
        else:
            actual_w = 1.0
            actual_l = 0.0
        
        self.ratings[winner_id] = rating_w + self.k_factor * (actual_w - expected_w)
        self.ratings[loser_id] = rating_l + self.k_factor * (actual_l - expected_l)


class EnhancedProfileLoader:
    """Loads enhanced fighter profiles with physical attributes"""
    
    # Stance encoding
    STANCE_MAP = {"Orthodox": 0, "Southpaw": 1, "Switch": 2}
    
    # Default values for imputation
    DEFAULT_HEIGHT = 70  # inches
    DEFAULT_REACH = 72   # inches
    DEFAULT_AGE = 32
    
    def __init__(self):
        self.profiles: Dict[str, Dict] = {}
        self._load_profiles()
    
    def _load_profiles(self):
        """Load enhanced profiles from file"""
        if ENHANCED_PROFILES_FILE.exists():
            with open(ENHANCED_PROFILES_FILE, 'r', encoding='utf-8') as f:
                self.profiles = json.load(f)
            print(f"Loaded {len(self.profiles)} enhanced fighter profiles")
        else:
            print("Warning: Enhanced profiles not found. Physical features will be imputed.")
    
    def get_physical_features(self, fighter_id: str) -> Dict[str, float]:
        """Get physical attributes for a fighter"""
        fid = str(fighter_id)
        profile = self.profiles.get(fid, {})
        
        height = profile.get('height_inches') or self.DEFAULT_HEIGHT
        reach = profile.get('reach_inches') or self.DEFAULT_REACH
        age = profile.get('age') or self.DEFAULT_AGE
        stance_str = profile.get('stance', 'Orthodox')
        stance = self.STANCE_MAP.get(stance_str, 0)
        
        return {
            'height_inches': float(height),
            'reach_inches': float(reach),
            'age': float(age),
            'stance': stance,
            'is_southpaw': 1 if stance_str == 'Southpaw' else 0,
        }
    
    def get_strike_averages(self, fighter_id: str) -> Dict[str, float]:
        """Get average strike statistics from profile"""
        fid = str(fighter_id)
        profile = self.profiles.get(fid, {})
        strikes = profile.get('strikes', {})
        
        return {
            'avg_total_strikes': strikes.get('total_landed_avg', 0),
            'avg_head_strikes': strikes.get('head_landed_avg', 0),
            'avg_body_strikes': strikes.get('body_landed_avg', 0),
            'avg_leg_strikes': strikes.get('legs_landed_avg', 0),
            'avg_power_head': strikes.get('power_head_landed_avg', 0),
            'avg_power_body': strikes.get('power_body_landed_avg', 0),
            'avg_power_legs': strikes.get('power_legs_landed_avg', 0),
            'slpm': strikes.get('slpm', 0),
            'sapm': strikes.get('sapm', 0),
        }
    
    def get_grappling_averages(self, fighter_id: str) -> Dict[str, float]:
        """Get average grappling statistics from profile"""
        fid = str(fighter_id)
        profile = self.profiles.get(fid, {})
        grappling = profile.get('grappling', {})
        
        return {
            'avg_td_landed': grappling.get('td_landed_avg', 0),
            'td_acc': grappling.get('td_acc', 0),
            'td_def': grappling.get('td_def', 0.5),
            'avg_submissions': grappling.get('submissions_avg', 0),
            'avg_ctrl_time': grappling.get('ctrl_time_avg', 0),
        }
    
    def get_knockdown_averages(self, fighter_id: str) -> Dict[str, float]:
        """Get average knockdown statistics"""
        fid = str(fighter_id)
        profile = self.profiles.get(fid, {})
        knockdowns = profile.get('knockdowns', {})
        
        return {
            'avg_kd_landed': knockdowns.get('kd_landed_avg', 0),
            'avg_kd_absorbed': knockdowns.get('kd_absorbed_avg', 0),
        }


class FighterStatsCalculator:
    """Calculates historical statistics for fighters with enhanced metrics"""
    
    def __init__(self):
        self.fighter_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "fights": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "current_streak": 0,
            "last_fight_date": None,
            "first_fight_date": None,
            "ko_wins": 0,
            "sub_wins": 0,
            "dec_wins": 0,
            "ko_losses": 0,
            "sub_losses": 0,
            "sub_losses": 0,
            "history": [],
            "strikes_history": [],  # For consistency
            "last_ko_loss_date": None,
        })
        self.elo_system = ELORatingSystem()
        self.profile_loader = EnhancedProfileLoader()
    
    def update_stats(self, fight: pd.Series, is_local: bool):
        """Update fighter stats after a fight"""
        if is_local:
            fighter_id = str(fight['localteam_id'])
            opp_id = str(fight['awayteam_id'])
            prefix = "local_"
            opp_prefix = "away_"
            won = fight['winner'] == 1
            draw = fight['winner'] == -1
        else:
            fighter_id = str(fight['awayteam_id'])
            opp_id = str(fight['localteam_id'])
            prefix = "away_"
            opp_prefix = "local_"
            won = fight['winner'] == 0
            draw = fight['winner'] == -1
        
        if not fighter_id or fighter_id == 'nan':
            return
        
        stats = self.fighter_stats[fighter_id]
        method = fight.get('win_method', 'OTHER')
        
        # Update basic record
        stats['fights'] += 1
        if draw:
            stats['draws'] += 1
            stats['current_streak'] = 0
        elif won:
            stats['wins'] += 1
            if stats['current_streak'] > 0:
                stats['current_streak'] += 1
            else:
                stats['current_streak'] = 1
            if method == 'KO':
                stats['ko_wins'] += 1
            elif method == 'SUB':
                stats['sub_wins'] += 1
            else:
                stats['dec_wins'] += 1
        else:
            stats['losses'] += 1
            if stats['current_streak'] < 0:
                stats['current_streak'] -= 1
            else:
                stats['current_streak'] = -1
            if method == 'KO':
                stats['ko_losses'] += 1
            elif method == 'SUB':
                stats['sub_losses'] += 1
        
        # Track chin health
        if not won and method == 'KO':
            stats['last_ko_loss_date'] = fight['fight_date']
        
        # Store fight stats for rolling averages
        fight_stats = {
            'date': fight['fight_date'],
            'won': 1 if won else 0,
            'draw': 1 if draw else 0,
            'finished': 1 if method in ['KO', 'SUB'] else 0,
            'method': method,
            'strikes_landed': fight.get(f'{prefix}total_strikes', 0),
            'strikes_absorbed': fight.get(f'{opp_prefix}total_strikes', 0),
            'strikes_head': fight.get(f'{prefix}strikes_head', 0),
            'strikes_body': fight.get(f'{prefix}strikes_body', 0),
            'strikes_legs': fight.get(f'{prefix}strikes_legs', 0),
            'takedowns_landed': fight.get(f'{prefix}takedowns_landed', 0),
            'takedowns_absorbed': fight.get(f'{opp_prefix}takedowns_landed', 0),
            'knockdowns': fight.get(f'{prefix}knockdowns', 0),
            'control_time': fight.get(f'{prefix}control_time', 0),
            'fight_time': fight.get('fight_round', 3) * 300,
        }
        stats['history'].append(fight_stats)
        stats['strikes_history'].append(fight.get(f'{prefix}total_strikes', 0))
        
        # Update dates
        if stats['first_fight_date'] is None:
            stats['first_fight_date'] = fight.get('fight_date')
        stats['last_fight_date'] = fight.get('fight_date')
        
        # Update ELO ratings
        if not draw:
            if won:
                self.elo_system.update_ratings(fighter_id, opp_id)
            else:
                self.elo_system.update_ratings(opp_id, fighter_id)
        else:
            self.elo_system.update_ratings(fighter_id, opp_id, is_draw=True)
    
    def get_fighter_features(self, fighter_id: str, current_date: pd.Timestamp) -> Dict[str, float]:
        """Get comprehensive features for a fighter at a specific date"""
        stats = self.fighter_stats.get(str(fighter_id), {})
        
        # Get physical features from enhanced profiles
        physical = self.profile_loader.get_physical_features(fighter_id)
        
        if not stats:
            return self._get_empty_features(physical)
        
        # Basic stats
        fights = max(stats.get('fights', 0), 1)
        wins = stats.get('wins', 0)
        
        features = {
            # Physical attributes from enhanced profiles
            'height_inches': physical['height_inches'],
            'reach_inches': physical['reach_inches'],
            'age': physical['age'],
            'stance': physical['stance'],
            'is_southpaw': physical['is_southpaw'],
            
            # Performance stats
            'win_rate': wins / fights,
            'experience': fights,
            'current_streak': stats.get('current_streak', 0),
            'current_streak': stats.get('current_streak', 0),
            'elo_rating': self.elo_system.get_rating(str(fighter_id)),
        }
        
        # Consistency (Standard Deviation of strikes)
        strikes_hist = stats.get('strikes_history', [])
        if len(strikes_hist) >= 3:
            features['consistency'] = np.std(strikes_hist)
        else:
            features['consistency'] = 20.0  # Default assumption
            
        # Chin Health (Days since last KO loss)
        last_ko_date = stats.get('last_ko_loss_date')
        if last_ko_date is not None and not pd.isna(last_ko_date):
            days_since_ko = (current_date - last_ko_date).days
            features['days_since_ko'] = max(0, days_since_ko)
        else:
            features['days_since_ko'] = 365 * 5  # 5 years (assumed healthy)
        
        # Finish rates
        if wins > 0:
            features['ko_rate'] = stats.get('ko_wins', 0) / wins
            features['sub_rate'] = stats.get('sub_wins', 0) / wins
            features['finish_rate'] = (stats.get('ko_wins', 0) + stats.get('sub_wins', 0)) / wins
        else:
            features['ko_rate'] = 0.0
            features['sub_rate'] = 0.0
            features['finish_rate'] = 0.0
        
        # Loss vulnerability
        losses = max(stats.get('losses', 0), 1)
        features['ko_loss_rate'] = stats.get('ko_losses', 0) / losses
        features['sub_loss_rate'] = stats.get('sub_losses', 0) / losses
        
        # Days since last fight
        last_date = stats.get('last_fight_date')
        if last_date is not None and not pd.isna(last_date):
            days_since = (current_date - last_date).days
            features['days_since_last_fight'] = max(0, days_since)
        else:
            features['days_since_last_fight'] = 365
        
        # Activity
        first_date = stats.get('first_fight_date')
        if first_date is not None and last_date is not None:
            career_days = max((last_date - first_date).days, 365)
            features['activity'] = fights / (career_days / 365.25)
        else:
            features['activity'] = 1.0
        
        # Rolling averages (last 3 fights)
        history = stats.get('history', [])
        last_3 = history[-3:] if history else []
        
        if last_3:
            n = len(last_3)
            features['l3_win_rate'] = sum(x['won'] for x in last_3) / n
            features['l3_finish_rate'] = sum(x['finished'] for x in last_3) / n
            features['l3_strikes_landed'] = sum(x['strikes_landed'] for x in last_3) / n
            features['l3_strikes_absorbed'] = sum(x['strikes_absorbed'] for x in last_3) / n
            features['l3_takedowns_landed'] = sum(x['takedowns_landed'] for x in last_3) / n
            features['l3_takedowns_absorbed'] = sum(x['takedowns_absorbed'] for x in last_3) / n
            
            # Strike distribution
            total_head = sum(x.get('strikes_head', 0) for x in last_3)
            total_body = sum(x.get('strikes_body', 0) for x in last_3)
            total_legs = sum(x.get('strikes_legs', 0) for x in last_3)
            total_strikes = total_head + total_body + total_legs
            
            if total_strikes > 0:
                features['head_ratio'] = total_head / total_strikes
                features['body_ratio'] = total_body / total_strikes
                features['legs_ratio'] = total_legs / total_strikes
            else:
                features['head_ratio'] = 0.5
                features['body_ratio'] = 0.25
                features['legs_ratio'] = 0.25
            
            # Form indicator
            weights = [0.5, 0.3, 0.2][:n]
            weights = [w / sum(weights) for w in weights]
            features['form'] = sum(last_3[-(i+1)]['won'] * weights[i] for i in range(n))
        else:
            features['l3_win_rate'] = 0.0
            features['l3_finish_rate'] = 0.0
            features['l3_strikes_landed'] = 0.0
            features['l3_strikes_absorbed'] = 0.0
            features['l3_takedowns_landed'] = 0.0
            features['l3_takedowns_absorbed'] = 0.0
            features['head_ratio'] = 0.5
            features['body_ratio'] = 0.25
            features['legs_ratio'] = 0.25
            features['form'] = 0.0
        
        return features

    
    def _get_empty_features(self, physical: Dict) -> Dict[str, float]:
        """Return zeroed features for unknown fighters with physical data"""
        return {
            'height_inches': physical['height_inches'],
            'reach_inches': physical['reach_inches'],
            'age': physical['age'],
            'stance': physical['stance'],
            'is_southpaw': physical['is_southpaw'],
            'win_rate': 0.0,
            'experience': 0,
            'current_streak': 0,
            'elo_rating': 1500.0,
            'consistency': 20.0,
            'days_since_ko': 365 * 5,
            'ko_rate': 0.0,
            'sub_rate': 0.0,
            'finish_rate': 0.0,
            'ko_loss_rate': 0.0,
            'sub_loss_rate': 0.0,
            'days_since_last_fight': 365,
            'activity': 0.0,
            'l3_win_rate': 0.0,
            'l3_finish_rate': 0.0,
            'l3_strikes_landed': 0.0,
            'l3_strikes_absorbed': 0.0,
            'l3_takedowns_landed': 0.0,
            'l3_takedowns_absorbed': 0.0,
            'head_ratio': 0.5,
            'body_ratio': 0.25,
            'legs_ratio': 0.25,
            'form': 0.0,
        }
    
    def save_state(self, filepath: Path = None):
        """Save calculator state for inference"""
        if filepath is None:
            filepath = PROCESSED_DIR / "fighter_stats_state.json"
        
        state = {
            'fighter_stats': dict(self.fighter_stats),
            'elo_ratings': dict(self.elo_system.ratings),
        }
        
        # Convert dates to strings
        for fid, stats in state['fighter_stats'].items():
            if stats.get('last_fight_date') is not None:
                stats['last_fight_date'] = str(stats['last_fight_date'])
            if stats.get('first_fight_date') is not None:
                stats['first_fight_date'] = str(stats['first_fight_date'])
            if stats.get('last_ko_loss_date') is not None:
                stats['last_ko_loss_date'] = str(stats['last_ko_loss_date'])
            for h in stats.get('history', []):
                if 'date' in h and h['date'] is not None:
                    h['date'] = str(h['date'])
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"Saved fighter stats state to {filepath}")
    
    def load_state(self, filepath: Path = None):
        """Load calculator state for inference"""
        if filepath is None:
            filepath = PROCESSED_DIR / "fighter_stats_state.json"
        
        if not filepath.exists():
            print(f"State file not found: {filepath}")
            return False
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        for fid, stats in state.get('fighter_stats', {}).items():
            self.fighter_stats[fid] = stats
            if stats.get('last_fight_date'):
                stats['last_fight_date'] = pd.to_datetime(stats['last_fight_date'])
            if stats.get('first_fight_date'):
                stats['first_fight_date'] = pd.to_datetime(stats['first_fight_date'])
            if stats.get('last_ko_loss_date'):
                stats['last_ko_loss_date'] = pd.to_datetime(stats['last_ko_loss_date'])
        
        for fid, rating in state.get('elo_ratings', {}).items():
            self.elo_system.ratings[fid] = rating
        
        print(f"Loaded state for {len(self.fighter_stats)} fighters")
        return True


class EnhancedFeatureEngineer:
    """Creates enhanced features including physical attributes"""
    
    def __init__(self):
        self.stats_calc = FighterStatsCalculator()
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create comprehensive features from historical fight data"""
        # Sort by date
        df = df.sort_values('fight_date').reset_index(drop=True)
        
        features_list = []
        
        for idx, row in df.iterrows():
            current_date = row['fight_date']
            
            # Get fighter features BEFORE this fight
            local_feats = self.stats_calc.get_fighter_features(row['localteam_id'], current_date)
            away_feats = self.stats_calc.get_fighter_features(row['awayteam_id'], current_date)
            
            # Create feature row
            features = {
                'match_id': row['match_id'],
                'fight_date': row['fight_date'],
                'weight_class': row.get('weight_class_encoded', 5),
                'is_title_fight': 1 if row.get('fight_round') == 5 else 0,
                
                # ===== PHYSICAL FEATURES (NEW) =====
                'local_height': local_feats['height_inches'],
                'local_reach': local_feats['reach_inches'],
                'local_age': local_feats['age'],
                'local_is_southpaw': local_feats['is_southpaw'],
                
                'away_height': away_feats['height_inches'],
                'away_reach': away_feats['reach_inches'],
                'away_age': away_feats['age'],
                'away_is_southpaw': away_feats['is_southpaw'],
                
                # Physical differentials
                'diff_height': local_feats['height_inches'] - away_feats['height_inches'],
                'diff_reach': local_feats['reach_inches'] - away_feats['reach_inches'],
                'diff_age': local_feats['age'] - away_feats['age'],
                'stance_matchup': 1 if local_feats['is_southpaw'] != away_feats['is_southpaw'] else 0,
                
                # ===== LOCAL FIGHTER FEATURES =====
                'local_win_rate': local_feats['win_rate'],
                'local_experience': local_feats['experience'],
                'local_streak': local_feats['current_streak'],
                'local_elo': local_feats['elo_rating'],
                'local_ko_rate': local_feats['ko_rate'],
                'local_sub_rate': local_feats['sub_rate'],
                'local_finish_rate': local_feats['finish_rate'],
                'local_ko_loss_rate': local_feats['ko_loss_rate'],
                'local_days_since': local_feats['days_since_last_fight'],
                'local_activity': local_feats['activity'],
                'local_l3_strikes': local_feats['l3_strikes_landed'],
                'local_l3_absorbed': local_feats['l3_strikes_absorbed'],
                'local_l3_td': local_feats['l3_takedowns_landed'],
                'local_form': local_feats['form'],
                'local_form': local_feats['form'],
                'local_head_ratio': local_feats['head_ratio'],
                'local_body_ratio': local_feats['body_ratio'],
                'local_consistency': local_feats['consistency'],
                'local_chin': local_feats['days_since_ko'],
                
                # ===== AWAY FIGHTER FEATURES =====
                'away_win_rate': away_feats['win_rate'],
                'away_experience': away_feats['experience'],
                'away_streak': away_feats['current_streak'],
                'away_elo': away_feats['elo_rating'],
                'away_ko_rate': away_feats['ko_rate'],
                'away_sub_rate': away_feats['sub_rate'],
                'away_finish_rate': away_feats['finish_rate'],
                'away_ko_loss_rate': away_feats['ko_loss_rate'],
                'away_days_since': away_feats['days_since_last_fight'],
                'away_activity': away_feats['activity'],
                'away_l3_strikes': away_feats['l3_strikes_landed'],
                'away_l3_absorbed': away_feats['l3_strikes_absorbed'],
                'away_l3_td': away_feats['l3_takedowns_landed'],
                'away_form': away_feats['form'],
                'away_head_ratio': away_feats['head_ratio'],
                'away_body_ratio': away_feats['body_ratio'],
                'away_consistency': away_feats['consistency'],
                'away_chin': away_feats['days_since_ko'],
                
                # ===== DIFFERENTIALS (Local - Away) =====
                'diff_win_rate': local_feats['win_rate'] - away_feats['win_rate'],
                'diff_experience': local_feats['experience'] - away_feats['experience'],
                'diff_streak': local_feats['current_streak'] - away_feats['current_streak'],
                'diff_elo': local_feats['elo_rating'] - away_feats['elo_rating'],
                'diff_ko_rate': local_feats['ko_rate'] - away_feats['ko_rate'],
                'diff_sub_rate': local_feats['sub_rate'] - away_feats['sub_rate'],
                'diff_finish_rate': local_feats['finish_rate'] - away_feats['finish_rate'],
                'diff_strikes': local_feats['l3_strikes_landed'] - away_feats['l3_strikes_landed'],
                'diff_absorbed': local_feats['l3_strikes_absorbed'] - away_feats['l3_strikes_absorbed'],
                'diff_td': local_feats['l3_takedowns_landed'] - away_feats['l3_takedowns_landed'],
                'diff_form': local_feats['form'] - away_feats['form'],
                'diff_activity': local_feats['activity'] - away_feats['activity'],
                'diff_consistency': local_feats['consistency'] - away_feats['consistency'],
                'diff_chin': local_feats['days_since_ko'] - away_feats['days_since_ko'],
                
                # ===== MATCHUP FEATURES =====
                'local_striker_score': local_feats['l3_strikes_landed'] - local_feats['l3_takedowns_landed'],
                'away_striker_score': away_feats['l3_strikes_landed'] - away_feats['l3_takedowns_landed'],
                
                # ===== TARGETS =====
                'winner': row['winner'],
                'win_method': row['win_method'],
                'fight_round': row['fight_round'],
                
                # Stats targets (for regression)
                'local_total_strikes': row.get('local_total_strikes', 0),
                'away_total_strikes': row.get('away_total_strikes', 0),
                'local_strikes_head': row.get('local_strikes_head', 0),
                'local_strikes_body': row.get('local_strikes_body', 0),
                'local_strikes_legs': row.get('local_strikes_legs', 0),
                'away_strikes_head': row.get('away_strikes_head', 0),
                'away_strikes_body': row.get('away_strikes_body', 0),
                'away_strikes_legs': row.get('away_strikes_legs', 0),
                'local_takedowns_landed': row.get('local_takedowns_landed', 0),
                'away_takedowns_landed': row.get('away_takedowns_landed', 0),
            }
            
            features_list.append(features)
            
            # Update stats with this fight result
            self.stats_calc.update_stats(row, is_local=True)
            self.stats_calc.update_stats(row, is_local=False)
        
        feature_df = pd.DataFrame(features_list)
        
        # Define feature columns (order matters for model input)
        feature_cols = [
            # Context
            'weight_class', 'is_title_fight',
            
            # Physical features (NEW)
            'local_height', 'local_reach', 'local_age', 'local_is_southpaw',
            'away_height', 'away_reach', 'away_age', 'away_is_southpaw',
            'diff_height', 'diff_reach', 'diff_age', 'stance_matchup',
            
            # Local fighter
            'local_win_rate', 'local_experience', 'local_streak', 'local_elo',
            'local_ko_rate', 'local_sub_rate', 'local_finish_rate', 'local_ko_loss_rate',
            'local_days_since', 'local_activity',
            'local_l3_strikes', 'local_l3_absorbed', 'local_l3_td',
            'local_form', 'local_head_ratio', 'local_body_ratio',
            'local_consistency', 'local_chin',
            
            # Away fighter
            'away_win_rate', 'away_experience', 'away_streak', 'away_elo',
            'away_ko_rate', 'away_sub_rate', 'away_finish_rate', 'away_ko_loss_rate',
            'away_days_since', 'away_activity',
            'away_l3_strikes', 'away_l3_absorbed', 'away_l3_td',
            'away_form', 'away_head_ratio', 'away_body_ratio',
            'away_consistency', 'away_chin',
            
            # Differentials
            'diff_win_rate', 'diff_experience', 'diff_streak', 'diff_elo',
            'diff_ko_rate', 'diff_sub_rate', 'diff_finish_rate',
            'diff_strikes', 'diff_absorbed', 'diff_td', 'diff_form', 'diff_activity',
            'diff_consistency', 'diff_chin',
            
            # Matchup
            'local_striker_score', 'away_striker_score',
        ]
        
        # Save stats calculator state for inference
        self.stats_calc.save_state()
        
        return feature_df, feature_cols
    
    def save_features(self, df: pd.DataFrame, filename: str = "features.csv"):
        """Save feature DataFrame"""
        output_path = PROCESSED_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"Saved features to {output_path}")
        return output_path


def engineer_features(processed_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create features from processed data"""
    engineer = EnhancedFeatureEngineer()
    feature_df, feature_cols = engineer.create_features(processed_df)
    engineer.save_features(feature_df)
    return feature_df, feature_cols


# Keep old class for backward compatibility
FeatureEngineer = EnhancedFeatureEngineer


if __name__ == "__main__":
    from app.pipeline.data_processor import process_data
    
    # Load and process raw data
    processed_df = process_data()
    
    # Engineer features
    feature_df, feature_cols = engineer_features(processed_df)
    print(f"Created {len(feature_df)} rows with {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
