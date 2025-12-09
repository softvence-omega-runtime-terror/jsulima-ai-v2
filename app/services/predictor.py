"""
Prediction Service v2
Handles loading models and generating predictions using enhanced features
Now includes physical attributes (height, reach, age, stance)
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.pipeline.feature_engineer import FighterStatsCalculator


MODELS_DIR = Path(__file__).parent.parent / "models" / "saved"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ENHANCED_PROFILES_FILE = PROCESSED_DIR / "enhanced_fighter_profiles.json"


class Predictor:
    """UFC Match Predictor using trained ML models with enhanced features"""
    
    def __init__(self):
        self.models = {}
        self.feature_cols = []
        self.method_encoder = None
        self.stats_calc = FighterStatsCalculator()
        self.fighter_profiles = {}
        self.enhanced_profiles = {}
        self._loaded = False
        self.load_resources()
    
    def load_resources(self):
        """Load trained models and resources"""
        try:
            print("Loading models...")
            self.models['winner'] = joblib.load(MODELS_DIR / "winner_model.pkl")
            self.models['method'] = joblib.load(MODELS_DIR / "method_model.pkl")
            self.models['round'] = joblib.load(MODELS_DIR / "round_model.pkl")
            self.models['stats'] = joblib.load(MODELS_DIR / "stats_models.pkl")
            
            self.method_encoder = joblib.load(MODELS_DIR / "method_encoder.pkl")
            self.feature_cols = joblib.load(MODELS_DIR / "feature_columns.pkl")
            
            # Load fighter profiles for display
            profiles_path = PROCESSED_DIR / "fighter_profiles.json"
            if profiles_path.exists():
                with open(profiles_path, 'r', encoding='utf-8') as f:
                    self.fighter_profiles = json.load(f)
                print(f"Loaded {len(self.fighter_profiles)} fighter profiles")
            
            # Load enhanced profiles for physical data
            if ENHANCED_PROFILES_FILE.exists():
                with open(ENHANCED_PROFILES_FILE, 'r', encoding='utf-8') as f:
                    self.enhanced_profiles = json.load(f)
                print(f"Loaded {len(self.enhanced_profiles)} enhanced profiles")
            
            # Load fighter stats state
            state_loaded = self.stats_calc.load_state()
            if state_loaded:
                print("Fighter stats state loaded for accurate predictions")
            else:
                print("Warning: Fighter stats state not found - using profiles for approximation")
            
            print("All resources loaded successfully")
            self._loaded = True
        except Exception as e:
            print(f"Error loading resources: {e}")
            self._loaded = False
    
    def load_models(self):
        """Alias for load_resources (for compatibility)"""
        self.load_resources()
    
    def predict_match(self, local_id: str, away_id: str, date: str = None, weight_class: str = "Lightweight") -> Dict[str, Any]:
        """
        Predict outcome of a match using trained ML models
        
        Args:
            local_id: ID of local fighter
            away_id: ID of away fighter
            date: Match date (optional, defaults to today)
            weight_class: Weight class of the fight
            
        Returns:
            Prediction dictionary with winner, method, round, and stats
        """
        if not self.models:
            return {"error": "Models not loaded"}
            
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Parse date
        try:
            if '.' in date:
                current_date = pd.to_datetime(date, format='%d.%m.%Y')
            else:
                current_date = pd.to_datetime(date)
        except:
            current_date = pd.Timestamp.now()
            
        # Get Fighter Profiles for display
        local_profile = self.fighter_profiles.get(str(local_id), {})
        away_profile = self.fighter_profiles.get(str(away_id), {})
        
        # Get Enhanced Profiles for physical data
        local_enhanced = self.enhanced_profiles.get(str(local_id), {})
        away_enhanced = self.enhanced_profiles.get(str(away_id), {})
        
        # Merge enhanced data into profiles for display
        if local_enhanced:
            local_profile = {**local_profile, **local_enhanced}
        if away_enhanced:
            away_profile = {**away_profile, **away_enhanced}
        
        # Construct features using the same logic as training
        features = self._construct_features(str(local_id), str(away_id), current_date, weight_class)
        
        # Create DataFrame with correct column order
        X = pd.DataFrame([features], columns=self.feature_cols).fillna(0)
        
        # Make Predictions
        # Winner
        winner_prob = self.models['winner'].predict_proba(X.values)[0]
        winner_pred = self.models['winner'].predict(X.values)[0]
        
        # Method
        method_pred_idx = self.models['method'].predict(X.values)[0]
        method_pred = self.method_encoder.inverse_transform([method_pred_idx])[0]
        method_probs = self.models['method'].predict_proba(X.values)[0]
        method_conf = float(max(method_probs))
        
        # Round
        round_pred = self.models['round'].predict(X.values)[0]
        
        # Stats - use ML models
        stats_preds = {}
        for target, model in self.models['stats'].items():
            stats_preds[target] = float(model.predict(X.values)[0])
            
        # Construct Response
        local_name = local_profile.get('name', 'Local Fighter')
        away_name = away_profile.get('name', 'Away Fighter')
        
        predicted_winner = local_name if winner_pred == 1 else away_name
        confidence = float(winner_prob[1] if winner_pred == 1 else winner_prob[0])
        
        # PvP History
        pvp_history = self._get_pvp_history(local_profile, str(away_id))
        
        # Comparison Stats
        comparison = self._compare_fighters(local_profile, away_profile)
        
        return {
            "match_info": {
                "date": date,
                "local_fighter": local_name,
                "away_fighter": away_name,
                "weight_class": weight_class,
            },
            "prediction": {
                "winner": predicted_winner,
                "winner_is_local": bool(winner_pred == 1),
                "confidence": round(confidence * 100, 1),
                "method": method_pred,
                "method_confidence": round(method_conf * 100, 1),
                "round": int(round_pred),
                "predicted_stats": {
                    # Total strikes
                    "local_total_strikes": int(round(stats_preds.get('local_total_strikes', 0))),
                    "away_total_strikes": int(round(stats_preds.get('away_total_strikes', 0))),
                    # Head strikes
                    "local_strikes_head": int(round(stats_preds.get('local_strikes_head', 0))),
                    "away_strikes_head": int(round(stats_preds.get('away_strikes_head', 0))),
                    # Body strikes
                    "local_strikes_body": int(round(stats_preds.get('local_strikes_body', 0))),
                    "away_strikes_body": int(round(stats_preds.get('away_strikes_body', 0))),
                    # Leg strikes
                    "local_strikes_legs": int(round(stats_preds.get('local_strikes_legs', 0))),
                    "away_strikes_legs": int(round(stats_preds.get('away_strikes_legs', 0))),
                    # Takedowns
                    "local_takedowns": float(round(stats_preds.get('local_takedowns_landed', 0), 1)),
                    "away_takedowns": float(round(stats_preds.get('away_takedowns_landed', 0), 1)),
                }
            },
            "comparison": comparison,
            "pvp_history": pvp_history,
            "profiles": {
                "local": local_profile,
                "away": away_profile
            }
        }
    
    def _construct_features(self, local_id: str, away_id: str, current_date: pd.Timestamp, weight_class: str) -> Dict[str, float]:
        """
        Construct feature vector using the exact same logic as EnhancedFeatureEngineer
        Uses the loaded FighterStatsCalculator state
        """
        # Weight class encoding
        WEIGHT_CLASSES = {
            "Strawweight": 1, "Women's Strawweight": 1,
            "Flyweight": 2, "Women's Flyweight": 2,
            "Bantamweight": 3, "Women's Bantamweight": 3,
            "Featherweight": 4, "Women's Featherweight": 4,
            "Lightweight": 5,
            "Welterweight": 6,
            "Middleweight": 7,
            "Light Heavyweight": 8,
            "Heavyweight": 9,
            "Catch Weight": 5,
        }
        
        weight_class_encoded = WEIGHT_CLASSES.get(weight_class, 5)
        
        # Get fighter features from stats calculator (includes physical attributes now)
        local_feats = self.stats_calc.get_fighter_features(local_id, current_date)
        away_feats = self.stats_calc.get_fighter_features(away_id, current_date)
        
        # Build feature dict matching the exact order from training
        features = {
            # Context
            'weight_class': weight_class_encoded,
            'is_title_fight': 0,
            
            # Physical features (NEW)
            'local_height': local_feats.get('height_inches', 70),
            'local_reach': local_feats.get('reach_inches', 72),
            'local_age': local_feats.get('age', 32),
            'local_is_southpaw': local_feats.get('is_southpaw', 0),
            
            'away_height': away_feats.get('height_inches', 70),
            'away_reach': away_feats.get('reach_inches', 72),
            'away_age': away_feats.get('age', 32),
            'away_is_southpaw': away_feats.get('is_southpaw', 0),
            
            # Physical differentials
            'diff_height': local_feats.get('height_inches', 70) - away_feats.get('height_inches', 70),
            'diff_reach': local_feats.get('reach_inches', 72) - away_feats.get('reach_inches', 72),
            'diff_age': local_feats.get('age', 32) - away_feats.get('age', 32),
            'stance_matchup': 1 if local_feats.get('is_southpaw', 0) != away_feats.get('is_southpaw', 0) else 0,
            
            # Local fighter
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
            'local_head_ratio': local_feats['head_ratio'],
            'local_body_ratio': local_feats['body_ratio'],
            
            # Away fighter
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
            
            # Differentials
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
            
            # Matchup
            'local_striker_score': local_feats['l3_strikes_landed'] - local_feats['l3_takedowns_landed'],
            'away_striker_score': away_feats['l3_strikes_landed'] - away_feats['l3_takedowns_landed'],
            
            # New Features
            'local_consistency': local_feats.get('consistency', 20.0),
            'local_chin': local_feats.get('days_since_ko', 365*5),
            'away_consistency': away_feats.get('consistency', 20.0),
            'away_chin': away_feats.get('days_since_ko', 365*5),
            'diff_consistency': local_feats.get('consistency', 20.0) - away_feats.get('consistency', 20.0),
            'diff_chin': local_feats.get('days_since_ko', 365*5) - away_feats.get('days_since_ko', 365*5),
        }
        
        return features

    def _get_pvp_history(self, profile: Dict, opponent_id: str) -> List[Dict]:
        """Get past matches between these two fighters"""
        history = profile.get('pvp_history', {})
        if opponent_id in history:
            return [history[opponent_id]]
        return []

    def _compare_fighters(self, local: Dict, away: Dict) -> Dict[str, Any]:
        """Generate comparison stats for display"""
        # Use enhanced profile structure
        local_record = local.get('record', {})
        away_record = away.get('record', {})
        local_strikes = local.get('strikes', {})
        away_strikes = away.get('strikes', {})
        local_grappling = local.get('grappling', {})
        away_grappling = away.get('grappling', {})
        
        return {
            "record": {
                "local": f"{local_record.get('wins', 0)}-{local_record.get('losses', 0)}",
                "away": f"{away_record.get('wins', 0)}-{away_record.get('losses', 0)}"
            },
            "physical": {
                "height": {"local": local.get('height_inches', 0), "away": away.get('height_inches', 0)},
                "reach": {"local": local.get('reach_inches', 0), "away": away.get('reach_inches', 0)},
                "age": {"local": local.get('age', 0), "away": away.get('age', 0)},
            },
            "striking": {
                "slpm": {"local": local_strikes.get('slpm', 0), "away": away_strikes.get('slpm', 0)},
                "sapm": {"local": local_strikes.get('sapm', 0), "away": away_strikes.get('sapm', 0)},
                "total_avg": {"local": local_strikes.get('total_landed_avg', 0), "away": away_strikes.get('total_landed_avg', 0)},
            },
            "grappling": {
                "td_avg": {"local": local_grappling.get('td_landed_avg', 0), "away": away_grappling.get('td_landed_avg', 0)},
                "td_acc": {"local": local_grappling.get('td_acc', 0), "away": away_grappling.get('td_acc', 0)},
                "sub_avg": {"local": local_grappling.get('submissions_avg', 0), "away": away_grappling.get('submissions_avg', 0)},
            }
        }


# Global predictor instance
predictor = Predictor()


def get_predictor() -> Predictor:
    """Get the global predictor instance"""
    return predictor
