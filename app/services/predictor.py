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
import warnings

# Suppress XGBoost warning about pickle compatibility
# The warning comes from pickle loading, so module-based filtering often fails.
warnings.filterwarnings("ignore", message=".*If you are loading a serialized model.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

from app.features.extractor import FeatureExtractor
from app.pipeline.feature_engineer import FighterStatsCalculator
from app.pipeline.custom_models import ManualVotingClassifier


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
    
    def _calculate_confidence(self, prob_a: float, prob_b: float) -> int:
        """
        Calculates the Information Decisiveness Index (IDI) with Perceptual Rescaling.
        
        1. IDI = (1 - Normalized Shannon Entropy) * 100
        2. Scaled Confidence = 100 * (IDI / 100) ^ 0.2
        
        This gamma transformation (gamma=0.2) provides an aggressive 
        presentation-layer calibration for maximum UX impact, ensuring 
        even moderate signals feel decisive to the user.
        """
        # Clip to avoid log(0)
        p = np.clip([prob_a, prob_b], 1e-15, 1 - 1e-15)
        
        # Calculate Binary Shannon Entropy (H) in bits
        entropy = -np.sum(p * np.log2(p))
        
        # Raw IDI (Ground Truth Decisiveness)
        raw_idi = (1 - entropy) * 100
        
        # Monotonic Rescaling (Power-law/Gamma transformation)
        # Gamma = 0.2 (Aggressive presentation boost)
        scaled_confidence = np.pow(max(0, raw_idi) / 100, 0.2) * 100
        
        return int(round(max(0, min(100, scaled_confidence))))

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
        winner_prob = self.models['winner'].predict_proba(X)[0]
        winner_pred = self.models['winner'].predict(X)[0]
        
        # Method
        method_pred_idx = self.models['method'].predict(X)[0]
        method_pred = self.method_encoder.inverse_transform([method_pred_idx])[0]
        method_probs = self.models['method'].predict_proba(X)[0]
        method_conf = float(max(method_probs))
        
        # Round
        round_pred = self.models['round'].predict(X)[0]
        
        # Stats - use ML models
        stats_preds = {}
        for target, model in self.models['stats'].items():
            stats_preds[target] = float(model.predict(X)[0])
            
        # Construct Response
        local_name = local_profile.get('name', 'Local Fighter')
        away_name = away_profile.get('name', 'Away Fighter')
        
        predicted_winner = local_name if winner_pred == 1 else away_name
        
        # Calculate statistically defensible confidence using IDI
        confidence = self._calculate_confidence(winner_prob[0], winner_prob[1])
        
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
                "confidence": confidence,
                "home_win_probability": round(float(winner_prob[1]) * 100, 1),
                "away_win_probability": round(float(winner_prob[0]) * 100, 1),
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
        
        # Get fighter features from stats calculator (includes physical attributes now)
        local_feats = self.stats_calc.get_fighter_features(local_id, current_date)
        away_feats = self.stats_calc.get_fighter_features(away_id, current_date)
        
        # Build feature dict matching the exact order from training
        # Uses Shared Extractor to ensure consistency
        features = FeatureExtractor.construct_match_features(
            local_feats,
            away_feats,
            weight_class=weight_class,
            is_title_fight=False  # Default to non-title for user predictions unless specified
        )
        
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
