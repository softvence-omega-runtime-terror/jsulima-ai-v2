"""
Basketball Game Predictor - Single Bundle Version
Uses nba_prediction_full_bundle.pkl for all predictions
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "saved"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
TEAM_PROFILES_FILE = PROCESSED_DIR / "nba_team_profiles.json"


class BasketballPredictor:
    """NBA Game Predictor using single bundle pickle file"""
    
    def __init__(self):
        self.bundle = None
        self.models = {}
        self.feature_cols = []
        self.team_profiles = {}
        self._loaded = False
        self.load_resources()
    
    def load_resources(self):
        """Load the single nba prediction bundle"""
        try:
            print("Loading basketball prediction bundle...")
            
            # Load the single bundle file
            bundle_path = MODELS_DIR / "nba_prediction_full_bundle.pkl"
            self.bundle = joblib.load(bundle_path)
            print(f"Bundle loaded with keys: {list(self.bundle.keys())}")
            
            # Extract components from bundle
            # Adjust these keys based on what's actually in your bundle
            if isinstance(self.bundle, dict):
                self.models['winner'] = self.bundle.get('winner_model')
                self.models['stats'] = self.bundle.get('stats_models', {})
                self.models['scoring'] = self.bundle.get('scoring_model')
                
                # Feature columns might be stored differently
                if 'feature_columns' in self.bundle:
                    self.feature_cols = self.bundle['feature_columns']
                elif 'feature_names' in self.bundle:
                    self.feature_cols = self.bundle['feature_names']
            else:
                # Bundle might be a single model
                self.models['winner'] = self.bundle
            
            # Try to get feature columns from model
            if not self.feature_cols and self.models['winner'] is not None:
                if hasattr(self.models['winner'], 'feature_names_in_'):
                    self.feature_cols = list(self.models['winner'].feature_names_in_)
                elif hasattr(self.models['winner'], 'feature_importances_'):
                    # Create placeholder feature names
                    n_features = len(self.models['winner'].feature_importances_)
                    self.feature_cols = [f'feature_{i}' for i in range(n_features)]
            
            # Load team profiles
            if TEAM_PROFILES_FILE.exists():
                with open(TEAM_PROFILES_FILE, 'r', encoding='utf-8') as f:
                    self.team_profiles = json.load(f)
                print(f"Loaded {len(self.team_profiles)} team profiles")
            else:
                # Create minimal default profiles
                self.team_profiles = {
                    "1": {"name": "Home Team", "stats": {"ppg": 110.0}},
                    "2": {"name": "Away Team", "stats": {"ppg": 105.0}}
                }
            
            print("Basketball bundle loaded successfully")
            print(f"Feature columns: {len(self.feature_cols)}")
            self._loaded = True
        except Exception as e:
            print(f"Error loading basketball bundle: {e}")
            import traceback
            traceback.print_exc()
            self._loaded = False
    
    def _get_default_feature_columns(self):
        """Get default feature columns for NBA predictions"""
        return [
            'home_win_pct', 'away_win_pct', 'home_ppg', 'away_ppg',
            'home_ppg_allowed', 'away_ppg_allowed', 'home_fg_pct', 'away_fg_pct',
            'home_three_pct', 'away_three_pct', 'home_reb_margin', 'away_reb_margin',
            'home_streak', 'away_streak', 'home_days_rest', 'away_days_rest',
            'diff_win_pct', 'diff_ppg', 'diff_ppg_allowed', 'diff_fg_pct'
        ]
    
    def predict_game(self, hometeam_id: str, awayteam_id: str, date: str = None) -> Dict[str, Any]:
        """
        Predict NBA game outcome
        
        Args:
            hometeam_id: Home team ID
            awayteam_id: Away team ID
            date: Game date
            
        Returns:
            Prediction dictionary with winner, scores, and stats
        """
        if not self._loaded:
            return {"error": "Models not loaded", "status": "error"}
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Get team profiles
        home_profile = self.team_profiles.get(hometeam_id, {"name": f"Team {hometeam_id}", "stats": {}})
        away_profile = self.team_profiles.get(awayteam_id, {"name": f"Team {awayteam_id}", "stats": {}})
        
        # Get team names
        home_name = home_profile.get('name', f"Team {hometeam_id}")
        away_name = away_profile.get('name', f"Team {awayteam_id}")
        
        # Construct features for prediction
        features = self._construct_features(hometeam_id, awayteam_id, date, home_profile, away_profile)
        
        # Create DataFrame with correct columns
        X = pd.DataFrame([features], columns=self.feature_cols).fillna(0)
        
        # Make predictions
        prediction_result = {
            "match_info": {
                "date": date,
                "hometeam_id": hometeam_id,
                "hometeam_name": home_name,
                "awayteam_id": awayteam_id,
                "awayteam_name": away_name
            },
            "profiles": {
                "home": home_profile,
                "away": away_profile
            }
        }
        
        try:
            # Winner prediction
            if self.models.get('winner') is not None:
                if hasattr(self.models['winner'], 'predict_proba'):
                    winner_prob = self.models['winner'].predict_proba(X.values)[0]
                    winner_pred = self.models['winner'].predict(X.values)[0]
                    
                    is_home_winner = bool(winner_pred == 1)
                    confidence = float(winner_prob[1] if winner_pred == 1 else winner_prob[0]) * 100
                else:
                    # Fallback if no probability method
                    winner_pred = self.models['winner'].predict(X.values)[0]
                    is_home_winner = bool(winner_pred == 1)
                    confidence = 65.0 if is_home_winner else 35.0
                
                prediction_result["prediction"] = {
                    "winner": home_name if is_home_winner else away_name,
                    "home_wins": is_home_winner,
                    "confidence": round(confidence, 1)
                }
            else:
                # Fallback prediction
                prediction_result["prediction"] = {
                    "winner": home_name,
                    "home_wins": True,
                    "confidence": 55.0
                }
            
            # Score prediction (if available)
            if self.models.get('scoring') is not None:
                score_pred = self.models['scoring'].predict(X.values)
                
                # Handle different output formats
                if isinstance(score_pred, np.ndarray):
                    if len(score_pred.shape) == 2:
                        home_score = int(score_pred[0, 0]) if score_pred.shape[1] >= 1 else 105
                        away_score = int(score_pred[0, 1]) if score_pred.shape[1] >= 2 else 100
                    else:
                        home_score = int(score_pred[0]) if len(score_pred) >= 1 else 105
                        away_score = int(score_pred[1]) if len(score_pred) >= 2 else 100
                else:
                    home_score = 105
                    away_score = 100
            else:
                # Default scores based on team stats
                home_ppg = home_profile.get('stats', {}).get('ppg', 110.0)
                away_ppg = away_profile.get('stats', {}).get('ppg', 105.0)
                home_score = int(home_ppg)
                away_score = int(away_ppg)
            
            # Apply home court advantage
            home_score += 3
            away_score -= 1
            
            if 'prediction' not in prediction_result:
                prediction_result["prediction"] = {}
            
            prediction_result["prediction"]["predicted_score"] = {
                "home": home_score,
                "away": away_score
            }
            
            # Add comparison stats
            prediction_result["comparison"] = self._compare_teams(home_profile, away_profile)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback prediction
            prediction_result["prediction"] = {
                "winner": home_name,
                "home_wins": True,
                "confidence": 55.0,
                "predicted_score": {
                    "home": 105,
                    "away": 100
                }
            }
            prediction_result["comparison"] = self._compare_teams(home_profile, away_profile)
        
        return prediction_result
    
    def _construct_features(self, hometeam_id: str, awayteam_id: str, date: str,
                           home_profile: Dict, away_profile: Dict) -> Dict[str, float]:
        """
        Construct feature vector based on team profiles
        """
        # Initialize all features to 0
        features = {col: 0.0 for col in self.feature_cols}
        
        # Extract stats from profiles
        home_stats = home_profile.get('stats', {})
        away_stats = away_profile.get('stats', {})
        
        # Common feature mappings
        feature_values = {
            # Win percentages
            'home_win_pct': home_profile.get('win_pct', 0.5),
            'away_win_pct': away_profile.get('win_pct', 0.5),
            'diff_win_pct': home_profile.get('win_pct', 0.5) - away_profile.get('win_pct', 0.5),
            
            # Offensive stats
            'home_ppg': home_stats.get('ppg', 110.0),
            'away_ppg': away_stats.get('ppg', 105.0),
            'home_fg_pct': home_stats.get('fg_pct', 0.475),
            'away_fg_pct': away_stats.get('fg_pct', 0.465),
            'home_three_pct': home_stats.get('three_pct', 0.365),
            'away_three_pct': away_stats.get('three_pct', 0.355),
            
            # Defensive stats
            'home_ppg_allowed': home_stats.get('ppg_allowed', 108.0),
            'away_ppg_allowed': away_stats.get('ppg_allowed', 107.0),
            
            # Rebounding
            'home_reb_margin': home_stats.get('reb_margin', 1.5),
            'away_reb_margin': away_stats.get('reb_margin', 1.0),
            
            # Other stats
            'home_streak': home_profile.get('streak', 0),
            'away_streak': away_profile.get('streak', 0),
            
            # Differentials
            'diff_ppg': home_stats.get('ppg', 110.0) - away_stats.get('ppg', 105.0),
            'diff_ppg_allowed': home_stats.get('ppg_allowed', 108.0) - away_stats.get('ppg_allowed', 107.0),
            'diff_fg_pct': home_stats.get('fg_pct', 0.475) - away_stats.get('fg_pct', 0.465),
        }
        
        # Update features with actual values
        for key, value in feature_values.items():
            if key in features:
                features[key] = value
        
        return features
    
    def _compare_teams(self, home: Dict, away: Dict) -> Dict[str, Any]:
        """Generate comparison stats for display"""
        home_stats = home.get('stats', {})
        away_stats = away.get('stats', {})
        
        return {
            "record": {
                "home": f"{home.get('wins', 0)}-{home.get('losses', 0)}",
                "away": f"{away.get('wins', 0)}-{away.get('losses', 0)}"
            },
            "offense": {
                "ppg": {"home": home_stats.get('ppg', 0), "away": away_stats.get('ppg', 0)},
                "fg_pct": {"home": home_stats.get('fg_pct', 0), "away": away_stats.get('fg_pct', 0)},
            }
        }


# Global basketball predictor instance
basketball_predictor = BasketballPredictor()


def get_basketball_predictor() -> BasketballPredictor:
    """Get the global basketball predictor instance"""
    return basketball_predictor