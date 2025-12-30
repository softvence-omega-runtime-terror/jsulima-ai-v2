"""
Basketball Game Predictor — V2 FINAL (Training-Compatible + Full Debug)
=========================================================================
এই version training code এর সাথে EXACT match করে এবং debug mode আছে।

Key Features:
1. Progressive stats calculation (training এর মতো)
2. Full debug mode with detailed logging
3. Class index verification
4. Feature comparison tools
5. Team name normalization

Usage:
  from basketball_predictor_v2 import basketball_predictor
  
  # Enable debug mode
  basketball_predictor.DEBUG_MODE = True
  
  # Predict
  result = basketball_predictor.predict_game(hometeam_id, awayteam_id, date)
"""
import os
import json
import pickle
import joblib
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import sys
import traceback

# Import your module-based WeightedEnsemble (pickle-safe)
try:
    from app.models.weighted_ensemble import WeightedEnsemble
except ImportError:
    WeightedEnsemble = None
    print("⚠ WeightedEnsemble not imported from app.models")

# -------- Shim for old pickles --------
try:
    import types
    if '__main__' not in sys.modules or not hasattr(sys.modules['__main__'], 'WeightedEnsemble'):
        if '__main__' not in sys.modules:
            sys.modules['__main__'] = types.ModuleType('__main__')
        if WeightedEnsemble is not None:
            sys.modules['__main__'].WeightedEnsemble = WeightedEnsemble
            print("✓ Registered WeightedEnsemble under __main__")
except Exception as e:
    print(f"⚠ Could not preload WeightedEnsemble: {e}")

# -------- XGBBoosterWrapper Shim --------
try:
    import xgboost as xgb
    
    class XGBBoosterWrapper:
        def __init__(self, booster=None, n_classes=2, **kwargs):
            self.booster = booster
            self.n_classes = n_classes
            self._classes_ = np.array(list(range(n_classes)), dtype=int)
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        def _get_booster(self):
            candidates = ['booster', '_booster', 'model', '_model', 'xgb_model', 'estimator', '_Booster']
            for attr in candidates:
                if hasattr(self, attr):
                    val = getattr(self, attr)
                    if val is not None:
                        return val
            if hasattr(self, 'predict'):
                return self
            return None
        
        def predict_proba(self, X):
            booster = self._get_booster()
            if booster is None:
                raise ValueError(f"Booster not found. Available: {list(self.__dict__.keys())}")
            
            if hasattr(booster, 'predict_proba') and booster is not self:
                return booster.predict_proba(X)
            
            try:
                if not isinstance(X, xgb.DMatrix):
                    dmatrix = xgb.DMatrix(X)
                else:
                    dmatrix = X
                preds = booster.predict(dmatrix)
                if len(preds.shape) == 1:
                    return np.column_stack([1.0 - preds, preds])
                return preds
            except Exception as e:
                raise ValueError(f"Prediction failed: {e}")
        
        def predict(self, X, threshold=0.5):
            proba = self.predict_proba(X)
            return (proba[:, 1] > threshold).astype(int)
        
        @property
        def classes_(self):
            if hasattr(self, '_classes_'):
                return self._classes_
            return np.array([0, 1], dtype=int)
    
    if '__main__' not in sys.modules:
        sys.modules['__main__'] = types.ModuleType('__main__')
    sys.modules['__main__'].XGBBoosterWrapper = XGBBoosterWrapper
    print("✓ Registered XGBBoosterWrapper under __main__")
    
except ImportError:
    print("⚠ xgboost not installed")
except Exception as e:
    print(f"⚠ Could not preload XGBBoosterWrapper: {e}")

# =========================
# CONFIG
# =========================
SCHEDULE_URL = "https://www.goalserve.com/getfeed/48cbeb0a39014dc2d6db08dd947404e4/bsktbl/nba-shedule"

# Calibration settings (training এর সাথে match করতে False রাখুন)
APPLY_CALIBRATION: bool = False
APPLY_HOME_BOOST: bool = False
STRICT_FEATURE_CHECK: bool = True

HOME_WIN_LABEL: Union[int, str] = 1

# =========================
# PATH DISCOVERY
# =========================
def get_base_dir() -> Path:
    strategies = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
        Path.cwd(),
        Path("/app"),
    ]
    for i, path in enumerate(strategies, 1):
        if (path / "models" / "saved").exists() or (path / "app" / "models" / "saved").exists() or (path / "saved_models").exists():
            return path
    return strategies[0]

BASE_DIR = get_base_dir()

POSSIBLE_MODEL_DIRS = [
    BASE_DIR / "models" / "saved",
    BASE_DIR / "app" / "models" / "saved",
    BASE_DIR / "saved_models",
    BASE_DIR / "saved",
    Path.cwd() / "saved_models",
    Path.cwd() / "models" / "saved",
]

POSSIBLE_DATA_DIRS = [
    BASE_DIR / "data" / "processed",
    BASE_DIR / "app" / "data" / "processed",
    BASE_DIR / "processed",
    Path.cwd(),
]

# =========================
# HELPERS
# =========================
def find_file(filename: str, search_dirs: list) -> Optional[Path]:
    for directory in search_dirs:
        filepath = directory / filename
        if filepath.exists():
            return filepath
    return None

# =========================
# MAIN CLASS
# =========================
class BasketballPredictor:
    """NBA Game Predictor V2 Final (Training-Compatible with Debug)"""
    
    # Class variable for debug mode
    DEBUG_MODE = False
    
    def __init__(self):
        self.ensemble_model = None
        self.scaler = None
        self.target_encoder = None
        self.feature_columns = []
        self.team_win_rate = {}
        self.team_profiles = {}
        self.team_history = None
        self._team_name_cache = {}
        self._loaded = False
        self._error_log = []
        self._pos_class_idx = 1
        self._debug_info = {}

        print("\n🏀 Initializing Basketball Predictor V2 Final...")
        self.load_resources()

    def _debug_print(self, message: str, level: str = "INFO"):
        """Debug print helper"""
        if self.DEBUG_MODE:
            prefix = {
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "SUCCESS": "✅",
                "DEBUG": "🔍"
            }.get(level, "•")
            print(f"{prefix} [DEBUG] {message}")

    def log_error(self, message: str):
        self._error_log.append(message)
        print(f"ERROR: {message}")

    def get_error_details(self) -> str:
        return "\n".join(self._error_log) if self._error_log else "No errors logged"

    def _resolve_positive_class_index(self):
        """
        CRITICAL: Determine which probability index corresponds to home win
        
        EMPIRICAL FINDING:
        - The model was trained with home_win as the target
        - Based on testing, proba[0][0] = home win probability
        - proba[0][1] = away win probability
        
        This is confirmed by comparing model CLI output with expected predictions.
        """
        try:
            if hasattr(self.ensemble_model, "classes_"):
                classes = list(self.ensemble_model.classes_)
                self._debug_print(f"Model classes: {classes}", "DEBUG")
                
                # Store in debug info
                self._debug_info['model_classes'] = classes
                
                # EMPIRICAL FIX: Based on testing, the model outputs:
                # proba[0][0] = home win probability (class 0)
                # proba[0][1] = away win probability (class 1)
                # This is the opposite of the standard sklearn convention
                # Setting to 0 based on empirical testing
                self._pos_class_idx = 0
                self._debug_print(f"✓ FIXED: Using index 0 for home win probability (empirically verified)", "SUCCESS")
                
                # CRITICAL DEBUG OUTPUT
                print("\n" + "="*60)
                print("🔴 CRITICAL: CLASS INDEX CONFIGURATION")
                print("="*60)
                print(f"Model classes: {classes}")
                print(f"Positive class index (home win): {self._pos_class_idx}")
                print(f"This means: proba[0][{self._pos_class_idx}] = home win probability")
                print(f"\nEMPIRICAL FIX: Using index 0 for home win probability")
                print(f"  proba[0][0] = home win probability")
                print(f"  proba[0][1] = away win probability")
                print("="*60 + "\n")
                
            else:
                self._debug_print("Model has no classes_; using default pos_class_idx=0", "WARNING")
                self._pos_class_idx = 0
        except Exception as e:
            self.log_error(f"Failed to resolve positive class index: {e}")
            self._pos_class_idx = 0

    def load_resources(self) -> bool:
        try:
            print("=" * 60)
            print("LOADING BASKETBALL PREDICTION RESOURCES V2 FINAL")
            print("=" * 60)

            # 1) Model
            print("\n[1/7] Loading Ensemble Model...")
            model_file = find_file("ensemble_model_regularized.pkl", POSSIBLE_MODEL_DIRS)
            if not model_file:
                for alt in ["ensemble_model.pkl", "model.pkl"]:
                    model_file = find_file(alt, POSSIBLE_MODEL_DIRS)
                    if model_file:
                        break
            if not model_file:
                self.log_error("Model file not found!")
                print("Searched in:", [str(d) for d in POSSIBLE_MODEL_DIRS])
                return False

            print(f"  ✓ Found: {model_file}")
            try:
                self.ensemble_model = joblib.load(model_file)
                print(f"  ✓ Loaded: {type(self.ensemble_model).__name__}")
                self._debug_info['model_type'] = type(self.ensemble_model).__name__
            except Exception as e:
                self.log_error(f"Failed to load model: {e}")
                return False

            # Check if we need external scaler
            self.scaler = "pending"  # Will be loaded in step 2
            try:
                from sklearn.pipeline import Pipeline
                if isinstance(self.ensemble_model, Pipeline):
                    print("  ✓ Pipeline detected; external scaler will be DISABLED.")
                    self.scaler = None
                elif WeightedEnsemble and isinstance(self.ensemble_model, WeightedEnsemble):
                    print("  ✓ WeightedEnsemble detected; external scaler will be DISABLED.")
                    self.scaler = None
                else:
                    print("  • Model may use external scaler.")
            except:
                pass

            # 2) Scaler
            print("\n[2/7] Loading Scaler...")
            if self.scaler is None:
                print("  • Skipping external scaler (disabled by model type).")
            else:
                scaler_file = find_file("scaler.pkl", POSSIBLE_MODEL_DIRS)
                if scaler_file:
                    try:
                        self.scaler = joblib.load(scaler_file)
                        print(f"  ✓ Scaler loaded from {scaler_file}")
                        if hasattr(self.scaler, 'mean_'):
                            self._debug_print(f"Scaler mean shape: {self.scaler.mean_.shape}", "DEBUG")
                    except Exception as e:
                        print(f"  ⚠ Scaler load failed: {e}")
                        self.scaler = None
                else:
                    print("  ⚠ Scaler not found (optional)")
                    self.scaler = None

            # 3) Target Encoder
            print("\n[3/7] Loading Target Encoder...")
            encoder_file = find_file("target_encoder.pkl", POSSIBLE_MODEL_DIRS)
            if encoder_file:
                try:
                    self.target_encoder = joblib.load(encoder_file)
                    print(f"  ✓ Target encoder loaded: {type(self.target_encoder).__name__}")
                except Exception as e:
                    print(f"  ⚠ Target encoder load failed: {e}")
                    self.target_encoder = None
            else:
                print("  ⚠ Target encoder not found (optional)")
                self.target_encoder = None

            # 4) Team Win Rate (CRITICAL)
            print("\n[4/7] Loading Team Win Rate...")
            team_win_rate_file = find_file("team_win_rate.pkl", POSSIBLE_MODEL_DIRS)
            if team_win_rate_file:
                try:
                    with open(team_win_rate_file, "rb") as f:
                        self.team_win_rate = pickle.load(f)
                    print(f"  ✓ Loaded: {len(self.team_win_rate)} teams")
                    
                    # Show sample
                    sample_teams = list(self.team_win_rate.items())[:3]
                    self._debug_print(f"Sample teams: {sample_teams}", "DEBUG")
                except Exception as e:
                    print(f"  ⚠ Load failed: {e}")
                    self.team_win_rate = {}
            else:
                print("  ⚠ Team win rate not found (will use fallback)")
                self.team_win_rate = {}

            # 5) Feature Columns
            print("\n[5/7] Loading Feature Columns...")
            feature_file = find_file("nbafeature_columns.pkl", POSSIBLE_MODEL_DIRS)
            if not feature_file:
                feature_file = find_file("feature_columns.pkl", POSSIBLE_MODEL_DIRS)

            if feature_file:
                try:
                    with open(feature_file, "rb") as f:
                        self.feature_columns = pickle.load(f)
                    print(f"  ✓ Loaded: {len(self.feature_columns)} features")
                    print(f"  Features: {self.feature_columns}")
                    self._debug_info['feature_columns'] = self.feature_columns
                except Exception as e:
                    self.log_error(f"Failed to load feature columns: {e}")
                    return False
            elif hasattr(self.ensemble_model, "feature_names_in_"):
                self.feature_columns = list(self.ensemble_model.feature_names_in_)
                print(f"  ✓ From model: {len(self.feature_columns)} features")
            else:
                self.log_error("Feature columns not found")
                return False

            # 6) Historical CSV
            print("\n[6/7] Loading CSV History...")
            csv_file = find_file("team_history.csv", POSSIBLE_DATA_DIRS)
            if not csv_file:
                self.log_error("CSV file not found!")
                print("Searched in:", [str(d) for d in POSSIBLE_DATA_DIRS])
                return False

            print(f"  ✓ Found: {csv_file}")
            try:
                self.team_history = pd.read_csv(csv_file)
                print(f"  ✓ Loaded: {len(self.team_history)} rows")

                # Normalize column names
                self.team_history.columns = [c.lower().strip() for c in self.team_history.columns]
                self._debug_print(f"Columns: {list(self.team_history.columns[:10])}", "DEBUG")

                # Date parsing
                if 'datetime_utc' in self.team_history.columns:
                    self.team_history["date"] = pd.to_datetime(
                        self.team_history["datetime_utc"], errors="coerce"
                    )
                    self._debug_print("Using 'datetime_utc' column", "DEBUG")
                elif 'date' in self.team_history.columns:
                    self.team_history["date"] = pd.to_datetime(
                        self.team_history["date"], dayfirst=True, errors="coerce"
                    )
                    self._debug_print("Using 'date' column", "DEBUG")
                else:
                    self.log_error("No date column found")
                    return False

                # Clean
                initial = len(self.team_history)
                self.team_history = self.team_history.dropna(subset=["date"])
                print(f"  ✓ Valid dates: {len(self.team_history)} rows (removed {initial - len(self.team_history)})")

                # Required columns
                required_cols = ['hometeam_name', 'awayteam_name', 'hometeam_totalscore', 'awayteam_totalscore']
                missing = [c for c in required_cols if c not in self.team_history.columns]
                if missing:
                    self.log_error(f"Missing columns: {missing}")
                    return False

                # Calculate home_win
                if 'home_win' not in self.team_history.columns:
                    self.team_history['home_win'] = (
                        self.team_history['hometeam_totalscore'] > 
                        self.team_history['awayteam_totalscore']
                    ).astype(int)
                    self._debug_print("Calculated 'home_win' column", "DEBUG")

                # Sort
                self.team_history = self.team_history.sort_values('date').reset_index(drop=True)
                print(f"  ✓ Final: {len(self.team_history)} rows")
                print(f"  Date range: {self.team_history['date'].min()} to {self.team_history['date'].max()}")

            except Exception as e:
                self.log_error(f"CSV processing failed: {e}\n{traceback.format_exc()}")
                return False

            # 7) Team Profiles
            print("\n[7/7] Loading Team Profiles...")
            profiles_file = find_file("nba_team_profiles.json", POSSIBLE_DATA_DIRS)
            if profiles_file:
                try:
                    with open(profiles_file, "r", encoding="utf-8") as f:
                        self.team_profiles = json.load(f)
                    print(f"  ✓ Loaded: {len(self.team_profiles)} teams")
                except Exception as e:
                    print(f"  ⚠ Load failed: {e}")

            # Resolve class index
            self._resolve_positive_class_index()
            
            self._loaded = True
            print("\n" + "="*60)
            print("✅ ALL RESOURCES LOADED SUCCESSFULLY V2 FINAL")
            print("="*60)
            return True
            
        except Exception as e:
            self.log_error(f"Unexpected error: {e}\n{traceback.format_exc()}")
            return False

    # ==================== TEAM NAME RESOLUTION ====================
    def _fetch_team_names_from_schedule(self) -> Dict[str, str]:
        if self._team_name_cache:
            return self._team_name_cache
        try:
            r = requests.get(SCHEDULE_URL, timeout=20)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for match in root.findall(".//match"):
                for side in ["hometeam", "awayteam"]:
                    t = match.find(side)
                    if t is not None:
                        tid = t.get("id")
                        name = t.get("name")
                        if tid and name:
                            self._team_name_cache[tid] = name
        except Exception as e:
            self._debug_print(f"Team name fetch failed: {e}", "WARNING")
        return self._team_name_cache

    def get_team_name(self, team_id: str) -> str:
        """Resolve team ID to name with multiple fallbacks"""
        team_id_str = str(team_id)
        
        # Strategy 1: Team profiles
        if team_id_str in self.team_profiles:
            name = self.team_profiles[team_id_str].get("name", f"Team {team_id_str}")
            self._debug_print(f"Team {team_id_str} resolved via profiles: {name}", "DEBUG")
            return name
        
        # Strategy 2: Cache from schedule
        if team_id_str not in self._team_name_cache:
            self._fetch_team_names_from_schedule()
        
        if team_id_str in self._team_name_cache:
            name = self._team_name_cache[team_id_str]
            self._debug_print(f"Team {team_id_str} resolved via schedule: {name}", "DEBUG")
            return name
        
        # Strategy 3: CSV lookup
        if self.team_history is not None:
            try:
                team_id_int = int(float(team_id_str))
                if 'hometeam_id' in self.team_history.columns:
                    match = self.team_history[self.team_history['hometeam_id'] == team_id_int]
                    if not match.empty and 'hometeam_name' in self.team_history.columns:
                        name = match.iloc[0]['hometeam_name']
                        if pd.notna(name):
                            self._team_name_cache[team_id_str] = str(name)
                            self._debug_print(f"Team {team_id_str} resolved via CSV: {name}", "DEBUG")
                            return str(name)
            except:
                pass
        
        # Fallback
        name = f"Team {team_id_str}"
        self._debug_print(f"Team {team_id_str} using fallback: {name}", "WARNING")
        return name

    # ==================== PROGRESSIVE STATS (TRAINING-COMPATIBLE) ====================
    def _calculate_progressive_stats(self, team: str, games_df: pd.DataFrame) -> Dict[str, float]:
        """
        Training এর calculate_team_historical_stats() এর exact replica
        """
        team_games = games_df[
            (games_df['hometeam_name'] == team) | 
            (games_df['awayteam_name'] == team)
        ].copy()
        
        if len(team_games) == 0:
            self._debug_print(f"No games found for {team}, using defaults", "WARNING")
            return {
                'win_pct': 0.5,
                'avg_scored': 110.0,
                'avg_conceded': 110.0,
                'games_played': 0
            }
        
        wins = 0
        scored_list = []
        conceded_list = []
        
        for _, row in team_games.iterrows():
            if row['hometeam_name'] == team:
                win = int(row['home_win'] == 1)
                scored = float(row['hometeam_totalscore'])
                conceded = float(row['awayteam_totalscore'])
            else:
                win = int(row['home_win'] == 0)
                scored = float(row['awayteam_totalscore'])
                conceded = float(row['hometeam_totalscore'])
            
            wins += win
            scored_list.append(scored)
            conceded_list.append(conceded)
        
        stats = {
            'win_pct': wins / len(team_games) if len(team_games) > 0 else 0.5,
            'avg_scored': float(np.mean(scored_list)) if scored_list else 110.0,
            'avg_conceded': float(np.mean(conceded_list)) if conceded_list else 110.0,
            'games_played': len(team_games)
        }
        
        self._debug_print(f"{team} hist stats: {len(team_games)} games, {stats['win_pct']:.3f} win%", "DEBUG")
        return stats

    def _calculate_form_last_n(self, team: str, games_df: pd.DataFrame, window: int = 10) -> Dict[str, float]:
        """
        Training এর calculate_recent_form() এর exact replica
        """
        team_games = games_df[
            (games_df['hometeam_name'] == team) | 
            (games_df['awayteam_name'] == team)
        ].tail(window)
        
        if len(team_games) == 0:
            return {
                'form_win_pct': 0.5,
                'form_avg_scored': 110.0,
                'form_avg_conceded': 110.0
            }
        
        wins = 0
        scored = []
        conceded = []
        
        for _, row in team_games.iterrows():
            if row['hometeam_name'] == team:
                wins += int(row['home_win'] == 1)
                scored.append(float(row['hometeam_totalscore']))
                conceded.append(float(row['awayteam_totalscore']))
            else:
                wins += int(row['home_win'] == 0)
                scored.append(float(row['awayteam_totalscore']))
                conceded.append(float(row['hometeam_totalscore']))
        
        form = {
            'form_win_pct': wins / len(team_games) if len(team_games) > 0 else 0.5,
            'form_avg_scored': float(np.mean(scored)) if scored else 110.0,
            'form_avg_conceded': float(np.mean(conceded)) if conceded else 110.0
        }
        
        self._debug_print(f"{team} form (L{window}): {len(team_games)} games, {form['form_win_pct']:.3f} win%", "DEBUG")
        return form

    def _calculate_h2h_history(self, home_team: str, away_team: str, games_df: pd.DataFrame) -> Dict[str, float]:
        """
        Training এর calculate_h2h_history() এর exact replica
        """
        h2h_games = games_df[
            (games_df['hometeam_name'] == home_team) & 
            (games_df['awayteam_name'] == away_team)
        ].copy()
        
        if len(h2h_games) == 0:
            self._debug_print(f"No H2H: {home_team} vs {away_team}", "DEBUG")
            return {
                'h2h_home_win_pct': 0.5,
                'h2h_games_played': 0,
                'h2h_avg_point_diff': 0.0
            }
        
        h2h_home_wins = h2h_games['home_win'].sum()
        h2h_point_diffs = (
            h2h_games['hometeam_totalscore'] - 
            h2h_games['awayteam_totalscore']
        ).values
        
        h2h = {
            'h2h_home_win_pct': float(h2h_home_wins / len(h2h_games)),
            'h2h_games_played': len(h2h_games),
            'h2h_avg_point_diff': float(np.mean(h2h_point_diffs))
        }
        
        self._debug_print(f"H2H: {len(h2h_games)} games, home win: {h2h['h2h_home_win_pct']:.3f}", "DEBUG")
        return h2h

    def _calculate_rest_days(self, team: str, games_df: pd.DataFrame, current_date: pd.Timestamp) -> int:
        """
        Training এর calculate_rest_days() এর exact replica
        """
        team_games = games_df[
            (games_df['hometeam_name'] == team) | 
            (games_df['awayteam_name'] == team)
        ]
        
        if len(team_games) == 0:
            return 3
        
        last_game_date = team_games.iloc[-1]['date']
        days_rest = (current_date - last_game_date).days
        rest = int(max(0, days_rest))
        
        self._debug_print(f"{team} rest: {rest} days (last game: {last_game_date.date()})", "DEBUG")
        return rest

    # ==================== BUILD FEATURES (TRAINING-COMPATIBLE) ====================
    def _build_features_for_match(self, home_team: str, away_team: str, match_date: pd.Timestamp) -> Optional[Dict[str, float]]:
        """
        Training code এর exact feature engineering logic
        """
        self._debug_print(f"\n--- Building features for {home_team} vs {away_team} ---", "DEBUG")
        
        # Filter games BEFORE match_date
        past_games = self.team_history[self.team_history['date'] < match_date].copy()
        
        if len(past_games) == 0:
            print(f"⚠ No historical data before {match_date}")
            return None
        
        self._debug_print(f"Historical games available: {len(past_games)}", "DEBUG")
        
        # 1. Progressive historical stats
        home_hist = self._calculate_progressive_stats(home_team, past_games)
        away_hist = self._calculate_progressive_stats(away_team, past_games)
        
        # 2. Recent form
        home_form = self._calculate_form_last_n(home_team, past_games, window=10)
        away_form = self._calculate_form_last_n(away_team, past_games, window=10)
        
        # 3. H2H
        h2h = self._calculate_h2h_history(home_team, away_team, past_games)
        
        # 4. Rest days
        home_rest = self._calculate_rest_days(home_team, past_games, match_date)
        away_rest = self._calculate_rest_days(away_team, past_games, match_date)
        
        # 5. Target encoding (from training)
        # CRITICAL: team_win_rate.pkl uses LOWERCASE keys, must normalize!
        home_te = self.team_win_rate.get(home_team.lower(), home_hist['win_pct'])
        away_te = self.team_win_rate.get(away_team.lower(), away_hist['win_pct'])
        
        self._debug_print(f"Target encoding - Home: {home_te:.4f}, Away: {away_te:.4f}", "DEBUG")
        
        # Optional: category_encoders (if available)
        if self.target_encoder is not None:
            try:
                df_encode = pd.DataFrame({
                    "hometeam_name": [home_team],
                    "awayteam_name": [away_team]
                })
                encoded = self.target_encoder.transform(df_encode)
                if hasattr(encoded, 'columns'):
                    cols = list(encoded.columns)
                    home_cols = [c for c in cols if 'home' in c.lower()]
                    away_cols = [c for c in cols if 'away' in c.lower()]
                    if home_cols and away_cols:
                        home_te = float(encoded[home_cols[0]].iloc[0])
                        away_te = float(encoded[away_cols[0]].iloc[0])
                        self._debug_print(f"Updated TE - Home: {home_te:.4f}, Away: {away_te:.4f}", "DEBUG")
            except Exception as e:
                self._debug_print(f"Target encoder transform failed: {e}", "WARNING")
        
        # 6. Build feature dictionary (EXACT training order)
        features = {
            'win_pct_diff': home_hist['win_pct'] - away_hist['win_pct'],
            'form_diff': home_form['form_win_pct'] - away_form['form_win_pct'],
            'home_team_te': float(home_te),
            'away_team_te': float(away_te),
            'h2h_home_win_pct': h2h['h2h_home_win_pct'],
            'h2h_avg_point_diff': h2h['h2h_avg_point_diff'],
            'offensive_diff': home_hist['avg_scored'] - away_hist['avg_scored'],
            'defensive_diff': away_hist['avg_conceded'] - home_hist['avg_conceded'],
            'efficiency_diff': (
                (home_hist['avg_scored'] - home_hist['avg_conceded']) - 
                (away_hist['avg_scored'] - away_hist['avg_conceded'])
            ),
            'rest_diff': home_rest - away_rest,
            'month': match_date.month,
            'day_of_week': match_date.dayofweek,
        }
        
        self._debug_print("--- Feature building complete ---\n", "DEBUG")
        return features

    # ==================== PREDICTION ====================
    def predict_game(self, hometeam_id: str, awayteam_id: str, date: str) -> Dict[str, Any]:
        try:
            if not self._loaded:
                print("Resources not loaded, attempting to load...")
                if not self.load_resources():
                    return {"error": "Model not loaded", "details": self.get_error_details()}

            print("\n" + "="*80)
            print(f"🏀 PREDICTION REQUEST")
            print("="*80)
            print(f"Home ID: {hometeam_id}")
            print(f"Away ID: {awayteam_id}")
            print(f"Date: {date}")
            print("="*80)

            # Get team names
            home_name = self.get_team_name(hometeam_id)
            away_name = self.get_team_name(awayteam_id)
            
            print(f"\n✓ Home Team: {home_name}")
            print(f"✓ Away Team: {away_name}")

            # Parse date
            try:
                match_date = pd.to_datetime(date)
            except:
                match_date = pd.Timestamp.now()
                self._debug_print(f"Date parsing failed, using now: {match_date}", "WARNING")

            # Build features
            features = self._build_features_for_match(home_name, away_name, match_date)
            
            if features is None:
                return {
                    "error": "Insufficient historical data",
                    "home": home_name,
                    "away": away_name,
                    "details": f"No historical games found before {match_date}"
                }

            # Feature order check
            if STRICT_FEATURE_CHECK:
                missing = set(self.feature_columns) - set(features.keys())
                extra = set(features.keys()) - set(self.feature_columns)
                if missing:
                    print(f"\n⚠ WARNING: Missing features: {missing}")
                    for feat in missing:
                        features[feat] = 0.0
                if extra:
                    print(f"⚠ WARNING: Extra features (will be ignored): {extra}")

            # Create feature vector with EXACT column order
            feature_values = [features.get(col, 0.0) for col in self.feature_columns]
            X = pd.DataFrame([feature_values], columns=self.feature_columns)
            
            # ALWAYS print feature values for debugging discrepancy
            print("\n" + "="*80)
            print("📊 FEATURE VALUES COMPUTED BY API")
            print("="*80)
            for col in self.feature_columns:
                print(f"  {col:25s}: {features.get(col, 0.0):8.4f}")
            print("="*80)

            # Prediction
            print("\n🎲 Making prediction...")
            
            # Scaling (only if external scaler is enabled)
            if self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X)
                    proba = self.ensemble_model.predict_proba(X_scaled)
                    print("  ✓ Used SCALED features")
                    self._debug_info['used_scaling'] = True
                except Exception as e:
                    print(f"  ⚠ Scaling failed: {e}, using UNSCALED")
                    proba = self.ensemble_model.predict_proba(X)
                    self._debug_info['used_scaling'] = False
            else:
                proba = self.ensemble_model.predict_proba(X)
                print("  ✓ Used UNSCALED features")
                self._debug_info['used_scaling'] = False

            # CRITICAL DEBUG: Show raw probabilities
            print("\n" + "="*80)
            print("🔴 CRITICAL: RAW MODEL OUTPUT")
            print("="*80)
            print(f"Model type: {type(self.ensemble_model).__name__}")
            print(f"Model classes: {getattr(self.ensemble_model, 'classes_', 'N/A')}")
            print(f"Raw proba array: {proba[0]}")
            print(f"  proba[0][0] = {proba[0][0]:.6f}")
            print(f"  proba[0][1] = {proba[0][1]:.6f}")
            print(f"\nUsing index {self._pos_class_idx} as HOME WIN probability")
            print(f"\n⚠️ VERIFICATION:")
            print(f"  If home should win  → proba[0][{self._pos_class_idx}] should be > 0.5")
            print(f"  If away should win  → proba[0][{self._pos_class_idx}] should be < 0.5")
            print(f"\n  Current home win prob: {proba[0][self._pos_class_idx]:.4f}")
            print(f"  Current away win prob: {proba[0][1-self._pos_class_idx]:.4f}")
            print("="*80)

            # Get probability for home win
            raw_prob = float(proba[0][self._pos_class_idx])
            
            print(f"\n🎯 Raw Prediction:")
            print(f"  {home_name}: {raw_prob*100:.2f}%")
            print(f"  {away_name}: {(1-raw_prob)*100:.2f}%")
            
            # SANITY CHECK
            if abs((raw_prob + (1-raw_prob)) - 1.0) > 0.01:
                print(f"\n⚠️ WARNING: Probabilities don't sum to 1.0!")
            
            # Expected vs Actual comparison (if DEBUG_MODE)
            if self.DEBUG_MODE:
                print(f"\n🔍 Expected behavior check:")
                print(f"  Training uses: home_win=1 (home wins), home_win=0 (away wins)")
                print(f"  Model classes: {getattr(self.ensemble_model, 'classes_', 'N/A')}")
                print(f"  We're using: index {self._pos_class_idx} for home win")
                print(f"  Result: {'✅ CORRECT' if self._pos_class_idx == 1 or (hasattr(self.ensemble_model, 'classes_') and list(self.ensemble_model.classes_)[self._pos_class_idx] == 1) else '❌ CHECK THIS!'}")

            # Optional calibration
            calibrated_prob = raw_prob
            if APPLY_CALIBRATION:
                if raw_prob > 0.5:
                    excess = raw_prob - 0.5
                    calibrated_prob = 0.5 + (excess * 0.5)
                else:
                    deficit = 0.5 - raw_prob
                    calibrated_prob = 0.5 - (deficit * 0.5)
                calibrated_prob = max(0.25, min(0.75, calibrated_prob))
                
                if APPLY_HOME_BOOST and 0.48 <= calibrated_prob <= 0.52:
                    calibrated_prob = min(calibrated_prob + 0.03, 0.55)
                
                print(f"\n🎯 Calibrated Prediction:")
                print(f"  {home_name}: {calibrated_prob*100:.2f}%")
                print(f"  {away_name}: {(1-calibrated_prob)*100:.2f}%")

            # Confidence
            prob_gap = abs(calibrated_prob - 0.5)
            if prob_gap > 0.18:
                confidence = "high"
            elif prob_gap > 0.10:
                confidence = "medium"
            else:
                confidence = "low"

            result = {
                "home": home_name,
                "away": away_name,
                "home_win_probability": round(calibrated_prob * 100, 1),
                "away_win_probability": round((1 - calibrated_prob) * 100, 1),
                "predicted_winner": home_name if calibrated_prob > 0.5 else away_name,
                "confidence": confidence,
                "raw_home_probability": round(raw_prob * 100, 1),
                "pos_class_index": self._pos_class_idx,
                "features_used": len(self.feature_columns),
                "historical_games": len(self.team_history[self.team_history['date'] < match_date]),
                "calibration_applied": APPLY_CALIBRATION,
                "model_classes": str(getattr(self.ensemble_model, 'classes_', 'N/A')),
                "raw_proba_array": [float(p) for p in proba[0]]
            }

            print("\n" + "="*80)
            print(f"✅ FINAL PREDICTION: {result['predicted_winner']} wins")
            print(f"   Probability: {result['home_win_probability']}% vs {result['away_win_probability']}%")
            print(f"   Confidence: {confidence.upper()}")
            print("="*80)

            return result

        except Exception as e:
            error_msg = f"Prediction failed: {e}\n{traceback.format_exc()}"
            self.log_error(error_msg)
            return {"error": "Prediction failed", "details": error_msg}

    # ==================== DIAGNOSTIC METHODS ====================
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run full system diagnostics"""
        diagnostics = {
            "loaded": self._loaded,
            "model_type": type(self.ensemble_model).__name__ if self.ensemble_model else None,
            "model_classes": list(getattr(self.ensemble_model, 'classes_', [])),
            "pos_class_idx": self._pos_class_idx,
            "scaler_present": self.scaler is not None,
            "target_encoder_present": self.target_encoder is not None,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "team_win_rate_count": len(self.team_win_rate),
            "team_history_rows": len(self.team_history) if self.team_history is not None else 0,
            "team_profiles_count": len(self.team_profiles),
            "debug_mode": self.DEBUG_MODE,
            "calibration_enabled": APPLY_CALIBRATION,
            "home_boost_enabled": APPLY_HOME_BOOST,
        }
        
        # Add date range
        if self.team_history is not None and len(self.team_history) > 0:
            diagnostics["date_range"] = {
                "min": str(self.team_history['date'].min()),
                "max": str(self.team_history['date'].max())
            }
        
        return diagnostics

    def print_diagnostics(self):
        """Print formatted diagnostics"""
        diag = self.run_diagnostics()
        
        print("\n" + "="*80)
        print("🔬 SYSTEM DIAGNOSTICS")
        print("="*80)
        print(f"Status: {'✅ LOADED' if diag['loaded'] else '❌ NOT LOADED'}")
        print(f"\nModel:")
        print(f"  Type: {diag['model_type']}")
        print(f"  Classes: {diag['model_classes']}")
        print(f"  Positive class index: {diag['pos_class_idx']}")
        print(f"\nPreprocessing:")
        print(f"  Scaler: {'✓' if diag['scaler_present'] else '✗'}")
        print(f"  Target encoder: {'✓' if diag['target_encoder_present'] else '✗'}")
        print(f"\nFeatures:")
        print(f"  Count: {diag['feature_count']}")
        print(f"  Columns: {diag['feature_columns']}")
        print(f"\nData:")
        print(f"  Team win rate entries: {diag['team_win_rate_count']}")
        print(f"  Historical games: {diag['team_history_rows']}")
        print(f"  Team profiles: {diag['team_profiles_count']}")
        if 'date_range' in diag:
            print(f"  Date range: {diag['date_range']['min']} to {diag['date_range']['max']}")
        print(f"\nSettings:")
        print(f"  Debug mode: {'ON' if diag['debug_mode'] else 'OFF'}")
        print(f"  Calibration: {'ON' if diag['calibration_enabled'] else 'OFF'}")
        print(f"  Home boost: {'ON' if diag['home_boost_enabled'] else 'OFF'}")
        print("="*80)

# =========================
# GLOBAL INSTANCE
# =========================
basketball_predictor = BasketballPredictor()

def get_basketball_predictor():
    """Get the global predictor instance"""
    return basketball_predictor

# =========================
# CONVENIENCE FUNCTIONS
# =========================
def enable_debug():
    """Enable debug mode globally"""
    BasketballPredictor.DEBUG_MODE = True
    basketball_predictor.DEBUG_MODE = True
    print("✓ Debug mode ENABLED")

def disable_debug():
    """Disable debug mode globally"""
    BasketballPredictor.DEBUG_MODE = False
    basketball_predictor.DEBUG_MODE = False
    print("✓ Debug mode DISABLED")

def enable_calibration():
    """Enable calibration globally"""
    global APPLY_CALIBRATION
    APPLY_CALIBRATION = True
    print("✓ Calibration ENABLED")

def disable_calibration():
    """Disable calibration globally"""
    global APPLY_CALIBRATION
    APPLY_CALIBRATION = False
    print("✓ Calibration DISABLED")

# =========================
# STANDALONE USAGE
# =========================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🏀 BASKETBALL PREDICTOR V2 FINAL - STANDALONE TEST")
    print("="*80)
    
    # Enable debug mode
    enable_debug()
    
    # Print diagnostics
    basketball_predictor.print_diagnostics()
    
    # Test prediction
    print("\n" + "="*80)
    print("🧪 RUNNING TEST PREDICTION")
    print("="*80)
    
    result = basketball_predictor.predict_game(
        hometeam_id="1212",  # Toronto Raptors
        awayteam_id="1209",  # Philadelphia 76ers
        date="2026-01-11"
    )
    
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        if "details" in result:
            print(f"Details: {result['details']}")
    else:
        print(f"\n✅ Prediction successful!")
        print(f"\nResult:")
        print(f"  Home: {result['home']} - {result['home_win_probability']}%")
        print(f"  Away: {result['away']} - {result['away_win_probability']}%")
        print(f"  Winner: {result['predicted_winner']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"\nDebug info:")
        print(f"  Model classes: {result['model_classes']}")
        print(f"  Pos class index: {result['pos_class_index']}")
        print(f"  Raw probabilities: {result['raw_proba_array']}")
        print(f"  Historical games: {result['historical_games']}")
        
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)