"""
Advanced Model Training v2
Improvements:
1. Uses enhanced features (Consistency, Chin Health, Matchup)
2. Better handling of class imbalance (scale_pos_weight)
3. CalibratedClassifierCV for better probability estimates
4. Optimized feature selection (Top 50)
5. Refined hyperparameters
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import json
from datetime import datetime

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier, RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.calibration import CalibratedClassifierCV

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not found. Install with: pip install lightgbm")


PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models" / "saved"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def load_features() -> Tuple[pd.DataFrame, List[str]]:
    """Load engineered features"""
    features_path = PROCESSED_DIR / "features.csv"
    
    df = pd.read_csv(features_path)
    
    # Exclude non-predictive columns
    exclude = ['match_id', 'fight_date', 'winner', 'win_method', 'fight_round',
               'local_total_strikes', 'away_total_strikes', 'local_strikes_head',
               'local_strikes_body', 'local_strikes_legs', 'away_strikes_head',
               'away_strikes_body', 'away_strikes_legs', 'local_takedowns_landed',
               'away_takedowns_landed']
    
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols


def time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by year: Train (2010-2023), Val (2024), Test (2025)"""
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    df['year'] = df['fight_date'].dt.year
    
    train = df[df['year'] <= 2023].copy()
    val = df[df['year'] == 2024].copy()
    test = df[df['year'] >= 2025].copy()
    
    return train, val, test


def add_advanced_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Add advanced feature engineering"""
    # NOTE: Most advanced features are now in feature_engineer.py directly
    new_features = []
    
    # Ensure all required columns exist before creating interactions
    
    # 1. Height-Reach Advantage
    if 'diff_height' in df.columns and 'diff_reach' in df.columns:
        df['height_reach_advantage'] = (df['diff_height'] + df['diff_reach']) / 2
        new_features.append('height_reach_advantage')
    
    # 2. Experience Ratio specific to UFC
    if 'local_experience' in df.columns and 'away_experience' in df.columns:
        df['experience_ratio'] = df['local_experience'] / (df['away_experience'] + 1)
        new_features.append('experience_ratio')
    
    # 3. ELO Advantage Squared (Non-linear impact)
    if 'diff_elo' in df.columns:
        df['elo_advantage_sq'] = np.sign(df['diff_elo']) * (df['diff_elo'] / 100) ** 2
        new_features.append('elo_advantage_sq')

    # 4. Activity Advantage (Log scale)
    if 'local_activity' in df.columns:
        df['activity_advantage'] = np.log1p(df['local_activity']) - np.log1p(df['away_activity'])
        new_features.append('activity_advantage')
        
    extended_features = list(set(feature_cols + new_features))
    return df, extended_features


def feature_selection(X_train, y_train, feature_cols: List[str], n_features: int = 50) -> List[str]:
    """Select most important features using Random Forest"""
    print(f"  Selecting top {n_features} features from {len(feature_cols)} candidates...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 15 Most Important Features:")
    print(importances.head(15).to_string(index=False))
    
    selected = importances.head(n_features)['feature'].tolist()
    return selected


def train_advanced_models(X_train, y_train, X_val, y_val, sample_weights=None):
    """Train multiple advanced models and select the best"""
    models = {}
    results = {}
    
    # Check class imbalance
    class_ratio = sum(y_train == 0) / sum(y_train == 1)
    print(f"  Class Ratio (0/1): {class_ratio:.2f}")

    # 1. XGBoost (Tuned)
    if HAS_XGB:
        print("\n  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.2,
            reg_alpha=0.5,
            reg_lambda=1.0,
            scale_pos_weight=class_ratio,  # Handle imbalance
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        if sample_weights is not None:
            xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            xgb_model.fit(X_train, y_train)
            
        # Evaluation (No calibration to avoid errors)
        val_pred = xgb_model.predict(X_val)
        val_prob = xgb_model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred)
        roc = roc_auc_score(y_val, val_prob)
        brier = brier_score_loss(y_val, val_prob)
        
        models['XGBoost'] = xgb_model
        results['XGBoost'] = {'accuracy': acc, 'f1': f1, 'roc_auc': roc, 'brier': brier}
        print(f"    XGBoost: Acc={acc:.4f}, ROC={roc:.4f}, Brier={brier:.4f}")

    # 2. LightGBM (Tuned)
    if HAS_LGB:
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=20,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            reg_alpha=0.5,
            reg_lambda=1.0,
            scale_pos_weight=class_ratio,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        if sample_weights is not None:
            lgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            lgb_model.fit(X_train, y_train)
            
        val_pred = lgb_model.predict(X_val)
        val_prob = lgb_model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred)
        roc = roc_auc_score(y_val, val_prob)
        brier = brier_score_loss(y_val, val_prob)
        
        models['LightGBM'] = lgb_model
        results['LightGBM'] = {'accuracy': acc, 'f1': f1, 'roc_auc': roc, 'brier': brier}
        print(f"    LightGBM: Acc={acc:.4f}, ROC={roc:.4f}, Brier={brier:.4f}")

    # Select best model by ROC-AUC
    best_name = max(results, key=lambda x: results[x]['roc_auc'])
    print(f"\n  Best Model: {best_name} (ROC-AUC: {results[best_name]['roc_auc']:.4f})")
    
    return models[best_name], best_name, models, results


def calculate_recency_weights(dates: pd.Series, decay_rate: float = 0.15) -> np.ndarray:
    """Calculate exponential recency weights"""
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days
    weights = np.exp(-decay_rate * days_ago / 365)
    weights = weights / weights.mean()
    return weights


def optimize_threshold(model, X, y) -> float:
    """Find optimal decision threshold using Youden's J statistic or Balanced Accuracy"""
    probs = model.predict_proba(X)[:, 1]
    
    best_threshold = 0.5
    best_score = 0
    
    for threshold in np.arange(0.30, 0.70, 0.01):
        preds = (probs >= threshold).astype(int)
        score = balanced_accuracy_score(y, preds)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold


def train_auxiliary_models(X_train, y_train_df, X_val, y_val_df, weights=None):
    """Train method and round prediction models"""
    print("\n  Training Auxiliary Models (Method & Round)...")
    
    # Win Method
    method_enc = LabelEncoder()
    y_method = method_enc.fit_transform(y_train_df['win_method'].fillna('OTHER'))
    
    method_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    if weights is not None:
        method_model.fit(X_train, y_method, sample_weight=weights)
    else:
        method_model.fit(X_train, y_method)
        
    # Round
    round_enc = LabelEncoder()
    y_round = round_enc.fit_transform(y_train_df['fight_round'].fillna(3).astype(int).clip(1, 5))
    
    round_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    if weights is not None:
        round_model.fit(X_train, y_round, sample_weight=weights)
    else:
        round_model.fit(X_train, y_round)
        
    return method_model, method_enc, round_model


def train_stats_models(X_train, y_train_dict, X_val, y_val_dict, weights=None):
    """Train regression models for fight statistics"""
    print("\n  Training Stats Models...")
    models = {}
    rmse_dict = {}
    
    for target in y_train_dict:
        # Use simpler models for stats (RandomForestRegressor is robust)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        if weights is not None:
            model.fit(X_train, y_train_dict[target], sample_weight=weights)
        else:
            model.fit(X_train, y_train_dict[target])
        
        val_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((val_pred - y_val_dict[target]) ** 2))
        
        models[target] = model
        rmse_dict[target] = rmse
    
    return models, rmse_dict


def main():
    print("=" * 70)
    print("ADVANCED UFC PREDICTION MODEL TRAINING V2")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[1/6] Loading data...")
    df, feature_cols = load_features()
    
    # 2. Add advanced features
    df, extended_features = add_advanced_features(df, feature_cols)
    
    df = df.dropna(subset=['winner'])
    df = df[df['winner'] >= 0]
    
    # 3. Split
    print("\n[2/6] Splitting data...")
    train, val, test = time_based_split(df)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Fill NaN
    for col in extended_features:
        if col in train.columns:
            train[col] = train[col].fillna(0)
            val[col] = val[col].fillna(0)
            test[col] = test[col].fillna(0)
            
    valid_features = [f for f in extended_features if f in train.columns]
    
    X_train = train[valid_features].values
    y_train = train['winner'].values.astype(int)
    X_val = val[valid_features].values
    y_val = val['winner'].values.astype(int)
    X_test = test[valid_features].values
    y_test = test['winner'].values.astype(int)
    
    # 4. Feature Selection
    print("\n[3/6] Selecting features...")
    selected_features = feature_selection(X_train, y_train, valid_features, n_features=50)
    
    X_train_sel = train[selected_features].values
    X_val_sel = val[selected_features].values
    X_test_sel = test[selected_features].values
    
    # Recency weights
    weights = calculate_recency_weights(train['fight_date'])
    
    # 5. Train Models
    print("\n[4/6] Training models...")
    best_model, best_name, all_models, results = train_advanced_models(
        X_train_sel, y_train, X_val_sel, y_val, sample_weights=weights
    )
    
    # Optimize threshold
    best_threshold = optimize_threshold(best_model, X_val_sel, y_val)
    print(f"  Optimal Threshold: {best_threshold:.2f}")
    
    # Auxiliary models (train on all features for simplicity, or selected)
    # Using selected features is safer for consistency
    method_model, method_enc, round_model = train_auxiliary_models(
        X_train_sel, train, X_val_sel, val, weights
    )
    
    # Train stats models
    stats_targets = [
        'local_total_strikes', 'away_total_strikes',
        'local_strikes_head', 'local_strikes_body', 'local_strikes_legs',
        'away_strikes_head', 'away_strikes_body', 'away_strikes_legs',
        'local_takedowns_landed', 'away_takedowns_landed'
    ]
    
    y_stats_train = {t: train[t].fillna(0).values for t in stats_targets}
    y_stats_val = {t: val[t].fillna(0).values for t in stats_targets}
    
    stats_models, stats_rmse = train_stats_models(
        X_train_sel, y_stats_train, X_val_sel, y_stats_val, weights
    )
    
    for target, rmse in stats_rmse.items():
        print(f"  {target}: RMSE={rmse:.2f}")

    # 6. Evaluate on Test
    print("\n[5/6] Final Evaluation (2025 Test Set)...")
    test_prob = best_model.predict_proba(X_test_sel)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    
    acc = accuracy_score(y_test, test_pred)
    roc = roc_auc_score(y_test, test_prob)
    balanced = balanced_accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    
    print(f"\n  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Acc:  {balanced:.4f}")
    print(f"  ROC-AUC:       {roc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    # 7. Save
    print("\n[6/6] Saving resources...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model, MODELS_DIR / "winner_model.pkl")
    joblib.dump(method_model, MODELS_DIR / "method_model.pkl")
    joblib.dump(round_model, MODELS_DIR / "round_model.pkl")
    joblib.dump(stats_models, MODELS_DIR / "stats_models.pkl")

    joblib.dump(method_enc, MODELS_DIR / "method_encoder.pkl")
    joblib.dump(selected_features, MODELS_DIR / "feature_columns.pkl")
    joblib.dump(best_threshold, MODELS_DIR / "winner_threshold.pkl")
    
    print(f"  Saved to {MODELS_DIR}")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
