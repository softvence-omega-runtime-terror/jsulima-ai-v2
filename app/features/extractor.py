import pandas as pd
import numpy as np
from typing import Dict, Any, List

class FeatureExtractor:
    """
    Centralized logic for constructing feature, eliminating skew between 
    training and inference.
    """
    
    # Weight class encoding mapping
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

    @staticmethod
    def construct_match_features(
        local_feats: Dict[str, Any], 
        away_feats: Dict[str, Any], 
        weight_class: str,
        is_title_fight: bool = False
    ) -> Dict[str, Any]:
        """
        Construct the flat feature dictionary from two fighter stats objects.
        This is the SINGLE SOURCE OF TRUTH for feature engineering.
        """
        
        weight_class_encoded = FeatureExtractor.WEIGHT_CLASSES.get(weight_class, 5)
        
        features = {
            # Context
            'weight_class': weight_class_encoded,
            'is_title_fight': 1 if is_title_fight else 0,
            
            # ===== PHYSICAL FEATURES =====
            # Local
            'local_height': local_feats.get('height_inches', 70),
            'local_reach': local_feats.get('reach_inches', 72),
            'local_age': local_feats.get('age', 32),
            'local_is_southpaw': local_feats.get('is_southpaw', 0),
            
            # Away
            'away_height': away_feats.get('height_inches', 70),
            'away_reach': away_feats.get('reach_inches', 72),
            'away_age': away_feats.get('age', 32),
            'away_is_southpaw': away_feats.get('is_southpaw', 0),
            
            # Differentials
            'diff_height': local_feats.get('height_inches', 70) - away_feats.get('height_inches', 70),
            'diff_reach': local_feats.get('reach_inches', 72) - away_feats.get('reach_inches', 72),
            'diff_age': local_feats.get('age', 32) - away_feats.get('age', 32),
            'stance_matchup': 1 if local_feats.get('is_southpaw', 0) != away_feats.get('is_southpaw', 0) else 0,
            
            # ===== ENHANCED STATS =====
            # Local
            'local_slpm': local_feats.get('slpm', 0),
            'local_sapm': local_feats.get('sapm', 0),
            'local_td_acc': local_feats.get('td_acc', 0),
            'local_td_def': local_feats.get('td_def', 0),
            'local_avg_td': local_feats.get('avg_td', 0),
            'local_sub_att': local_feats.get('avg_sub_att', 0),
            'local_avg_kd': local_feats.get('avg_kd_landed', 0),
            'local_avg_ctrl': local_feats.get('avg_ctrl_time', 0),
            
            # Away
            'away_slpm': away_feats.get('slpm', 0),
            'away_sapm': away_feats.get('sapm', 0),
            'away_td_acc': away_feats.get('td_acc', 0),
            'away_td_def': away_feats.get('td_def', 0),
            'away_avg_td': away_feats.get('avg_td', 0),
            'away_sub_att': away_feats.get('avg_sub_att', 0),
            'away_avg_kd': away_feats.get('avg_kd_landed', 0),
            'away_avg_ctrl': away_feats.get('avg_ctrl_time', 0),
            
            # Differentials
            'diff_slpm': local_feats.get('slpm', 0) - away_feats.get('slpm', 0),
            'diff_sapm': local_feats.get('sapm', 0) - away_feats.get('sapm', 0),
            'diff_td_acc': local_feats.get('td_acc', 0) - away_feats.get('td_acc', 0),
            'diff_td_def': local_feats.get('td_def', 0) - away_feats.get('td_def', 0),
            'diff_avg_td': local_feats.get('avg_td', 0) - away_feats.get('avg_td', 0),
            'diff_sub_att': local_feats.get('avg_sub_att', 0) - away_feats.get('avg_sub_att', 0),
            'diff_avg_kd': local_feats.get('avg_kd_landed', 0) - away_feats.get('avg_kd_landed', 0),
            'diff_avg_ctrl': local_feats.get('avg_ctrl_time', 0) - away_feats.get('avg_ctrl_time', 0),

            # ===== HISTORICAL PERFORMANCE =====
            # Local
            'local_win_rate': local_feats.get('win_rate', 0),
            'local_experience': local_feats.get('experience', 0),
            'local_streak': local_feats.get('current_streak', 0),
            'local_elo': local_feats.get('elo_rating', 1500),
            'local_ko_rate': local_feats.get('ko_rate', 0),
            'local_sub_rate': local_feats.get('sub_rate', 0),
            'local_finish_rate': local_feats.get('finish_rate', 0),
            'local_ko_loss_rate': local_feats.get('ko_loss_rate', 0),
            'local_days_since': local_feats.get('days_since_last_fight', 365),
            'local_activity': local_feats.get('activity', 0),
            'local_l3_strikes': local_feats.get('l3_strikes_landed', 0),
            'local_l3_absorbed': local_feats.get('l3_strikes_absorbed', 0),
            'local_l3_td': local_feats.get('l3_takedowns_landed', 0),
            'local_l3_kd': local_feats.get('l3_knockdowns', 0),
            'local_l3_ctrl': local_feats.get('l3_ctrl_time', 0),
            'local_form': local_feats.get('form', 0),
            'local_head_ratio': local_feats.get('head_ratio', 0.5),
            'local_body_ratio': local_feats.get('body_ratio', 0.25),
            'local_consistency': local_feats.get('consistency', 20.0),
            'local_chin': local_feats.get('days_since_ko', 365*5),
            
            # Away
            'away_win_rate': away_feats.get('win_rate', 0),
            'away_experience': away_feats.get('experience', 0),
            'away_streak': away_feats.get('current_streak', 0),
            'away_elo': away_feats.get('elo_rating', 1500),
            'away_ko_rate': away_feats.get('ko_rate', 0),
            'away_sub_rate': away_feats.get('sub_rate', 0),
            'away_finish_rate': away_feats.get('finish_rate', 0),
            'away_ko_loss_rate': away_feats.get('ko_loss_rate', 0),
            'away_days_since': away_feats.get('days_since_last_fight', 365),
            'away_activity': away_feats.get('activity', 0),
            'away_l3_strikes': away_feats.get('l3_strikes_landed', 0),
            'away_l3_absorbed': away_feats.get('l3_strikes_absorbed', 0),
            'away_l3_td': away_feats.get('l3_takedowns_landed', 0),
            'away_l3_kd': away_feats.get('l3_knockdowns', 0),
            'away_l3_ctrl': away_feats.get('l3_ctrl_time', 0),
            'away_form': away_feats.get('form', 0),
            'away_head_ratio': away_feats.get('head_ratio', 0.5),
            'away_body_ratio': away_feats.get('body_ratio', 0.25),
            'away_consistency': away_feats.get('consistency', 20.0),
            'away_chin': away_feats.get('days_since_ko', 365*5),
            
            # Differentials
            'diff_win_rate': local_feats.get('win_rate', 0) - away_feats.get('win_rate', 0),
            'diff_experience': local_feats.get('experience', 0) - away_feats.get('experience', 0),
            'diff_streak': local_feats.get('current_streak', 0) - away_feats.get('current_streak', 0),
            'diff_elo': local_feats.get('elo_rating', 1500) - away_feats.get('elo_rating', 1500),
            'diff_ko_rate': local_feats.get('ko_rate', 0) - away_feats.get('ko_rate', 0),
            'diff_sub_rate': local_feats.get('sub_rate', 0) - away_feats.get('sub_rate', 0),
            'diff_finish_rate': local_feats.get('finish_rate', 0) - away_feats.get('finish_rate', 0),
            'diff_strikes': local_feats.get('l3_strikes_landed', 0) - away_feats.get('l3_strikes_landed', 0),
            'diff_absorbed': local_feats.get('l3_strikes_absorbed', 0) - away_feats.get('l3_strikes_absorbed', 0),
            'diff_td': local_feats.get('l3_takedowns_landed', 0) - away_feats.get('l3_takedowns_landed', 0),
            'diff_kd': local_feats.get('l3_knockdowns', 0) - away_feats.get('l3_knockdowns', 0),
            'diff_ctrl': local_feats.get('l3_ctrl_time', 0) - away_feats.get('l3_ctrl_time', 0),
            'diff_form': local_feats.get('form', 0) - away_feats.get('form', 0),
            'diff_activity': local_feats.get('activity', 0) - away_feats.get('activity', 0),
            'diff_consistency': local_feats.get('consistency', 0) - away_feats.get('consistency', 0),
            'diff_chin': local_feats.get('days_since_ko', 0) - away_feats.get('days_since_ko', 0),
            
            # Matchup
            'local_striker_score': local_feats.get('l3_strikes_landed', 0) - local_feats.get('l3_takedowns_landed', 0),
            'away_striker_score': away_feats.get('l3_strikes_landed', 0) - away_feats.get('l3_takedowns_landed', 0),
            
            # ===== OPPONENT QUALITY (NEW) =====
            'local_avg_opp_elo': local_feats.get('avg_opp_elo', 1500),
            'local_avg_opp_win_rate': local_feats.get('avg_opp_win_rate', 0.5),
            'away_avg_opp_elo': away_feats.get('avg_opp_elo', 1500),
            'away_avg_opp_win_rate': away_feats.get('avg_opp_win_rate', 0.5),
            
            'diff_avg_opp_elo': local_feats.get('avg_opp_elo', 1500) - away_feats.get('avg_opp_elo', 1500),
            'diff_avg_opp_win_rate': local_feats.get('avg_opp_win_rate', 0.5) - away_feats.get('avg_opp_win_rate', 0.5),
        }
        
        return features
