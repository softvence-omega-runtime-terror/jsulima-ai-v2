from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class GameOverview(BaseModel):
    matchup: str
    date: str
    venue: str
    home_team: str
    away_team: str

class ModelProbabilities(BaseModel):
    home_win_prob_pct: Optional[float]
    away_win_prob_pct: Optional[float]
    model_confidence_0_100: Optional[float]
    ai_confidence_pct: Optional[float]

class CoreStats(BaseModel):
    season_win_pct: Optional[float]
    ppg: Optional[float]
    opp_ppg: Optional[float]
    net_rating: Optional[float]
    fg_pct: Optional[float]
    three_pt_pct: Optional[float]
    ft_pct: Optional[float]
    efg_pct: Optional[float]

class EfficiencyStats(BaseModel):
    ortg: Optional[float]
    drtg: Optional[float]
    pace: Optional[float]
    tov_pct: Optional[float]
    orb_pct: Optional[float]
    drb_pct: Optional[float]
    recent_ortg_5: Optional[float]
    recent_drtg_5: Optional[float]
    recent_net_5: Optional[float]

class LineupBench(BaseModel):
    home_bench_ppg: Optional[float]
    away_bench_ppg: Optional[float]
    home_rotation_size: Optional[float]
    away_rotation_size: Optional[float]
    home_top5_minutes_avg: Optional[float]
    away_top5_minutes_avg: Optional[float]

class PredictedScore(BaseModel):
    home: int
    away: int

class Prediction(BaseModel):
    predicted_winner: str
    predicted_score: PredictedScore
    predicted_total_points: Optional[float]
    predicted_home_point_diff: Optional[float]

class NBAGamePrediction(BaseModel):
    game_overview: GameOverview
    model_probabilities: ModelProbabilities
    core_home: CoreStats
    core_away: CoreStats
    efficiency_home: EfficiencyStats
    efficiency_away: EfficiencyStats
    lineup_bench: LineupBench
    prediction: Prediction


