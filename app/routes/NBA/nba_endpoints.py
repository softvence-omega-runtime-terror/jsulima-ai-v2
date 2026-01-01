from fastapi import APIRouter, HTTPException, Query
from typing import List

from app.routes.NBA.nba_service import predict_upcoming_from_goalserve, predict_specific_game
from app.schemas.nba_schemas import (
    NBAGameOverview,
    NBAGamePrediction,
    NBAPredictionOnly,
    NBAStatsBreakdown,
)

router = APIRouter()

@router.get("/upcoming-games", response_model=List[NBAGamePrediction], 
            summary="Get Upcoming NBA Game Predictions",
            description="""
This endpoint retrieves upcoming NBA games and generates predictions for match outcome, score, and point differential using a trained Machine Learning pipeline.

**Model Prediction Flow:**

1.  **Data Ingestion**: Upcoming match data is fetched from Goalserve API, relative to historical team/player stats.
2.  **Feature Engineering**: Player stats are aggregated into team metrics (ORTG, DRTG) with rolling averages for recent form.
3.  **Contextual Analysis**: Incorporates Elo ratings, rest days, back-to-back status, and home-court advantage.
4.  **Win Probability**: Random Forest Classifier predicts home win probability; away probability is 1 - home.
5.  **Score Forecasting**: Random Forest Regressors predict total points and spread to derive final scores.
""")
async def get_nba_predictions(limit: int = Query(10, description="Number of upcoming games to predict")):
    """
    Get upcoming NBA game predictions using the trained models.
    """
    try:
        predictions = predict_upcoming_from_goalserve(limit=limit)
        return predictions

    except Exception as e:
        # Log the error potentially
        print(f"Error predicting NBA games: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate predictions: {str(e)}")


@router.get(
    "/upcoming-games/overview",
    response_model=NBAGameOverview,
    summary="Get game overview and probabilities for a specific NBA match",
)
async def get_nba_game_overview(
    match_date: str = Query(..., description="Scheduled date for the match (YYYY-MM-DD)"),
    home_team_id: int = Query(..., description="Home team ID"),
    away_team_id: int = Query(..., description="Away team ID"),
):
    try:
        prediction = predict_specific_game(match_date, home_team_id, away_team_id)
        return {
            "game_overview": prediction.get("game_overview"),
            "model_probabilities": prediction.get("model_probabilities"),
        }
    except Exception as e:
        print(f"Error generating game overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch game overview: {str(e)}")


@router.get(
    "/upcoming-games/prediction",
    response_model=NBAPredictionOnly,
    summary="Get score and winner prediction for a specific NBA match",
)
async def get_nba_game_prediction(
    match_date: str = Query(..., description="Scheduled date for the match (YYYY-MM-DD)"),
    home_team_id: int = Query(..., description="Home team ID"),
    away_team_id: int = Query(..., description="Away team ID"),
):
    try:
        prediction = predict_specific_game(match_date, home_team_id, away_team_id)
        return {"prediction": prediction.get("prediction")}
    except Exception as e:
        print(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch prediction: {str(e)}")


@router.get(
    "/upcoming-games/stats",
    response_model=NBAStatsBreakdown,
    summary="Get statistical breakdown for a specific NBA match",
)
async def get_nba_game_stats(
    match_date: str = Query(..., description="Scheduled date for the match (YYYY-MM-DD)"),
    home_team_id: int = Query(..., description="Home team ID"),
    away_team_id: int = Query(..., description="Away team ID"),
):
    try:
        prediction = predict_specific_game(match_date, home_team_id, away_team_id)
        return {
            "core_home": prediction.get("core_home"),
            "core_away": prediction.get("core_away"),
            "efficiency_home": prediction.get("efficiency_home"),
            "efficiency_away": prediction.get("efficiency_away"),
            "lineup_bench": prediction.get("lineup_bench"),
        }
    except Exception as e:
        print(f"Error generating stats breakdown: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats breakdown: {str(e)}")
