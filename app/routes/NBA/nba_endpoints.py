from fastapi import APIRouter, HTTPException, Query
from typing import List

from app.routes.NBA.nba_service import predict_upcoming_from_goalserve
from app.schemas.nba_schemas import NBAGamePrediction

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
