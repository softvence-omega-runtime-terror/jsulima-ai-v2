"""
UFC Prediction API Routes

This module defines the FastAPI endpoints for UFC fight predictions.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional

from app.schemas.ufc_prediction import (
    FighterStats,
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    FeatureImportance,
    ErrorResponse,
    PlayerPredictionRequest,
    PlayerPredictionResponse,
    TeamInfo
)
from app.models.predictor import get_predictor, UFCPredictor
from app.services.player_service import (
    get_fighter_stats_for_prediction,
    get_player_name_by_id
)
from app.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)



def convert_date_format(date_str: Optional[str]) -> Optional[str]:
    """
    Convert date from DD-MM-YYYY format to YYYY-MM-DD format.
    
    Args:
        date_str: Date string in DD-MM-YYYY format
    
    Returns:
        Date string in YYYY-MM-DD format, or None if date_str is None
    """
    if date_str is None:
        return None
    
    try:
        # Parse DD-MM-YYYY format
        parts = date_str.split('-')
        if len(parts) == 3:
            day, month, year = parts
            # Return in YYYY-MM-DD format
            return f"{year}-{month}-{day}"
        return date_str
    except Exception:
        # If parsing fails, return as is (might be in different format)
        return date_str


@router.post(
    "/",
    response_model=PlayerPredictionResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict UFC Fight Winner by Player IDs",
    description="Predict the winner of a UFC fight between two players using their IDs and historical statistics"
)
async def predict_fight_by_players(request: PlayerPredictionRequest):
    """
    Predict the winner of a UFC fight between two players.
    
    Uses historical statistics from both players to make a prediction.
    Returns nested objects with localteam and awayteam information.
    """
    try:
        predictor = get_predictor()
        
        # Convert date format from DD-MM-YYYY to YYYY-MM-DD for internal processing
        converted_date = convert_date_format(request.date)
        
        # Get player names
        try:
            localteam_name = get_player_name_by_id(request.localteam)
            awayteam_name = get_player_name_by_id(request.awayteam)
        except ValueError as e:
            logger.error(f"Player lookup failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        
        # Get fighter statistics based on historical data
        try:
            fighter_stats = get_fighter_stats_for_prediction(
                request.localteam,
                request.awayteam,
                date=converted_date
            )
        except ValueError as e:
            logger.error(f"Failed to get player statistics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        
        # Make prediction
        prediction_result = predictor.predict_single(fighter_stats)
        
        # Determine probabilities
        # prediction_result['prediction']: 1 = local wins, 0 = away wins
        localteam_prob = prediction_result['probability_local']
        awayteam_prob = prediction_result['probability_away']
        
        # Determine winner
        localteam_is_winner = prediction_result['prediction'] == 1
        awayteam_is_winner = prediction_result['prediction'] == 0
        
        # Build response
        result = {
            "localteam": {
                "id": request.localteam,
                "name": localteam_name,
                "win_probability": localteam_prob,
                "is_winner": localteam_is_winner
            },
            "awayteam": {
                "id": request.awayteam,
                "name": awayteam_name,
                "win_probability": awayteam_prob,
                "is_winner": awayteam_is_winner
            }
        }
        
        winner_name = localteam_name if localteam_is_winner else awayteam_name
        logger.info(
            f"Player prediction: {winner_name} wins with "
            f"{prediction_result['confidence']:.2%} confidence"
        )
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Player prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

