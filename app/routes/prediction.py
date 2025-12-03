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

router = APIRouter(prefix="", tags=["UFC Predictions"])
logger = get_logger(__name__)


# @router.post(
#     "/predict",
#     response_model=PredictionResponse,
#     responses={500: {"model": ErrorResponse}},
#     summary="Predict UFC Fight Winner",
#     description="Predict the winner of a single UFC fight based on fighter statistics"
# )
# async def predict_fight(request: PredictionRequest):
#     """
#     Predict the winner of a UFC fight.
#     
#     Returns prediction with confidence scores and probabilities for both fighters.
#     """
#     try:
#         predictor = get_predictor()
#         
#         # Convert Pydantic model to dict
#         fighter_stats = request.fighter_stats.dict()
#         
#         # Make prediction
#         result = predictor.predict_single(fighter_stats)
#         
#         logger.info(f"Prediction made: {result['winner']} with confidence {result['confidence']:.2%}")
#         
#         return result
#         
#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#             detail=str(e)
#         )
#     except Exception as e:
#         logger.exception("Prediction failed")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Prediction failed: {str(e)}"
#         )


# @router.post(
#     "/predict/batch",
#     response_model=BatchPredictionResponse,
#     responses={500: {"model": ErrorResponse}},
#     summary="Batch Predict UFC Fights",
#     description="Predict winners for multiple UFC fights in a single request"
# )
# async def predict_fights_batch(request: BatchPredictionRequest):
#     """
#     Predict winners for multiple UFC fights.
#     
#     Returns predictions for all fights with success/failure counts.
#     """
#     try:
#         predictor = get_predictor()
#         
#         # Convert Pydantic models to dicts
#         fights = [fight.dict() for fight in request.fights]
#         
#         # Make batch predictions
#         predictions = predictor.predict_batch(fights)
#         
#         # Count successful and failed predictions
#         successful = sum(1 for p in predictions if 'error' not in p)
#         failed = len(predictions) - successful
#         
#         logger.info(f"Batch prediction: {successful} successful, {failed} failed out of {len(predictions)} total")
#         
#         return {
#             "predictions": predictions,
#             "total": len(predictions),
#             "successful": successful,
#             "failed": failed
#         }
#         
#     except Exception as e:
#         logger.exception("Batch prediction failed")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Batch prediction failed: {str(e)}"
#         )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Get Model Information",
    description="Get information about the loaded UFC prediction model including metrics and feature importance"
)
async def get_model_info(include_features: bool = True):
    """
    Get information about the loaded UFC prediction model.
    
    Args:
        include_features: Whether to include feature importance in response
    
    Returns model type, metrics, and optionally feature importance.
    """
    try:
        predictor = get_predictor()
        
        # Get basic model info
        info = predictor.get_model_info()
        
        # Add feature importance if requested
        if include_features:
            try:
                feature_importance = predictor.get_feature_importance(top_n=15)
                info['feature_importance'] = feature_importance
            except AttributeError:
                # Model doesn't support feature importance
                info['feature_importance'] = None
        
        logger.info("Model info retrieved successfully")
        
        return info
        
    except Exception as e:
        logger.exception("Failed to get model info")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


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
    "/predict/",
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


@router.get(
    "/health",
    summary="Health Check",
    description="Check if the UFC prediction service is healthy and model is loaded"
)
async def health_check():
    """
    Health check endpoint to verify the service is running and model is loaded.
    """
    try:
        predictor = get_predictor()
        
        return {
            "status": "healthy",
            "model_loaded": predictor.model is not None,
            "scaler_loaded": predictor.scaler is not None,
            "features_loaded": predictor.feature_names is not None
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }