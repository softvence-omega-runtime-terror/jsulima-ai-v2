from fastapi import APIRouter, HTTPException, Depends
import logging
from typing import List, Dict
import json
import os

from app.schemas.lineup import MatchRequest, LineupPredictionResponse
from app.models.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["lineup_prediction"]
)

def get_model_manager():
    """Dependency to access the global model manager"""
    from app.main import model_manager
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return model_manager

def get_preprocessor():
    """Dependency to access the data preprocessor"""
    return DataPreprocessor()

def load_nba_roster() -> Dict[str, List[str]]:
    """Load NBA team rosters from JSON file"""
    roster_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'nba_roster.json')
    try:
        with open(roster_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading roster: {e}")
        raise HTTPException(status_code=500, detail="Could not load player roster")

@router.post(
    "/predict",
    response_model=LineupPredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input"},
        503: {"description": "Model not available"}
    }
)
async def predict_match_lineup(
    request: MatchRequest,
    mm=Depends(get_model_manager),
    preprocessor=Depends(get_preprocessor)
) -> LineupPredictionResponse:
    """
    Predict player lineups for a match based on match details and team rosters
    
    Request body:
    - date: Match date (e.g., "Jan 21, 2026")
    - timezone: Match timezone (e.g., "EDT")
    - status: Match status (e.g., "Not Started")
    - time: Match time (e.g., "10:00 PM")
    - venue_name: Venue name
    - attendance: Attendance (optional)
    - id: Match ID
    - hometeam: Team name
    - awayteam: Team name
    
    Returns:
    - home_starters: List of 5 predicted starting players for home team
    - away_starters: List of 5 predicted starting players for away team
    """
    try:
        roster = load_nba_roster()
        
        # Get players for both teams
        home_players = roster.get(request.hometeam.name, [])
        away_players = roster.get(request.awayteam.name, [])
        
        if not home_players:
            raise HTTPException(
                status_code=400,
                detail=f"Team '{request.hometeam.name}' not found in roster"
            )
        if not away_players:
            raise HTTPException(
                status_code=400,
                detail=f"Team '{request.awayteam.name}' not found in roster"
            )
        
        # Prepare player data with match context
        match_context = {
            'date': request.date,
            'timezone': request.timezone,
            'status': request.status,
            'time': request.time,
            'venue_name': request.venue_name,
        }
        
        # Create player records for preprocessing
        players_data = []
        
        for player_name in home_players:
            players_data.append({
                'player_name': player_name,
                'team_name': request.hometeam.name,
                'is_home': 1,
                **match_context
            })
        
        for player_name in away_players:
            players_data.append({
                'player_name': player_name,
                'team_name': request.awayteam.name,
                'is_home': 0,
                **match_context
            })
        
        # Preprocess features
        features_df = preprocessor.preprocess_match_data(players_data, match_context)[0]
        
        if len(features_df) == 0:
            raise HTTPException(status_code=400, detail="Could not preprocess players")
        
        # Make predictions
        predictions = mm.predict(features_df)
        probabilities = mm.predict_proba(features_df)
        
        # Separate predictions by team
        home_predictions = []
        away_predictions = []
        
        for idx, player_data in enumerate(players_data):
            player_info = {
                'name': player_data['player_name'],
                'probability': float(probabilities[idx])
            }
            
            if player_data['is_home']:
                home_predictions.append(player_info)
            else:
                away_predictions.append(player_info)
        
        # Sort by probability and get top 5
        home_predictions.sort(key=lambda x: x['probability'], reverse=True)
        away_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        home_starters = [p['name'] for p in home_predictions[:5]]
        away_starters = [p['name'] for p in away_predictions[:5]]
        
        return LineupPredictionResponse(
            match_id=request.id,
            date=request.date,
            venue=request.venue_name,
            hometeam=request.hometeam.name,
            awayteam=request.awayteam.name,
            home_starters=home_starters,
            away_starters=away_starters
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

@router.get("/health")
async def health_check(mm=Depends(get_model_manager)):
    """Check if model is loaded and ready"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(mm.model).__name__
    }
