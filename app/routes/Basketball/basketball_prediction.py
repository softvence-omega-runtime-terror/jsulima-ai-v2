"""
Basketball Prediction Router
FastAPI endpoints for NBA game predictions with simplified request/response
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
from datetime import datetime
import traceback

from app.schemas.basketball_schemas import BasketballPredictionRequest
from app.services.Basketball.basketball_predictor import get_basketball_predictor


router = APIRouter()


def _format_time():
    """Get current time in HH:MM format"""
    return datetime.now().strftime("%H:%M")


@router.post("/predict")
async def predict_basketball_game(
    request: BasketballPredictionRequest = Body(
        ...,
        openapi_examples={
            "normal": {
                "summary": "Basketball Game Prediction",
                "value": {
                    "match_id": "311495",
                    "date": "12.04.2026",
                    "awayteam_id": "1225",
                    "hometeam_id": "1550"
                }
            }
        }
    )
):
    """
    Predict NBA game outcome using ML models
    
    Request body:
    - match_id: Match identifier
    - date: Game date (dd.MM.yyyy)
    - awayteam_id: Away team ID
    - hometeam_id: Home team ID
    
    Returns simplified JSON response with match details and team predictions.
    """
    try:
        predictor = get_basketball_predictor()
        
        # Convert date format if needed
        if request.date:
            # Parse dd.MM.yyyy to yyyy-mm-dd for internal use
            try:
                date_obj = datetime.strptime(request.date, "%d.%m.%Y")
                internal_date = date_obj.strftime("%Y-%m-%d")
            except:
                internal_date = datetime.now().strftime("%Y-%m-%d")
        else:
            internal_date = datetime.now().strftime("%Y-%m-%d")
        
        # Get prediction
        result = predictor.predict_game(
            hometeam_id=request.hometeam_id,
            awayteam_id=request.awayteam_id,
            date=internal_date
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        pred = result.get('prediction', {})
        match_info = result.get('match_info', {})
        
        # Determine winner and probabilities
        is_home_winner = pred.get('home_wins', True)
        confidence = pred.get('confidence', 50.0)
        
        # Calculate win probabilities
        if is_home_winner:
            home_prob = f"{confidence:.1f}%"
            away_prob = f"{100 - confidence:.1f}%"
        else:
            home_prob = f"{100 - confidence:.1f}%"
            away_prob = f"{confidence:.1f}%"
        
        # Get team names
        home_name = match_info.get('hometeam_name', f"Team {request.hometeam_id}")
        away_name = match_info.get('awayteam_name', f"Team {request.awayteam_id}")
        
        # Build the response in the requested format
        response = {
            "match": {
                "date": request.date,
                "time": _format_time(),
                "match_id": request.match_id,
                "awayteam": {
                    "awayteam_id": request.awayteam_id,
                    "awayteam_name": away_name,
                    "win_probability": away_prob,
                    "is_winner": not is_home_winner
                },
                "hometeam": {
                    "hometeam_id": request.hometeam_id,
                    "hometeam_name": home_name,
                    "win_probability": home_prob,
                    "is_winner": is_home_winner
                }
            }
        }
        
        return response
            
    except Exception as e:
        with open("basketball_debug_error.log", "a") as f:
            traceback.print_exc(file=f)
        raise HTTPException(status_code=500, detail=f"Basketball prediction error: {str(e)}")


