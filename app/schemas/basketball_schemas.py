"""
Basketball Prediction Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional


class BasketballPredictionRequest(BaseModel):
    """Basketball game prediction request"""
    match_id: str = Field(..., description="Match identifier")
    date: str = Field(..., description="Game date (dd.MM.yyyy)")
    awayteam_id: str = Field(..., description="Away team ID")
    hometeam_id: str = Field(..., description="Home team ID")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "match_id": "311495",
                    "date": "12.04.2026",
                    "awayteam_id": "1225",
                    "hometeam_id": "1550"
                }
            ]
        }
    }


class TeamPrediction(BaseModel):
    """Team prediction details"""
    team_id: str
    team_name: str
    win_probability: str
    is_winner: bool


class BasketballPredictionResponse(BaseModel):
    """Basketball prediction response"""
    match: dict