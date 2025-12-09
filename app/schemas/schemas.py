"""
API Schemas Module
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional


# Request Models
class FighterStats(BaseModel):
    """Fighter statistics for prediction input"""
    win_rate: float = Field(default=0.5, ge=0, le=1, description="Win rate (0-1)")
    ko_rate: float = Field(default=0.0, ge=0, le=1, description="KO win rate")
    sub_rate: float = Field(default=0.0, ge=0, le=1, description="Submission win rate")
    avg_strikes: float = Field(default=0.0, ge=0, description="Average strikes per fight")
    td_accuracy: float = Field(default=0.0, ge=0, le=1, description="Takedown accuracy")
    td_defense: float = Field(default=0.5, ge=0, le=1, description="Takedown defense")
    avg_knockdowns: float = Field(default=0.0, ge=0, description="Average knockdowns")
    experience: int = Field(default=0, ge=0, description="Number of fights")


class PredictionRequest(BaseModel):
    """Fight prediction request"""
    date: Optional[str] = Field(default=None, description="Fight date (dd.MM.yyyy)")
    localteam: int = Field(..., description="Local fighter ID")
    awayteam: int = Field(..., description="Away fighter ID")
    weight_class: str = Field(default="Lightweight", description="Weight class")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "date": "22.11.2025",
                    "localteam": 101420,
                    "awayteam": 100795,
                    "weight_class": "Featherweight"
                }
            ]
        }
    }



# Health Check
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    models_loaded: bool = False
    version: str = "1.0.0"
