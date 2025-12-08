"""
Pydantic Schemas for UFC Prediction API

This module defines the request and response models for the prediction endpoints.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Union


class FighterStats(BaseModel):
    """Fighter statistics for a single UFC match."""
    
    # Strike statistics - Local fighter
    local_strikes_total_head: float = Field(..., ge=0, description="Local fighter total head strikes")
    local_strikes_total_body: float = Field(..., ge=0, description="Local fighter total body strikes")
    local_strikes_total_leg: float = Field(..., ge=0, description="Local fighter total leg strikes")
    local_strikes_power_head: float = Field(..., ge=0, description="Local fighter power head strikes")
    local_strikes_power_body: float = Field(..., ge=0, description="Local fighter power body strikes")
    local_strikes_power_leg: float = Field(..., ge=0, description="Local fighter power leg strikes")
    
    # Grappling statistics - Local fighter
    local_takedowns_att: float = Field(..., ge=0, description="Local fighter takedown attempts")
    local_takedowns_landed: float = Field(..., ge=0, description="Local fighter successful takedowns")
    local_submissions: float = Field(..., ge=0, description="Local fighter submission attempts")
    local_knockdowns: float = Field(..., ge=0, description="Local fighter knockdowns")
    
    # Strike statistics - Away fighter
    away_strikes_total_head: float = Field(..., ge=0, description="Away fighter total head strikes")
    away_strikes_total_body: float = Field(..., ge=0, description="Away fighter total body strikes")
    away_strikes_total_leg: float = Field(..., ge=0, description="Away fighter total leg strikes")
    away_strikes_power_head: float = Field(..., ge=0, description="Away fighter power head strikes")
    away_strikes_power_body: float = Field(..., ge=0, description="Away fighter power body strikes")
    away_strikes_power_leg: float = Field(..., ge=0, description="Away fighter power leg strikes")
    
    # Grappling statistics - Away fighter
    away_takedowns_att: float = Field(..., ge=0, description="Away fighter takedown attempts")
    away_takedowns_landed: float = Field(..., ge=0, description="Away fighter successful takedowns")
    away_submissions: float = Field(..., ge=0, description="Away fighter submission attempts")
    away_knockdowns: float = Field(..., ge=0, description="Away fighter knockdowns")
    
    # Control time (Format: "MM:SS" or similar)
    local_control_time: str = Field(..., description="Local fighter control time (e.g. '5:00')")
    away_control_time: str = Field(..., description="Away fighter control time (e.g. '4:30')")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "local_strikes_total_head": 85,
                "local_strikes_total_body": 12,
                "local_strikes_total_leg": 8,
                "local_strikes_power_head": 35,
                "local_strikes_power_body": 8,
                "local_strikes_power_leg": 6,
                "local_takedowns_att": 3,
                "local_takedowns_landed": 2,
                "local_submissions": 0,
                "local_knockdowns": 1,
                "local_control_time": "2:00",
                "away_strikes_total_head": 70,
                "away_strikes_total_body": 10,
                "away_strikes_total_leg": 6,
                "away_strikes_power_head": 25,
                "away_strikes_power_body": 7,
                "away_strikes_power_leg": 5,
                "away_takedowns_att": 2,
                "away_takedowns_landed": 1,
                "away_submissions": 0,
                "away_knockdowns": 0,
                "away_control_time": "1:00"
            }
        }
    )


class PredictionRequest(BaseModel):
    """Request model for single fight prediction."""
    
    fighter_stats: FighterStats = Field(..., description="Fight statistics")


class BatchPredictionRequest(BaseModel):
    """Request model for batch fight predictions."""
    
    fights: List[FighterStats] = Field(..., min_items=1, description="List of fight statistics")


class PredictionResponse(BaseModel):
    """Response model for fight prediction."""
    
    prediction: int = Field(..., description="Predicted winner (1=Local, 0=Away)")
    winner: str = Field(..., description="Winner label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probability_local: float = Field(..., ge=0, le=1, description="Probability of local fighter winning")
    probability_away: float = Field(..., ge=0, le=1, description="Probability of away fighter winning")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[Union[PredictionResponse, Dict]] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of fights")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")


class FeatureImportance(BaseModel):
    """Feature importance information."""
    
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_type: str = Field(..., description="Type of ML model")
    num_features: int = Field(..., description="Number of features")
    accuracy: Optional[float] = Field(None, description="Model accuracy on test set")
    roc_auc: Optional[float] = Field(None, description="ROC-AUC score")
    feature_importance: Optional[List[FeatureImportance]] = Field(None, description="Top important features")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class PlayerPredictionRequest(BaseModel):
    """Request model for player ID-based prediction."""
    
    localteam: int = Field(default=88040, description="Local team/player ID", )
    awayteam: int = Field(default=97830, description="Away team/player ID")
    date: Optional[str] = Field(
        "14.12.2025",
        description="Date to filter historical data (DD-MM-YYYY format). If not provided, uses all historical data",
    )



class TeamInfo(BaseModel):
    """Team/player information in prediction response."""
    
    id: int = Field(..., description="Team/player ID")
    name: str = Field(..., description="Team/player name")
    win_probability: float = Field(..., ge=0, le=1, description="Probability of winning")
    is_winner: bool = Field(..., description="Whether this team is the predicted winner")


class PlayerPredictionResponse(BaseModel):
    """Response model for player ID-based prediction."""
    
    localteam: TeamInfo = Field(..., description="Local team information")
    awayteam: TeamInfo = Field(..., description="Away team information")
    