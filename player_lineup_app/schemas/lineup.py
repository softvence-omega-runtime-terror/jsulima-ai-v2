from pydantic import BaseModel, Field
from typing import List

class TeamInfo(BaseModel):
    """Team information"""
    name: str = Field(..., description="Team name")
    totalscore: str = Field("", description="Team total score")

class MatchRequest(BaseModel):
    """
    Schema for match details and lineup prediction request
    """
    date: str = Field(..., description="Match date (e.g., 'Jan 21, 2026')")
    timezone: str = Field(..., description="Match timezone (e.g., 'EDT')")
    status: str = Field(..., description="Match status (e.g., 'Not Started')")
    time: str = Field(..., description="Match time (e.g., '10:00 PM')")
    venue_name: str = Field(..., description="Venue name")
    attendance: str = Field("", description="Attendance")
    id: str = Field(..., description="Match ID")
    hometeam: TeamInfo = Field(..., description="Home team information")
    awayteam: TeamInfo = Field(..., description="Away team information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "Jan 21, 2026",
                "timezone": "EDT",
                "status": "Not Started",
                "time": "10:00 PM",
                "venue_name": "Golden 1 Center, Sacramento, CA",
                "attendance": "",
                "id": "310921",
                "hometeam": {
                    "name": "Sacramento Kings",
                    "totalscore": ""
                },
                "awayteam": {
                    "name": "Toronto Raptors",
                    "totalscore": ""
                }
            }
        }

class LineupPredictionResponse(BaseModel):
    """
    Schema for lineup prediction response
    """
    match_id: str = Field(..., description="Match ID")
    date: str = Field(..., description="Match date")
    venue: str = Field(..., description="Venue name")
    hometeam: str = Field(..., description="Home team name")
    awayteam: str = Field(..., description="Away team name")
    home_starters: List[str] = Field(..., description="List of 5 predicted home team starters")
    away_starters: List[str] = Field(..., description="List of 5 predicted away team starters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "match_id": "310921",
                "date": "Jan 21, 2026",
                "venue": "Golden 1 Center, Sacramento, CA",
                "hometeam": "Sacramento Kings",
                "awayteam": "Toronto Raptors",
                "home_starters": ["De'Aaron Fox", "Domantas Sabonis", "Kevin Huerter", "Malik Monk", "Harrison Barnes"],
                "away_starters": ["Scottie Barnes", "RJ Barrett", "Pascal Siakam", "Gary Trent Jr.", "OG Anunoby"]
            }
        }

