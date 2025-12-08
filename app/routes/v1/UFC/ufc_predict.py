from fastapi import APIRouter, Depends, HTTPException
from app.services.loader import DataLoader
from fastapi.responses import PlainTextResponse

router = APIRouter()

data_loader = DataLoader()
player_service = data_loader.get_player_service()


@router.get("/")
async def get_median_strikes(local_id: int = 88040, away_id: int = 97830):
    local_stats = player_service.get_median_strikes(local_id)
    away_stats = player_service.get_median_strikes(away_id)
    return {
        "status": "success",
        "data": {
            "local_fighter": local_stats,   
            "away_fighter": away_stats}
        }