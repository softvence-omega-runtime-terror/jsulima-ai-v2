from fastapi import APIRouter, Depends, HTTPException
from app.services.loader import DataLoader
from fastapi.responses import PlainTextResponse

router = APIRouter()

data_loader = DataLoader()
player_service = data_loader.get_player_service()


@router.get("/median-strikes")
async def get_median_strikes(local_id: int, away_id: int):
    local_stats = player_service.get_median_strikes(local_id)
    away_stats = player_service.get_median_strikes(away_id)
    return {
        "status": "success",
        "data": {
            "local_fighter": local_stats,   
            "away_fighter": away_stats}
        }