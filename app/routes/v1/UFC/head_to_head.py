from fastapi import APIRouter, Query
from app.services.UFC.match_history import get_head_to_head

router = APIRouter()

@router.get("/")
def get_h2h_stats(
    local_team_id: str = Query(..., description="ID of the local/first fighter"),
    away_team_id: str = Query(..., description="ID of the away/second fighter")
):
    """
    Get head-to-head historical stats between two fighters.
    Stats (wins, losses, draws) are relative to local_team_id.
    """
    result = get_head_to_head(local_team_id, away_team_id)
    return result
