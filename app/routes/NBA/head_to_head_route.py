from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.NBA.head_to_head import get_head_to_head

router = APIRouter()

class HeadToHeadRequest(BaseModel):
    hometeam: str
    awayteam: str

@router.post("/")
async def get_nba_head_to_head(request: HeadToHeadRequest):
    """
    Get head-to-head statistics between two NBA teams.
    """
    result = get_head_to_head(request.hometeam, request.awayteam)
    
    if "error" in result:
        # Depending on the error nature, might want 404 or 500. 
        # For "data not found", 500 or 404 is appropriate.
        # For now, if it returns an error dictionary, we return it directly or raise HTTPException.
        # The service returns {"error": ...} on failure.
        if "NBA games data not found" in result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
        # For other errors, might simply obtain the result
    
    return result
