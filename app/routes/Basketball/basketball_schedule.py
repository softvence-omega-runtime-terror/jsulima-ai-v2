"""
Basketball Schedule Router Module
FastAPI endpoint for fetching upcoming NBA matches
"""
from fastapi import APIRouter, Query, HTTPException
import requests
from datetime import datetime
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

router = APIRouter()

# GoalServe API URL
SCHEDULE_URL = "https://www.goalserve.com/getfeed/48cbeb0a39014dc2d6db08dd947404e4/bsktbl/nba-shedule"


def parse_schedule_xml(xml_content: str) -> List[Dict[str, Any]]:
    """
    Parse XML content and extract upcoming matches
    
    Args:
        xml_content: Raw XML string from GoalServe API
        
    Returns:
        List of match dictionaries
    """
    try:
        root = ET.fromstring(xml_content)
        matches = []
        
        # Find all matches elements
        for matches_group in root.findall('.//matches'):
            season_type = matches_group.get('seasonType', 'Regular Season')
            
            # Iterate through each match
            for match in matches_group.findall('match'):
                status = match.get('status', '')
                
                # Only include upcoming matches (Not Started)
                if status == 'Not Started':
                    hometeam_elem = match.find('hometeam')
                    awayteam_elem = match.find('awayteam')
                    
                    if hometeam_elem is None or awayteam_elem is None:
                        continue
                    
                    match_data = {
                        "match_id": match.get('id', ''),
                        "date": match.get('date', ''),
                        "datetime_utc": match.get('datetime_utc', ''),
                        "formatted_date": match.get('formatted_date', ''),
                        "time": match.get('time', ''),
                        "type": season_type,
                        "venue_id": match.get('venue_id', ''),
                        "venue_name": match.get('venue_name', ''),
                        "hometeam": {
                            "hometeam_id": hometeam_elem.get('id', ''),
                            "hometeam_name": hometeam_elem.get('name', '')
                        },
                        "awayteam": {
                            "awayteam_id": awayteam_elem.get('id', ''),
                            "awayteam_name": awayteam_elem.get('name', '')
                        }
                    }
                    
                    matches.append(match_data)
        
        return matches
    
    except ET.ParseError as e:
        raise ValueError(f"XML parsing error: {str(e)}")


@router.get("/")
def get_upcoming_schedule(
    limit: int | None = Query(default=20, description="Number of matches to return")
):
    """
    Fetch only UPCOMING NBA matches from GoalServe.
    
    Query Parameters:
    - limit: Number of matches to return (default: 20)
    
    Returns:
    - status: success/error
    - total_upcoming: Total number of upcoming matches
    - matches: List of upcoming match details
    
    Example: /basketball_schedule?limit=10
    """
    
    try:
        response = requests.get(SCHEDULE_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch schedule: {str(e)}"
        )
    
    try:
        upcoming_matches = parse_schedule_xml(response.text)
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse schedule: {str(e)}"
        )
    
    # Sort by closest match first
    now = datetime.utcnow()
    
    def get_match_datetime(match: Dict) -> datetime:
        """Helper to parse match datetime"""
        try:
            # Parse datetime_utc format: "DD.MM.YYYY HH:MM"
            dt_str = match.get('datetime_utc', '')
            if dt_str:
                return datetime.strptime(dt_str, "%d.%m.%Y %H:%M")
        except:
            pass
        
        # Fallback: return far future date so unparseable matches go to end
        return datetime(2099, 12, 31)
    
    # Filter only future matches and sort
    upcoming_matches = [
        m for m in upcoming_matches 
        if get_match_datetime(m) > now
    ]
    
    upcoming_matches.sort(key=get_match_datetime)
    
    # Apply limit
    if limit is not None and limit > 0:
        upcoming_matches = upcoming_matches[:limit]
    
    return {
        "status": "success",
        "total_upcoming": len(upcoming_matches),
        "matches": upcoming_matches
    }


