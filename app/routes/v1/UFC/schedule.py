from fastapi import APIRouter, Query
import requests
from datetime import datetime
from app.services.UFC.schedule_pars import SCHEDULE_URL
from app.services.UFC.schedule_pars import parse_schedule_xml

router = APIRouter(prefix="/schedule", tags=["Schedule"])


@router.get("/")
def get_upcoming_schedule(limit: int| None = Query(default=None, description="Number of matches to return")):
    """
    Fetch only UPCOMING UFC matches from GoalServe.
    Limit results using: /schedule?limit=10
    """

    try:
        response = requests.get(SCHEDULE_URL, timeout=50)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to fetch schedule: {str(e)}"}

    all_events = parse_schedule_xml(response.text)

    upcoming_matches = []
    now = datetime.utcnow()

    for event in all_events:
        for m in event["matches"]:

            match_date = m["date"]
            match_time = m["time"]

            try:
                match_datetime = datetime.strptime(f"{match_date} {match_time}", "%d.%m.%Y %H:%M")
            except:
                # If time is missing or broken
                continue

            if match_datetime > now:
                m["event_name"] = event["category_name"]
                m["event_date"] = event["event_date"]
                upcoming_matches.append(m)

    # Sort by closest match
    upcoming_matches.sort(key=lambda x: datetime.strptime(f"{x['date']} {x['time']}", "%d.%m.%Y %H:%M"))

    # Apply limit
    if limit is not None:
        upcoming_matches = upcoming_matches[:limit]

    return {
        "status": "success",
        "total_upcoming": len(upcoming_matches),
        "matches": upcoming_matches
    }
