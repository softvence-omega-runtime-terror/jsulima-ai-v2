from fastapi import APIRouter, Query
import requests
from datetime import datetime
from app.services.UFC.schedule_pars import SCHEDULE_URL
from app.services.UFC.schedule_pars import parse_schedule_xml
from app.services.predictor import get_predictor

router = APIRouter()


@router.get("/")
def get_upcoming_schedule(limit: int| None = Query(default=20, description="Number of matches to return")):
    """
    Fetch only UPCOMING UFC matches from GoalServe and add predictions.
    Limit results using: /schedule?limit=10
    """

    try:
        response = requests.get(SCHEDULE_URL, timeout=50)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to fetch schedule: {str(e)}"}

    all_events = parse_schedule_xml(response.text)
    predictor = get_predictor()

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
                
                # Initialize default values to ensure they are always in the response
                m["home_win_probability"] = 50.0
                m["away_win_probability"] = 50.0
                m["confidence"] = 50

                # Add real Predictions if possible
                try:
                    prediction = predictor.predict_match(
                        local_id=m["localteam"]["id"],
                        away_id=m["awayteam"]["id"],
                        date=m["date"],
                        weight_class=m.get("type", "Lightweight")
                    )
                    
                    if "prediction" in prediction:
                        res = prediction["prediction"]
                        m["home_win_probability"] = float(res.get("home_win_probability", 50.0))
                        m["away_win_probability"] = float(res.get("away_win_probability", 50.0))
                        m["confidence"] = int(res.get("confidence", 50))
                        
                except Exception as e:
                    print(f"Error predicting match {m.get('match_id')}: {e}")

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
