import xml.etree.ElementTree as ET
from app.core.config import settings
SCHEDULE_URL = f"https://www.goalserve.com/getfeed/{settings.goalserve_api_key}/mma/schedule"

def parse_schedule_xml(xml_text):
    """Parse GoalServe UFC schedule XML into structured JSON."""
    root = ET.fromstring(xml_text)
    all_categories = []

    for category in root.findall("category"):
        cat_name = category.get("name")
        cat_date = category.get("date")
        cat_id   = category.get("id")

        matches_list = []

        for match in category.findall("match"):
            match_info = {
                "match_id": match.get("id"),
                "date": match.get("date"),
                "time": match.get("time"),
                #"status": match.get("status"),
                "type": match.get("type"),
                "is_main": match.get("ismain")
            }

            # Local fighter
            local = match.find("localteam")
            match_info["localteam"] = {
                "id": local.get("id"),
                "name": local.get("name")
            }

            # Away fighter
            away = match.find("awayteam")
            match_info["awayteam"] = {
                "id": away.get("id"),
                "name": away.get("name")
            }

            matches_list.append(match_info)

        all_categories.append({
            "category_id": cat_id,
            "category_name": cat_name,
            "event_date": cat_date,
            "matches": matches_list
        })

    return all_categories
