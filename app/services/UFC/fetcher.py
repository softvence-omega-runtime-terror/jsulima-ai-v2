import requests
from app.services.UFC.parser import parse_xml
from app.core.config import settings

BASE_URL = f"https://www.goalserve.com/getfeed/{settings.GOALSERVE_API_KEY}/mma/live?date="
START_YEAR = 2010
END_YEAR = 2025

def fetch_for_date(date_str, all_matches):
    url = BASE_URL + date_str
    print("Fetching:", url)

    try:
        response = requests.get(url, timeout=60)

        if response.status_code != 200:
            print(" ❌ Bad status:", response.status_code)
            return

        if "<scores" not in response.text:
            print(f" ⚠ No data on {date_str}")
            return

        parse_xml(response.text, date_str, all_matches)

    except Exception as e:
        print(" ❌ Error:", e)


import pandas as pd
from datetime import datetime, timedelta

import os

# ensure folder exists
os.makedirs("data", exist_ok=True)

def run_scraper():

    for year in range(START_YEAR, END_YEAR + 1):

        print("\n==============================")
        print("  YEAR:", year)
        print("==============================")

        all_matches = []

        start_date = datetime(year, 1, 1)
        end_date   = datetime(year, 12, 31)

        current = start_date

        while current <= end_date:
            date_str = current.strftime("%d.%m.%Y")
            fetch_for_date(date_str, all_matches)
            current += timedelta(days=1)

        df = pd.DataFrame(all_matches)

        filename = f"data/mma_matches_{year}.csv"
        df.to_csv(filename, index=False)

        print(f"🎉 Saved {len(df)} records → {filename}")

    print("\n🎯 ALL DONE FOR 2010–2025")