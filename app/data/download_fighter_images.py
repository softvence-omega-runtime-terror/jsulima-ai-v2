import os
import base64
import time
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Import settings from config.py
from app.core.config import settings

BASE_DIR = "../images/"
os.makedirs(BASE_DIR, exist_ok=True)

FIGHTERS_URL = f"{settings.GOALSERVE_BASE_URL}/{settings.GOALSERVE_API_KEY}/mma/fighters"
PROFILE_URL = f"{settings.GOALSERVE_BASE_URL}/{settings.GOALSERVE_API_KEY}/mma/fighter?profile={{fighter_id}}"


def get_fighter_ids():
    print("[+] Fetching fighter list...")
    response = requests.get(FIGHTERS_URL)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    ids = [f.attrib["id"] for f in root.findall(".//fighter")]

    print(f"[+] Found {len(ids)} fighter IDs.")
    return ids


def download_image(fighter_id):
    img_path = os.path.join(BASE_DIR, f"{fighter_id}.png")

    if os.path.exists(img_path):
        return

    url = PROFILE_URL.format(fighter_id=fighter_id)
    response = requests.get(url)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    image_node = root.find(".//image")

    if image_node is None or not image_node.text:
        return

    try:
        img_data = base64.b64decode(image_node.text)
        with open(img_path, "wb") as f:
            f.write(img_data)
    except Exception as e:
        print(f"[ERROR] Failed image for {fighter_id}: {e}")

    time.sleep(settings.API_REQUEST_DELAY)


def main():
    fighter_ids = get_fighter_ids()
    print("[+] Starting image downloads...")

    for fighter_id in tqdm(fighter_ids, desc="Downloading", ncols=80):
        download_image(fighter_id)

    print("[+] Done! Images saved to ../images/")


if __name__ == "__main__":
    main()
