import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import requests
from PIL import Image
from io import BytesIO
from utils import get_local_file_path, get_species
from pathlib import Path
from urllib.parse import urlparse
import re
from requests.adapters import HTTPAdapter
from datetime import datetime
import cuid

MAX_WORKERS = 15

session = requests.Session()
# Increase connection pool for multithreading
adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount("http://", adapter)
session.mount("https://", adapter)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# Set True to avoid downloading files that already exist in the save folder
# However, set False if nothing is saved because some images may share a name but ave different content,
# such as preview.jpg
# As such, for best results, delete all existing GBIF files and set this to False
SKIP_DUPLICATE_FILENAMES = True


def is_valid_image(content: bytes) -> bool:
    try:
        img = Image.open(BytesIO(content))
        img.verify()
        return True
    except Exception:
        return False
    

def safe_filename(url: str):
    url = url.strip()  # remove newline

    parsed = urlparse(url)

    name = os.path.basename(parsed.path)

    if not name:  # when path is empty (query-only URLs)
        name = parsed.query

    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)

    return name + ".jpg"


def download_one(item: Dict[str, str]):
    # Extract species and image URL from item
    species = item["species"]
    url = item["image"].strip() # Remove newline if present, was causing issues before

    # Create folder for images if needed
    folder = get_local_file_path(species)
    os.makedirs(folder, exist_ok=True)

    filename = safe_filename(url)
    filepath = os.path.join(folder, filename)

    if os.path.exists(filepath):
        # File with same name already downloaded
        if SKIP_DUPLICATE_FILENAMES:
            # Skip already downloaded files
            return f"Already exists: {filepath}"
        else:
            # Add a CUID to make filepath unique
            parts = filepath.split('.')
            parts[-2] += f"_{cuid.cuid()}"
            filepath = ".".join(parts)

    try:
        for attempt in range(3):
            try:
                resp = session.get(url, headers=headers, timeout=60, allow_redirects=True)
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise

        if resp.status_code != 200:
            return f"Failed with code {resp.status_code}: {url}"

        # Skip the Content-Type check — USGS S3 sometimes returns
        # 'binary/octet-stream' for valid JPEGs. PIL's is_valid_image() below
        # is the source of truth for whether the bytes are actually an image.
        if not is_valid_image(resp.content):
            ct = resp.headers.get("Content-Type", "?")
            return f"Invalid image (content-type {ct}): {url}"

        with open(filepath, "wb") as f:
            f.write(resp.content)

        return f"Saved: {filepath}"

    except Exception as e:
        return f"Error: {url} {e}"


def download_images(items: List[Dict[str, str]]):

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, item) for item in items]

        log_folder = Path('download_logs')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = log_folder / f"log_{timestamp}.txt"

        os.makedirs(log_folder, exist_ok=True)

        with open(log_path, 'w') as log:
            log.write(f'Items: {len(items)}\n')

            for future in as_completed(futures):
                print(future.result())
                log.write(f'{future.result()}\n')

            log.write('\n\n\n')


def get_items(species_name: str) -> List[Dict[str, str]]:
    url_file_path = Path('species_images') / f'{species_name}.txt'

    if not url_file_path.exists():
        print(url_file_path, 'does not exist')
        return []

    with open(url_file_path) as f:
        urls = f.readlines()

    return [{'species': species_name, 'image': url} for url in urls]


if __name__ == '__main__':
    species = get_species()

    items = []
    for sp in species:
        items.extend(get_items(sp))

    download_images(items)