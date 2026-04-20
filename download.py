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


def try_download(url):
    return session.get(url, headers=headers, timeout=60, allow_redirects=True)


def download_one(item: Dict[str, str]):
    # Extract species and image URL from item
    species = item["species"]
    url = item["image"].strip() # Remove newline if present, was causing issues before

    # Ecdysis placeholder check
    if url == 'https://ecdysis.org/images/image-icon.svg':
        return 'Placeholder encountered'

    # Create folder for images if needed
    folder = get_local_file_path(species)
    os.makedirs(folder, exist_ok=True)

    filename = safe_filename(url)
    filepath = os.path.join(folder, filename)

    # Skip already downloaded files
    if os.path.exists(filepath):
        return f"Already exists: {filepath}"

    try:
        for attempt in range(3):
            try:
                resp = session.get(url, headers=headers, timeout=60, allow_redirects=True)
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise

        # If 404, try capitalized JPG, which accounts for many more images, oddly
        if resp.status_code == 404:
            if url.lower().endswith(".jpg"):
                alt_url = re.sub(r'\.jpg$', '.JPG', url)
            elif url.lower().endswith(".jpeg"):
                alt_url = re.sub(r'\.jpeg$', '.JPEG', url)
            else:
                alt_url = None

            if alt_url:
                resp = try_download(alt_url)
                if resp.status_code == 200:
                    url = alt_url
            

        if resp.status_code != 200:
            return f"Failed with code {resp.status_code}: {url}"

        if "image" not in resp.headers.get("Content-Type", ""):
            return f"Not an image: {url}"

        if not is_valid_image(resp.content):
            return f"Invalid image: {url}"

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


def remove_tn(url: str) -> str:
    parts = url.split('.')
    filename = parts[-2]

    new_name = filename.removesuffix('_tn')
    parts[-2] = new_name

    return '.'.join(parts)


def get_items(species_name: str) -> List[Dict[str, str]]:
    url_file_path = Path('image_urls') / f'{species_name}.txt'

    if not url_file_path.exists():
        print(url_file_path, 'does not exist')
        return []

    with open(url_file_path) as f:
        urls = f.readlines()


    # Remove _tn at the end of each filename
    urls = [remove_tn(url) for url in urls]


    return [{'species': species_name, 'image': url} for url in urls]


if __name__ == '__main__':
    species = get_species()

    items = []
    for sp in species:
        items.extend(get_items(sp))

    download_images(items)