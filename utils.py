import pandas as pd
from rapidfuzz import fuzz
import re
from typing import List, Dict
import os
import requests
from pathlib import Path


def get_folder_contents(folder_path):
    return [entry.name for entry in os.scandir(folder_path)]

def get_files_in_folder(folder_path):
    return [entry.name for entry in os.scandir(folder_path) if entry.is_file()]

def get_subfolders(folder_path):
    return [entry.name for entry in os.scandir(folder_path) if entry.is_dir()]


def count_files_in_dir_tree(path: Path) -> int:
    '''Counts all files in a directory, including all files in descendant directories'''
    file_count = len(get_files_in_folder(path))
    subfolders = get_subfolders(path)
    
    for sub in subfolders:
        file_count += count_files_in_dir_tree(path / sub)

    return file_count


def get_species(filename='Genus_Species_List.csv', col_name="Species"):
    df = df = pd.read_csv(
        filename,
        usecols=[col_name]
    )

    species = df.Species.to_list()

    return species


# Source folder name. Override via env var DOWNLOAD_SOURCE so a single utils.py
# can serve multiple sources (GBIF, USGS, etc.) without code duplication.
DOWNLOAD_SOURCE = os.environ.get("DOWNLOAD_SOURCE", "USGS")


def get_local_file_path(species_name: str) -> str:
    return f"{DOWNLOAD_SOURCE}/{species_name}"


def remove_psithyrus(s):
    return re.sub(r"\s*(\[Psithyrus\]|\(Psithyrus\))\s*", " ", s, flags=re.IGNORECASE)


def species_close(a, b, fuzz_threshold=85):
    # Remove psithyrus as it is sometimes present and doesn't change result
    a = remove_psithyrus(a)
    b = remove_psithyrus(b)

    # Ensure strings are same length to remove suffixes
    a = a[:len(b)]
    b = b[:len(a)]

    return fuzz.ratio(a, b) >= fuzz_threshold


def download_images_old(items: List[Dict[str, str]]) -> None:
    # items a list of dicts mapping 'species' and 'image' to strings

    saved = []

    for item in items:
        species_name = item["species"]

        img_url = item["image"]

        # Make folder for genus
        folder = get_local_file_path(species_name)
        os.makedirs(folder, exist_ok=True)

        # Create filename safely, keeping the original image name
        filename = f"{img_url.split('/')[-1]}"
        filepath = os.path.join(folder, filename)

        # Download and save image
        response = requests.get(img_url)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Saved {filepath}")
            saved.append(species_name)
        else:
            print(f"Failed to download {img_url}")