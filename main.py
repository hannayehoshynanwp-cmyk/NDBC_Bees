"""
main.py — orchestrate the USGS Bee Lab image scrape.

Mirrors the structure of the GBIF main.py, with two changes:
  - swaps in get_bee_images_async from usgsUtils
  - writes counts into the NBDC column on bee_counts.csv (NBDC = USGS Bee Lab
    in this naming scheme, per the wider NBDC_Public workflow)

Writes incrementally so a crash partway through doesn't lose progress:
  - species_images/<Species>.txt   — written immediately after each species
  - output.txt                     — appended after each species
  - BP-26_bee_counts_USGS.csv      — written at the end
"""

import asyncio
import os
from pathlib import Path

import pandas as pd

from usgsUtils import get_bee_images_async
from utils import get_species


# Column on bee_counts.csv where USGS counts go. Reusing NBDC since this
# task is part of the NBDC workflow.
USGS_COLUMN = "NBDC"


async def main():
    species = get_species()

    out_dir = Path("species_images")
    os.makedirs(out_dir, exist_ok=True)

    results = []          # list of {species, image_url}
    species_counts = {}   # species -> int

    # Truncate output.txt at the start of a fresh run, then append per species.
    with open("output.txt", "w") as f:
        f.write("")

    for i, sp in enumerate(species, 1):
        print(f"\n[{i}/{len(species)}] {sp}")
        species_results = await get_bee_images_async(sp)
        results.extend(species_results)
        species_counts[sp] = len(species_results)

        # --- Write this species' per-species file immediately
        sp_path = out_dir / f"{sp}.txt"
        with open(sp_path, "w") as f:
            for record in species_results:
                f.write(f"{record['image_url']}\n")
        print(f"  wrote {sp_path} ({len(species_results)} urls)")

        # --- Append to flat output.txt immediately
        with open("output.txt", "a") as f:
            for record in species_results:
                f.write(f"Species: {record['species']}\nImage URL: {record['image_url']}\n\n")

    print(f"\n{len(results)} images found across all species")

    # --- Update bee_counts.csv with USGS counts in the NBDC column (final step)
    df = pd.read_csv("bee_counts.csv")

    if USGS_COLUMN not in df.columns:
        raise SystemExit(
            f"Expected column '{USGS_COLUMN}' in bee_counts.csv but it's missing. "
            f"Available columns: {list(df.columns)}"
        )

    for sp, count in species_counts.items():
        mask = df.iloc[:, 1] == sp           # column 1 is "Species"
        df.loc[mask, USGS_COLUMN] = count
        df.loc[mask, df.columns[2]] += count  # NofFiles total

    df.to_csv("BP-26_bee_counts_USGS.csv", index=False)
    print("\nBP-26_bee_counts_USGS.csv updated.")


if __name__ == "__main__":
    asyncio.run(main())