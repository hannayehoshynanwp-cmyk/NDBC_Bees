"""
count.py — recount images already on disk under USGS/<species>/ and write
the counts into the NBDC column on a copy of bee_counts.csv. Run this after
download.py finishes to update counts based on what actually downloaded
(some URLs may have failed).
"""

from utils import *

USGS_COLUMN = "NBDC"
SOURCE_DIR = "USGS"


def main():
    df = pd.read_csv("bee_counts.csv")

    if USGS_COLUMN not in df.columns:
        raise SystemExit(
            f"Expected column '{USGS_COLUMN}' in bee_counts.csv but it's missing. "
            f"Available columns: {list(df.columns)}"
        )

    root = Path(SOURCE_DIR)
    if not root.exists():
        print(f"{SOURCE_DIR}/ folder doesn't exist — nothing to count.")
        return

    for sp in get_subfolders(root):
        count = count_files_in_dir_tree(root / sp)

        mask = df.iloc[:, 1] == sp
        df.loc[mask, USGS_COLUMN] = count
        df.loc[mask, df.columns[2]] += count  # NofFiles total

    df.to_csv("BP-26_bee_counts_USGS_downloaded.csv", index=False)
    print("\nBP-26_bee_counts_USGS_downloaded.csv updated.")


if __name__ == "__main__":
    main()

