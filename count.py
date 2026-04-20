from utils import *

# Copy bee_counts.csv and add Big Bee counts
df = pd.read_csv("bee_counts.csv")

# NOTE: Update df.columns[N] below to point at the column in bee_counts.csv
# where you want the Big Bee counts written. Column 8 was the Ecdysis column
# in the original script — change this to a new/dedicated Big Bee column.
BIGBEE_COL_INDEX = 8   # <-- change me if needed
TOTAL_COL_INDEX = 2

root = Path('BigBee')
for sp in get_subfolders(root):
    count = count_files_in_dir_tree(root / sp)

    # Update the Big Bee and total file counts for this species' row
    mask = df.iloc[:, 1] == sp
    df.loc[mask, df.columns[BIGBEE_COL_INDEX]] = count
    df.loc[mask, df.columns[TOTAL_COL_INDEX]] += count

df.to_csv("bee_counts_downloaded.csv", index=False)

print("\nCSV updated.")