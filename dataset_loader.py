# dataset_loader.py
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR=Path("./")
OUTPUT_ALL=DATA_DIR / "all_data.csv"
OUTPUT_BALANCED=DATA_DIR / "all_data_balanced.csv"

# Collect all _classes.csv files
csv_files=list(DATA_DIR.rglob("_classes.csv"))

if not csv_files:
    raise FileNotFoundError("No _classes.csv files found under ./data directory!")

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"   - {f}")

# Merge them
dfs=[]
for f in csv_files:
    df=pd.read_csv(f)
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    dfs.append(df)

all_data=pd.concat(dfs, ignore_index=True)
all_data.to_csv(OUTPUT_ALL, index=False)
print(f"Saved merged dataset → {OUTPUT_ALL}")

# Balance dataset (equal empty vs occupied)
empty_df=all_data[all_data["space-empty"]==1]
occupied_df=all_data[all_data["space-occupied"]==1]

min_len=min(len(empty_df), len(occupied_df))
balanced=pd.concat([
    empty_df.sample(min_len, random_state=42),
    occupied_df.sample(min_len, random_state=42)
], ignore_index=True).sample(frac=1, random_state=42)  # shuffle

balanced.to_csv(OUTPUT_BALANCED, index=False)
print(f"Saved balanced dataset → {OUTPUT_BALANCED}")

# Stats
print("\n Dataset Stats:")
print(f"Total images: {len(all_data)}")
print(f"  Empty: {len(empty_df)}")
print(f"  Occupied: {len(occupied_df)}")
print(f"Balanced dataset size: {len(balanced)}")
