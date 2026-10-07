import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(r"C:\bridge_monitor\data\processed")
SPLIT_FILE = Path(r"C:\bridge_monitor\data\splits\recording_split.csv")

splits = pd.read_csv(SPLIT_FILE)

train_files = splits[
    splits["split"] == "train"
]["filename"].tolist()

print(f"Training recordings: {len(train_files)}")

print("\n" + "=" * 80)
print("NaN CHECK IN TRAINING DATA")
print("=" * 80)

total_nan = {
    "ch_11": 0,
    "ch_22": 0
}

for filename in train_files:

    path = PROCESSED_DIR / filename

    df = pd.read_csv(path)

    for col in ["ch_11", "ch_22"]:

        if col not in df.columns:
            continue

        nan_mask = df[col].isna()
        count = int(nan_mask.sum())

        if count == 0:
            continue

        total_nan[col] += count

        print(f"\nFile: {filename}")
        print(f"Sensor: {col}")
        print(f"NaN count: {count}")

        # Show timestamps where NaNs occur
        cols = ["ts", "fileid", col]

        available = [
            c for c in cols
            if c in df.columns
        ]

        print(
            df.loc[
                nan_mask,
                available
            ].to_string(index=False)
        )


print("\n" + "=" * 80)
print("TOTAL TRAINING NaNs")
print("=" * 80)

for col, count in total_nan.items():
    print(f"{col}: {count}")

print("\n" + "=" * 80)