import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATH TO YOUR RAW DATA
# --------------------------------------------------
RAW_DIR = Path(r"C:\Users\immad\Downloads\DiB")

# Files around the reported fracture period
FILES_TO_CHECK = [
    "2023_03_09T13_03_18.csv",
    "2023_03_09T21_53_28.csv",
    "2023_03_09T23_45_25.csv",
    "2023_03_10T08_34_30.csv",
    "2023_03_10T17_37_28.csv",
    "2023_03_10T20_51_26.csv",
    "2023_03_11T16_36_25.csv",
    "2023_03_11T17_53_02.csv",
    "2023_03_11T18_34_51.csv",
]

# --------------------------------------------------
# CHANNELS WE WANT TO INSPECT
# --------------------------------------------------
# According to the dataset documentation:
# ch_10 -> strain sensor SG10
# ch_22 -> accelerometer A5
# ch_30 -> temperature

CHANNELS = ["ch_10", "ch_22", "ch_30"]


print("=" * 80)
print("VÄNERSBORG DATASET INSPECTION")
print("=" * 80)

for filename in FILES_TO_CHECK:

    path = RAW_DIR / filename

    print("\n" + "-" * 80)
    print(f"FILE: {filename}")
    print("-" * 80)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")

    # --------------------------------------------------
    # Timestamp information
    # --------------------------------------------------
    if "ts" in df.columns:
        print(f"Start time : {df['ts'].iloc[0]}")
        print(f"End time   : {df['ts'].iloc[-1]}")

    # --------------------------------------------------
    # File IDs
    # --------------------------------------------------
    if "fileid" in df.columns:
        print(f"File IDs   : {df['fileid'].unique().tolist()}")

    # --------------------------------------------------
    # Event column
    # --------------------------------------------------
    if "event" in df.columns:
        print(f"Events     : {df['event'].unique().tolist()}")

    # --------------------------------------------------
    # Sensor statistics
    # --------------------------------------------------
    print("\nSensor statistics:")

    for col in CHANNELS:

        if col not in df.columns:
            print(f"{col}: NOT FOUND")
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        print(
            f"{col:6s} | "
            f"min = {series.min():.6f} | "
            f"max = {series.max():.6f} | "
            f"mean = {series.mean():.6f} | "
            f"std = {series.std():.6f}"
        )

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

# --------------------------------------------------
# DETAILED CHECK OF THE SUSPECTED FRACTURE FILE
# --------------------------------------------------

filename = "2023_03_09T23_45_25.csv"
path = RAW_DIR / filename

print("\n" + "=" * 80)
print("DETAILED CHECK:", filename)
print("=" * 80)

df = pd.read_csv(path)

# Find the 20 rows with the largest absolute acceleration values
df["abs_ch22"] = df["ch_22"].abs()

top_rows = df.nlargest(20, "abs_ch22")

print("\nTop 20 extreme ch_22 values:")
print(
    top_rows[
        ["ts", "id", "fileid", "ch_10", "ch_22", "ch_30"]
    ].to_string(index=False)
)

# How many values are unusually large?
print("\nNumber of extreme values:")

print("abs(ch_22) > 1    :", (df["ch_22"].abs() > 1).sum())
print("abs(ch_22) > 10   :", (df["ch_22"].abs() > 10).sum())
print("abs(ch_22) > 100  :", (df["ch_22"].abs() > 100).sum())
print("abs(ch_22) > 1000 :", (df["ch_22"].abs() > 1000).sum())

# Check each fileid separately
print("\nch_22 statistics by fileid:")

print(
    df.groupby("fileid")["ch_22"]
    .agg(["count", "min", "max", "mean", "std"])
    .to_string()
)

# Remove temporary column
df.drop(columns=["abs_ch22"], inplace=True)