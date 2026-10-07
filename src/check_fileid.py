import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\immad\Downloads\DiB")

files = [
    "2023_02_20T03_10_53.csv",
    "2023_02_20T03_17_23.csv"
]

target_fileid = 62019

print("=" * 80)
print("CHECKING FILEID CONTINUITY")
print("=" * 80)

parts = []

for filename in files:

    path = RAW_DIR / filename

    df = pd.read_csv(path)

    part = df[df["fileid"] == target_fileid].copy()

    if len(part) > 0:

        part["ts"] = pd.to_datetime(part["ts"], format="mixed")

        parts.append(part)

        print("\nFILE:", filename)
        print("Rows:", len(part))
        print("Start:", part["ts"].iloc[0])
        print("End  :", part["ts"].iloc[-1])

# ----------------------------------------------------------
# Combine the two pieces
# ----------------------------------------------------------

combined = pd.concat(parts, ignore_index=True)

combined = combined.sort_values("ts")

print("\n" + "=" * 80)
print("COMBINED FILEID 62019")
print("=" * 80)

print("Total rows:", len(combined))

print("Start:", combined["ts"].iloc[0])
print("End  :", combined["ts"].iloc[-1])

# ----------------------------------------------------------
# Check timestamp differences
# ----------------------------------------------------------

time_diff = combined["ts"].diff().dt.total_seconds()

print("\nTimestamp difference statistics:")
print(time_diff.describe())

print("\nMost common timestamp differences:")
print(time_diff.value_counts().head(10))

# ----------------------------------------------------------
# Check largest gaps
# ----------------------------------------------------------

large_gaps = time_diff[time_diff > 0.01]

print("\nNumber of gaps greater than 0.01 seconds:")
print(len(large_gaps))

if len(large_gaps) > 0:

    print("\nLargest gaps:")

    print(
        time_diff.nlargest(10)
    )

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)