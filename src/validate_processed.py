import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(r"C:\bridge_monitor\data\processed")

files = sorted(PROCESSED_DIR.glob("*.csv"))

print(f"Found {len(files)} processed recordings.\n")

all_results = []

for file_path in files:

    df = pd.read_csv(file_path)

    # Parse timestamp again
    df["ts"] = pd.to_datetime(df["ts"], format="mixed")

    result = {
        "file": file_path.name,
        "rows": len(df),
        "segments": df["segment_id"].nunique(),
        "start": df["ts"].min(),
        "end": df["ts"].max(),
    }

    # Count NaNs introduced/remaining in important sensors
    for col in ["ch_11", "ch_12", "ch_22"]:

        if col in df.columns:
            result[f"{col}_NaN"] = int(df[col].isna().sum())

    # Check whether huge values still exist
    for col in ["ch_11", "ch_12", "ch_22"]:

        if col in df.columns:

            huge = np.abs(df[col]) >= 1e14

            result[f"{col}_huge"] = int(huge.sum())

    # Timestamp gaps
    if "timestamp_gap" in df.columns:
        result["timestamp_gaps"] = int(
            df["timestamp_gap"].sum()
        )

    all_results.append(result)


result_df = pd.DataFrame(all_results)

print("=" * 80)
print("PROCESSED DATA VALIDATION")
print("=" * 80)

print("\nNaN counts:")
for col in ["ch_11_NaN", "ch_12_NaN", "ch_22_NaN"]:
    print(
        f"{col}: "
        f"{result_df[col].sum():,}"
    )

print("\nRemaining huge values:")
for col in ["ch_11_huge", "ch_12_huge", "ch_22_huge"]:
    print(
        f"{col}: "
        f"{result_df[col].sum():,}"
    )

print("\nFiles containing timestamp gaps:")

gap_files = result_df[
    result_df["timestamp_gaps"] > 0
]

if len(gap_files) == 0:
    print("None")
else:
    print(
        gap_files[
            ["file", "timestamp_gaps", "segments"]
        ].to_string(index=False)
    )

print("\nRecording count:")
print(len(result_df))

print("\nTotal rows:")
print(f"{result_df['rows'].sum():,}")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)