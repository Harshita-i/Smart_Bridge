import pandas as pd
from pathlib import Path
import numpy as np

# ============================================================
# PATH
# ============================================================

RAW_DIR = Path(r"C:\Users\immad\Downloads\DiB")

# ============================================================
# SENSOR CHANNELS
# ============================================================

STRAIN_CHANNELS = ["ch_9", "ch_10", "ch_11"]
ACCEL_CHANNEL = "ch_22"
TEMP_CHANNEL = "ch_30"

# ============================================================
# OUTPUT
# ============================================================

results = []

csv_files = sorted(
    [
        f for f in RAW_DIR.glob("*.csv")
        if f.name != "dataset_summary.csv"
    ]
)

print("=" * 90)
print("ANALYZING ALL VÄNERSBORG RECORDINGS")
print("=" * 90)

print(f"\nNumber of CSV recordings found: {len(csv_files)}")

for i, path in enumerate(csv_files, start=1):

    print(f"\n[{i}/{len(csv_files)}] Reading {path.name}")

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Convert sensor columns to numeric
    # --------------------------------------------------------

    for col in STRAIN_CHANNELS + [ACCEL_CHANNEL, TEMP_CHANNEL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    start_time = df["ts"].iloc[0]
    end_time = df["ts"].iloc[-1]

    duration_seconds = (
        pd.to_datetime(end_time) -
        pd.to_datetime(start_time)
    ).total_seconds()

    file_ids = df["fileid"].unique().tolist()

    # --------------------------------------------------------
    # Strain statistics
    # --------------------------------------------------------

    sg9_mean = df["ch_9"].mean()
    sg9_std = df["ch_9"].std()

    sg10_mean = df["ch_10"].mean()
    sg10_std = df["ch_10"].std()

    sg11_mean = df["ch_11"].mean()
    sg11_std = df["ch_11"].std()

    # --------------------------------------------------------
    # A5 acceleration
    # --------------------------------------------------------

    a5_valid = df["ch_22"].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    # Raw maximum
    a5_max = a5_valid.max()

    # RMS using all finite values
    a5_rms = np.sqrt(np.mean(a5_valid ** 2))

    # Number of very large acceleration values
    extreme_count = (a5_valid.abs() > 100).sum()

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    temp_mean = df["ch_30"].mean()

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({
        "filename": path.name,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "file_ids": ",".join(map(str, file_ids)),

        "sg9_mean": sg9_mean,
        "sg9_std": sg9_std,

        "sg10_mean": sg10_mean,
        "sg10_std": sg10_std,

        "sg11_mean": sg11_mean,
        "sg11_std": sg11_std,

        "a5_max": a5_max,
        "a5_rms": a5_rms,
        "a5_extreme_count": extreme_count,

        "temperature_mean": temp_mean
    })

# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(results)

summary["start_time"] = pd.to_datetime(summary["start_time"])

summary = summary.sort_values("start_time")

# ============================================================
# SAVE
# ============================================================

output_path = RAW_DIR / "recording_analysis.csv"

summary.to_csv(output_path, index=False)

# ============================================================
# PRINT IMPORTANT INFORMATION
# ============================================================

print("\n")
print("=" * 90)
print("RECORDING ANALYSIS COMPLETE")
print("=" * 90)

print(f"\nSaved to:")
print(output_path)

print("\n\nA5 EXTREME RECORDINGS:")
print(
    summary[
        summary["a5_extreme_count"] > 0
    ][
        [
            "filename",
            "start_time",
            "file_ids",
            "a5_max",
            "a5_rms",
            "a5_extreme_count"
        ]
    ].to_string(index=False)
)

print("\n\nALL RECORDINGS:")
print(
    summary[
        [
            "filename",
            "start_time",
            "duration_seconds",
            "sg10_mean",
            "sg10_std",
            "a5_rms",
            "a5_extreme_count",
            "temperature_mean"
        ]
    ].to_string(index=False)
)

print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)