import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(r"C:\Users\immad\Downloads\DiB")
OUTPUT_FILE = DATA_DIR / "all_sensor_analysis.csv"

# ============================================================
# SENSOR CHANNELS
# Based on the dataset channel mapping we established
# ============================================================

STRAIN_COLS = [f"ch_{i}" for i in range(1, 17)]
ACCEL_COLS = [f"ch_{i}" for i in range(18, 23)]

OTHER_COLS = [
    "ch_25",   # inclinometer
    "ch_28",   # wind direction
    "ch_29",   # wind speed
    "ch_30",   # air temperature
]

SENSOR_COLS = STRAIN_COLS + ACCEL_COLS + OTHER_COLS


# ============================================================
# FIND RAW CSV FILES
# ============================================================

csv_files = sorted(
    p for p in DATA_DIR.glob("*.csv")
    if p.name not in [
    "dataset_summary.csv",
    "recording_analysis.csv",
    "recording_labels.csv",
    "all_sensor_analysis.csv",
    "fileid_analysis.csv"
    ]
)

print(f"Found {len(csv_files)} raw CSV files.\n")


# ============================================================
# ANALYZE EACH FILE
# ============================================================

results = []

for file_path in csv_files:

    print(f"Analyzing: {file_path.name}")

    df = pd.read_csv(file_path)

    row = {
        "file": file_path.name,
        "rows": len(df),
        "columns": len(df.columns)
    }

    # --------------------------------------------------------
    # Timestamp information
    # --------------------------------------------------------

    if "ts" in df.columns:

        ts = pd.to_datetime(df["ts"], format="mixed")

        row["start_time"] = ts.min()
        row["end_time"] = ts.max()

        diffs = ts.diff().dt.total_seconds().dropna()

        if len(diffs) > 0:
            row["median_sampling_interval"] = diffs.median()
            row["max_timestamp_gap"] = diffs.max()
            row["timestamp_gaps_gt_0.01s"] = int(
                (diffs > 0.01).sum()
            )

    # --------------------------------------------------------
    # Sensor analysis
    # --------------------------------------------------------

    for col in SENSOR_COLS:

        if col not in df.columns:
            row[f"{col}_missing_column"] = True
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        row[f"{col}_missing"] = int(values.isna().sum())

        if values.notna().sum() == 0:
            continue

        row[f"{col}_min"] = values.min()
        row[f"{col}_max"] = values.max()
        row[f"{col}_mean"] = values.mean()
        row[f"{col}_std"] = values.std()

        # Number of infinite values
        row[f"{col}_inf"] = int(
            np.isinf(values).sum()
        )

    results.append(row)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)

print("\n==============================================")
print("ANALYSIS COMPLETE")
print("==============================================")
print(f"Files analyzed : {len(result_df)}")
print(f"Output saved   : {OUTPUT_FILE}")
print("==============================================")