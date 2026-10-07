import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path(r"C:\Users\immad\Downloads\DiB")

OUTPUT_DIR = Path(r"C:\bridge_monitor\data\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SENSOR CHANNELS
# ============================================================

STRAIN_COLS = [f"ch_{i}" for i in range(1, 17)]

ACCEL_COLS = [f"ch_{i}" for i in range(18, 23)]

OTHER_COLS = [
    "ch_25",   # inclinometer
    "ch_28",   # wind direction
    "ch_29",   # wind speed
    "ch_30",   # temperature
]

SENSOR_COLS = STRAIN_COLS + ACCEL_COLS + OTHER_COLS


# ============================================================
# RAW RECORDINGS
# ============================================================

EXCLUDED_FILES = {
    "dataset_summary.csv",
    "recording_analysis.csv",
    "recording_labels.csv",
    "all_sensor_analysis.csv",
    "fileid_analysis.csv",
}


csv_files = sorted(
    p for p in RAW_DIR.glob("*.csv")
    if p.name not in EXCLUDED_FILES
)

print(f"Found {len(csv_files)} raw recordings.")


# ============================================================
# PROCESS EACH RECORDING
# ============================================================

for file_path in csv_files:

    print(f"\nProcessing: {file_path.name}")

    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["ts"] = pd.to_datetime(
        df["ts"],
        format="mixed"
    )

    df = df.sort_values("ts").reset_index(drop=True)


    # --------------------------------------------------------
    # Keep only required columns + metadata
    # --------------------------------------------------------

    metadata_cols = [
        "ts",
        "id",
        "fileid",
        "event"
    ]

    keep_cols = [
        col
        for col in metadata_cols + SENSOR_COLS
        if col in df.columns
    ]

    df = df[keep_cols].copy()


    # --------------------------------------------------------
    # Convert sensor columns to numeric
    # --------------------------------------------------------

    for col in SENSOR_COLS:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


    # ========================================================
    # INVALID / SENTINEL VALUES
    # ========================================================

    # --------------------------------------------------------
    # ch_12
    #
    # Observed sentinel:
    # -1,000,000
    #
    # Treat this as invalid/missing in the processed copy.
    # --------------------------------------------------------

    if "ch_12" in df.columns:

        invalid_ch12 = (
            df["ch_12"] == -1_000_000
        )

        df.loc[invalid_ch12, "ch_12"] = np.nan


    # --------------------------------------------------------
    # Very large sentinel-like value
    #
    # 8e15 was observed in ch_11, ch_12 and ch_22.
    #
    # We mark it invalid rather than allowing it to dominate
    # model scaling.
    # --------------------------------------------------------

    for col in ["ch_11", "ch_12", "ch_22"]:

        if col in df.columns:

            invalid_large = (
                df[col].abs() >= 1e14
            )

            df.loc[
                invalid_large,
                col
            ] = np.nan


    # --------------------------------------------------------
    # Other extreme value observed in ch_12
    #
    # The March 2 recording contains values around -1e4 to
    # -1.7e4. We do NOT automatically classify these as invalid
    # because we have not established a documented physical
    # range for this channel.
    # --------------------------------------------------------


    # ========================================================
    # INVALID-VALUE FLAGS
    # ========================================================

    for col in ["ch_11", "ch_12", "ch_22"]:

        if col in df.columns:

            # 1 = value was replaced by NaN
            # 0 = value was retained

            # We cannot reconstruct the original value after
            # replacement, so this flag is created before
            # replacement in the next version if needed.

            pass


    # ========================================================
    # TIMESTAMP GAPS
    # ========================================================

    df["time_diff_seconds"] = (
        df["ts"]
        .diff()
        .dt.total_seconds()
    )

    # Expected sampling interval ≈ 0.005 seconds.
    #
    # A gap greater than 0.01 seconds means we should not
    # create an ML sequence across that boundary.

    df["timestamp_gap"] = (
        df["time_diff_seconds"] > 0.01
    )


    # --------------------------------------------------------
    # Segment ID
    # --------------------------------------------------------

    df["segment_id"] = (
        df["timestamp_gap"]
        .fillna(False)
        .astype(int)
        .cumsum()
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_file = OUTPUT_DIR / file_path.name

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )

    print(
        f"Rows: {len(df):,} | "
        f"Segments: {df['segment_id'].nunique()}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"Processed files saved to:")
print(OUTPUT_DIR)