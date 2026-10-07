import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path(r"C:\bridge_monitor\data\processed")
SPLIT_FILE = Path(r"C:\bridge_monitor\data\splits\recording_split.csv")

MODEL_DIR = Path(r"C:\bridge_monitor\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SENSOR CHANNELS
# ============================================================

FEATURE_COLS = (
    [f"ch_{i}" for i in range(1, 12)] +
    [f"ch_{i}" for i in range(13, 17)] +
    [f"ch_{i}" for i in range(18, 23)] +
    ["ch_25", "ch_28", "ch_29", "ch_30"]
)

print(f"Number of model features: {len(FEATURE_COLS)}")
print(FEATURE_COLS)


# ============================================================
# LOAD SPLIT
# ============================================================

splits = pd.read_csv(SPLIT_FILE)

train_files = splits[
    splits["split"] == "train"
]["filename"].tolist()

print(f"\nTraining recordings: {len(train_files)}")


# ============================================================
# COLLECT HEALTHY TRAINING DATA
# ============================================================

training_parts = []

for filename in train_files:

    path = PROCESSED_DIR / filename

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["ts"] = pd.to_datetime(
        df["ts"],
        format="mixed"
    )

    # --------------------------------------------------------
    # Interpolate only isolated ch_11 missing values
    # --------------------------------------------------------

    if "ch_11" in df.columns:

        df["ch_11"] = (
            df["ch_11"]
            .interpolate(
                method="linear",
                limit=2,
                limit_direction="both"
            )
        )

    # --------------------------------------------------------
    # Select model features
    # --------------------------------------------------------

    part = df[FEATURE_COLS].copy()

    training_parts.append(part)


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

train_data = pd.concat(
    training_parts,
    ignore_index=True
)

print(f"\nTraining rows: {len(train_data):,}")


# ============================================================
# CHECK REMAINING NaNs
# ============================================================

print("\nRemaining NaNs:")

nan_counts = train_data.isna().sum()

print(
    nan_counts[
        nan_counts > 0
    ]
)


# ============================================================
# DROP ROWS WITH NaN
# ============================================================

rows_before = len(train_data)

train_data = train_data.dropna()

rows_after = len(train_data)

print(
    f"\nRows removed because of NaN: "
    f"{rows_before - rows_after:,}"
)

print(
    f"Rows used for scaler fitting: "
    f"{rows_after:,}"
)


# ============================================================
# FIT SCALER
# ============================================================

scaler = MinMaxScaler()

scaler.fit(train_data)


# ============================================================
# SAVE SCALER
# ============================================================

scaler_path = MODEL_DIR / "normal_behavior_scaler.pkl"

joblib.dump(
    scaler,
    scaler_path
)

print("\n" + "=" * 70)
print("SCALER FIT COMPLETE")
print("=" * 70)

print(f"Features: {len(FEATURE_COLS)}")
print(f"Training rows used: {rows_after:,}")

print(f"\nScaler saved to:")
print(scaler_path)

print("\nIMPORTANT:")
print("Scaler was fitted ONLY on the 45 healthy training recordings.")

print("=" * 70)