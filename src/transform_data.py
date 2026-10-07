import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path(r"C:\bridge_monitor\data\processed")
SPLIT_FILE = Path(r"C:\bridge_monitor\data\splits\recording_split.csv")

SCALER_FILE = Path(
    r"C:\bridge_monitor\models\normal_behavior_scaler.pkl"
)

OUTPUT_DIR = Path(
    r"C:\bridge_monitor\data\scaled"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLS = (
    [f"ch_{i}" for i in range(1, 12)] +
    [f"ch_{i}" for i in range(13, 17)] +
    [f"ch_{i}" for i in range(18, 23)] +
    ["ch_25", "ch_28", "ch_29", "ch_30"]
)

print(f"Number of features: {len(FEATURE_COLS)}")


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(SCALER_FILE)

print("Scaler loaded successfully.")


# ============================================================
# LOAD SPLITS
# ============================================================

splits = pd.read_csv(SPLIT_FILE)

print(
    f"Total recordings: {len(splits)}"
)


# ============================================================
# PROCESS RECORDINGS
# ============================================================

for _, row in splits.iterrows():

    filename = row["filename"]
    split = row["split"]

    input_file = PROCESSED_DIR / filename
    output_file = OUTPUT_DIR / filename

    print(
        f"\nProcessing: {filename}"
    )

    df = pd.read_csv(input_file)

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["ts"] = pd.to_datetime(
        df["ts"],
        format="mixed"
    )

    # --------------------------------------------------------
    # Interpolate isolated ch_11 NaNs
    #
    # These were only 8 isolated samples in the training data.
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
    # Verify features
    # --------------------------------------------------------

    missing_features = [
        col
        for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"{filename} is missing: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Check remaining NaNs
    # --------------------------------------------------------

    feature_data = df[FEATURE_COLS].copy()

    nan_counts = feature_data.isna().sum()

    if nan_counts.sum() > 0:

        print(
            "\nWARNING: NaN values remain:"
        )

        print(
            nan_counts[
                nan_counts > 0
            ]
        )

        # We do NOT fill large missing regions.
        # Those rows will be marked invalid.

    # --------------------------------------------------------
    # Create validity mask
    # --------------------------------------------------------

    valid_mask = (
        ~feature_data.isna().any(axis=1)
    )

    # --------------------------------------------------------
    # Create scaled feature columns
    # --------------------------------------------------------

    scaled_columns = [
        f"{col}_scaled"
        for col in FEATURE_COLS
    ]

    df[scaled_columns] = pd.NA

    # --------------------------------------------------------
    # Transform only rows with complete feature data
    # --------------------------------------------------------

    if valid_mask.any():

        scaled_values = scaler.transform(
            feature_data.loc[
                valid_mask,
                FEATURE_COLS
            ]
        )

        df.loc[
            valid_mask,
            scaled_columns
        ] = scaled_values

    # --------------------------------------------------------
    # Validity flag
    # --------------------------------------------------------

    df["valid_for_model"] = valid_mask

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Valid rows: {valid_mask.sum():,}"
    )

    print(
        f"Invalid rows: {(~valid_mask).sum():,}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("SCALING COMPLETE")
print("=" * 70)

print(
    f"Scaled recordings saved to:\n{OUTPUT_DIR}"
)

print("=" * 70)