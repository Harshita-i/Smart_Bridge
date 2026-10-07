import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\bridge_monitor")

SCALED_DIR = BASE_DIR / "data" / "scaled"
SPLIT_FILE = BASE_DIR / "data" / "splits" / "recording_split.csv"
OUTPUT_DIR = BASE_DIR / "data" / "sequences"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 200       # 200 samples = 1 second at 200 Hz
STEP_SIZE = 200         # non-overlapping windows for now

# 24 reliable features used by the first model
FEATURES = [
    "ch_1",
    "ch_2",
    "ch_3",
    "ch_4",
    "ch_5",
    "ch_6",
    "ch_7",
    "ch_8",
    "ch_9",
    "ch_10",
    "ch_11",
    "ch_13",
    "ch_14",
    "ch_15",
    "ch_16",
    "ch_18",
    "ch_19",
    "ch_20",
    "ch_21",
    "ch_22",
    "ch_25",
    "ch_28",
    "ch_29",
    "ch_30",
]

SCALED_FEATURES = [f"{col}_scaled" for col in FEATURES]


# ============================================================
# LOAD RECORDING SPLIT
# ============================================================

print("=" * 70)
print("LSTM SEQUENCE GENERATION")
print("=" * 70)

split_df = pd.read_csv(SPLIT_FILE)

print("\nRecording split:")
print(split_df["split"].value_counts())


# ============================================================
# FUNCTION TO CREATE SEQUENCES
# ============================================================

def create_sequences(df):
    """
    Create fixed-length sequences.

    A sequence is created only when:
    - all 200 samples belong to the same segment
    - all 24 features are valid
    - no missing values occur
    """

    sequences = []

    # --------------------------------------------------------
    # Make sure timestamp is sorted
    # --------------------------------------------------------

    df = df.sort_values("ts").reset_index(drop=True)

    # --------------------------------------------------------
    # Find valid rows
    # --------------------------------------------------------

    valid_mask = df["valid_for_model"].astype(bool).to_numpy()

    # --------------------------------------------------------
    # Work segment-by-segment
    # --------------------------------------------------------

    if "segment_id" in df.columns:
        segments = df.groupby("segment_id", sort=False)
    else:
        # Fallback: treat entire recording as one segment
        segments = [(0, df)]

    for segment_id, segment in segments:

        segment = segment.reset_index(drop=True)

        # Need at least WINDOW_SIZE rows
        if len(segment) < WINDOW_SIZE:
            continue

        values = segment[SCALED_FEATURES].to_numpy(dtype=np.float32)

        segment_valid = valid_mask[segment.index]

        # ----------------------------------------------------
        # Generate windows
        # ----------------------------------------------------

        for start in range(
            0,
            len(segment) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):

            end = start + WINDOW_SIZE

            # Check validity of every row
            if not segment_valid[start:end].all():
                continue

            window = values[start:end]

            # Extra safety check
            if not np.isfinite(window).all():
                continue

            sequences.append(window)

    if not sequences:
        return np.empty(
            (0, WINDOW_SIZE, len(SCALED_FEATURES)),
            dtype=np.float32
        )

    return np.stack(sequences).astype(np.float32)


# ============================================================
# PROCESS EACH RECORDING
# ============================================================

total_sequences = 0

summary = []

for index, row in split_df.iterrows():

    filename = row["filename"]
    split = row["split"]

    input_file = SCALED_DIR / filename

    print("\n" + "-" * 70)
    print(f"Recording {index + 1}/{len(split_df)}")
    print(f"File : {filename}")
    print(f"Split: {split}")

    if not input_file.exists():
        print("WARNING: File not found. Skipping.")
        continue

    # --------------------------------------------------------
    # Load scaled recording
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    X = create_sequences(df)

    print(f"Rows in recording : {len(df):,}")
    print(f"Sequences created : {len(X):,}")

    # --------------------------------------------------------
    # Save recording-specific sequences
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / f"{Path(filename).stem}_sequences.npz"

    np.savez_compressed(
        output_file,
        X=X
    )

    print(f"Saved             : {output_file}")

    total_sequences += len(X)

    summary.append({
        "filename": filename,
        "split": split,
        "rows": len(df),
        "sequences": len(X),
        "output_file": output_file.name
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary)

summary_file = OUTPUT_DIR / "sequence_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE GENERATION COMPLETE")
print("=" * 70)

print(f"Window size       : {WINDOW_SIZE} samples")
print(f"Window duration   : {WINDOW_SIZE / 200:.2f} seconds")
print(f"Step size         : {STEP_SIZE} samples")
print(f"Number of features: {len(SCALED_FEATURES)}")
print(f"Total sequences   : {total_sequences:,}")

print("\nSequences by split:")

print(
    summary_df.groupby("split")["sequences"]
    .sum()
    .to_string()
)

print(f"\nSummary saved to:")
print(summary_file)

print("=" * 70)