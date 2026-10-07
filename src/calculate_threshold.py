import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\bridge_monitor")

SEQUENCE_DIR = BASE_DIR / "data" / "sequences"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "lstm_autoencoder_best.keras"

SUMMARY_FILE = SEQUENCE_DIR / "sequence_summary.csv"

OUTPUT_FILE = MODEL_DIR / "validation_reconstruction_errors.csv"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32

# Percentile used to define anomaly threshold
THRESHOLD_PERCENTILE = 95


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ANOMALY THRESHOLD CALCULATION")
print("=" * 70)

print(f"\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD VALIDATION FILES
# ============================================================

summary_df = pd.read_csv(SUMMARY_FILE)

validation_files = summary_df[
    summary_df["split"] == "validation"
]["output_file"].tolist()

print(f"\nValidation sequence files: {len(validation_files)}")


# ============================================================
# CALCULATE RECONSTRUCTION ERROR
# ============================================================

all_errors = []

print("\nCalculating reconstruction errors...")

for i, filename in enumerate(validation_files, start=1):

    filepath = SEQUENCE_DIR / filename

    print(
        f"[{i}/{len(validation_files)}] "
        f"{filename}"
    )

    data = np.load(filepath)

    X = data["X"].astype(np.float32)

    if len(X) == 0:
        continue

    # --------------------------------------------------------
    # Reconstruct sequences
    # --------------------------------------------------------

    reconstructed = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # --------------------------------------------------------
    # Mean squared error for each sequence
    # --------------------------------------------------------

    errors = np.mean(
        np.square(X - reconstructed),
        axis=(1, 2)
    )

    all_errors.extend(errors.tolist())


# ============================================================
# CONVERT TO ARRAY
# ============================================================

errors = np.array(
    all_errors,
    dtype=np.float32
)


# ============================================================
# ERROR STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("RECONSTRUCTION ERROR STATISTICS")
print("=" * 70)

print(f"Number of validation sequences: {len(errors):,}")

print(f"\nMinimum : {np.min(errors):.8f}")
print(f"Mean    : {np.mean(errors):.8f}")
print(f"Median  : {np.median(errors):.8f}")
print(f"Maximum : {np.max(errors):.8f}")

print("\nPercentiles:")

for percentile in [90, 95, 97, 99, 99.5]:

    value = np.percentile(
        errors,
        percentile
    )

    print(
        f"{percentile:5.1f}% : {value:.8f}"
    )


# ============================================================
# 95th PERCENTILE THRESHOLD
# ============================================================

threshold = np.percentile(
    errors,
    THRESHOLD_PERCENTILE
)

print("\n" + "=" * 70)

print(
    f"ANOMALY THRESHOLD "
    f"({THRESHOLD_PERCENTILE}th percentile):"
)

print(f"{threshold:.8f}")

print("=" * 70)


# ============================================================
# SAVE ERRORS
# ============================================================

errors_df = pd.DataFrame({
    "reconstruction_error": errors
})

errors_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nValidation errors saved to:")
print(OUTPUT_FILE)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_file = MODEL_DIR / "anomaly_threshold.txt"

with open(threshold_file, "w") as f:
    f.write(str(threshold))

print("\nThreshold saved to:")
print(threshold_file)

print("\n" + "=" * 70)
print("THRESHOLD CALCULATION COMPLETE")
print("=" * 70)