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
THRESHOLD_FILE = MODEL_DIR / "anomaly_threshold.txt"

OUTPUT_FILE = MODEL_DIR / "test_anomaly_results.csv"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("ANOMALY EVALUATION")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD THRESHOLD
# ============================================================

with open(THRESHOLD_FILE, "r") as f:
    threshold = float(f.read().strip())

print(f"\nAnomaly threshold: {threshold:.8f}")


# ============================================================
# LOAD SEQUENCE SUMMARY
# ============================================================

summary_df = pd.read_csv(SUMMARY_FILE)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_split(split_name):

    files = summary_df[
        summary_df["split"] == split_name
    ]["output_file"].tolist()

    print("\n" + "=" * 70)
    print(f"EVALUATING: {split_name}")
    print("=" * 70)

    print(f"Number of files: {len(files)}")

    all_errors = []
    all_filenames = []

    for i, filename in enumerate(files, start=1):

        filepath = SEQUENCE_DIR / filename

        print(
            f"[{i}/{len(files)}] {filename}"
        )

        data = np.load(filepath)

        X = data["X"].astype(np.float32)

        if len(X) == 0:
            continue

        # ----------------------------------------------------
        # Reconstruct sequences
        # ----------------------------------------------------

        reconstructed = model.predict(
            X,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        # ----------------------------------------------------
        # Reconstruction error per sequence
        # ----------------------------------------------------

        errors = np.mean(
            np.square(X - reconstructed),
            axis=(1, 2)
        )

        all_errors.extend(errors.tolist())

        all_filenames.extend(
            [filename] * len(errors)
        )

    errors = np.array(
        all_errors,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Anomaly classification
    # --------------------------------------------------------

    anomalous = errors > threshold

    anomaly_count = int(
        np.sum(anomalous)
    )

    total_count = len(errors)

    anomaly_percentage = (
        anomaly_count / total_count * 100
        if total_count > 0
        else 0
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(f"Total sequences : {total_count:,}")

    print(f"Minimum error   : {np.min(errors):.8f}")
    print(f"Mean error      : {np.mean(errors):.8f}")
    print(f"Median error    : {np.median(errors):.8f}")
    print(f"Maximum error   : {np.max(errors):.8f}")

    print(
        f"\nAnomalous sequences : "
        f"{anomaly_count:,}"
    )

    print(
        f"Normal sequences    : "
        f"{total_count - anomaly_count:,}"
    )

    print(
        f"Anomaly percentage  : "
        f"{anomaly_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    result_df = pd.DataFrame({
        "split": split_name,
        "filename": all_filenames,
        "reconstruction_error": errors,
        "threshold": threshold,
        "anomaly": anomalous
    })

    return result_df


# ============================================================
# EVALUATE FRACTURE
# ============================================================

fracture_results = evaluate_split(
    "test_fracture"
)


# ============================================================
# EVALUATE POST-FRACTURE
# ============================================================

post_results = evaluate_split(
    "test_post_fracture"
)


# ============================================================
# COMBINE
# ============================================================

results = pd.concat(
    [
        fracture_results,
        post_results
    ],
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANOMALY EVALUATION SUMMARY")
print("=" * 70)

for split_name, group in results.groupby("split"):

    print(f"\n{split_name}")

    print(f"Sequences: {len(group):,}")

    print(
        f"Anomalous: "
        f"{group['anomaly'].sum():,}"
    )

    print(
        f"Anomaly %: "
        f"{group['anomaly'].mean() * 100:.2f}%"
    )

    print(
        f"Mean error: "
        f"{group['reconstruction_error'].mean():.8f}"
    )

    print(
        f"Maximum error: "
        f"{group['reconstruction_error'].max():.8f}"
    )


print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 70)