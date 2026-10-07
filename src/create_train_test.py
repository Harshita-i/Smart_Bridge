import pandas as pd
from pathlib import Path
import random

DATA_DIR = Path(r"C:\Users\immad\Downloads\DiB")
LABEL_FILE = DATA_DIR / "recording_labels.csv"

OUTPUT_DIR = Path(r"C:\bridge_monitor\data\splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load recording labels
# ------------------------------------------------------------

labels = pd.read_csv(LABEL_FILE)

print("Label distribution:")
print(labels["state"].value_counts())
print()

# ------------------------------------------------------------
# Keep only healthy recordings
# ------------------------------------------------------------

healthy = labels[
    labels["state"] == "PRE_FRACTURE"
].copy()

healthy = healthy.sort_values("filename").reset_index(drop=True)

print(f"Healthy recordings: {len(healthy)}")

# ------------------------------------------------------------
# Reproducible random split
# ------------------------------------------------------------

random.seed(42)

files = healthy["filename"].tolist()

random.shuffle(files)

split_index = int(len(files) * 0.80)

train_files = sorted(files[:split_index])
val_files = sorted(files[split_index:])

# ------------------------------------------------------------
# Create split tables
# ------------------------------------------------------------

train_df = pd.DataFrame({
    "filename": train_files,
    "split": "train",
    "state": "PRE_FRACTURE"
})

val_df = pd.DataFrame({
    "filename": val_files,
    "split": "validation",
    "state": "PRE_FRACTURE"
})

# ------------------------------------------------------------
# Test sets
# ------------------------------------------------------------

fracture = labels[
    labels["state"] == "FRACTURE_EVENT"
].copy()

post_fracture = labels[
    labels["state"] == "POST_FRACTURE"
].copy()

fracture["split"] = "test_fracture"
post_fracture["split"] = "test_post_fracture"

# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

split_df = pd.concat(
    [
        train_df,
        val_df,
        fracture[["filename", "split", "state"]],
        post_fracture[["filename", "split", "state"]]
    ],
    ignore_index=True
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_file = OUTPUT_DIR / "recording_split.csv"

split_df.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RECORDING SPLIT")
print("=" * 70)

print("\nTraining recordings:")
print(len(train_df))

print("\nValidation recordings:")
print(len(val_df))

print("\nFracture test recordings:")
print(len(fracture))

print("\nPost-fracture test recordings:")
print(len(post_fracture))

print("\nSplit distribution:")
print(
    split_df["split"].value_counts()
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 70)