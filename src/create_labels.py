import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path(r"C:\Users\immad\Downloads\DiB")

INPUT_FILE = RAW_DIR / "recording_analysis.csv"
OUTPUT_FILE = RAW_DIR / "recording_labels.csv"

# ============================================================
# LOAD RECORDING SUMMARY
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["start_time"] = pd.to_datetime(df["start_time"])
df["end_time"] = pd.to_datetime(df["end_time"])

# ============================================================
# IMPORTANT TIME POINT
# ============================================================
# The fracture recording is:
#
# 2023_03_09T23_45_25.csv
#
# The extreme A5 response occurs around:
# 2023-03-09 23:47:11
#
# We keep the entire fracture recording separate.
# ============================================================

fracture_file = "2023_03_09T23_45_25.csv"

# ============================================================
# ASSIGN RECORDING-LEVEL LABEL
# ============================================================

def assign_state(row):

    if row["filename"] == fracture_file:
        return "FRACTURE_EVENT"

    elif row["start_time"] < pd.Timestamp("2023-03-09 23:45:25"):
        return "PRE_FRACTURE"

    else:
        return "POST_FRACTURE"


df["state"] = df.apply(assign_state, axis=1)

# ============================================================
# ADD NUMERIC LABEL
# ============================================================

state_to_label = {
    "PRE_FRACTURE": 0,
    "FRACTURE_EVENT": 1,
    "POST_FRACTURE": 2
}

df["label"] = df["state"].map(state_to_label)

# ============================================================
# SORT
# ============================================================

df = df.sort_values("start_time")

# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("=" * 80)
print("RECORDING LABELING COMPLETE")
print("=" * 80)

print("\nRecording counts:")
print(df["state"].value_counts())

print("\n\nRecordings by state:")

print(
    df[
        [
            "filename",
            "start_time",
            "end_time",
            "state",
            "label"
        ]
    ].to_string(index=False)
)

print("\n\nSaved to:")
print(OUTPUT_FILE)

print("=" * 80)