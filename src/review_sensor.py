import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"C:\Users\immad\Downloads\DiB")

analysis_file = DATA_DIR / "all_sensor_analysis.csv"

df = pd.read_csv(analysis_file)

# ------------------------------------------------------------
# Sensor groups
# ------------------------------------------------------------

strain_cols = [f"ch_{i}" for i in range(1, 17)]
accel_cols = [f"ch_{i}" for i in range(18, 23)]

other_cols = [
    "ch_25",
    "ch_28",
    "ch_29",
    "ch_30"
]

sensor_cols = strain_cols + accel_cols + other_cols


# ------------------------------------------------------------
# 1. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. SENSOR COLUMNS WITH MISSING VALUES")
print("=" * 70)

for col in sensor_cols:

    missing_col = f"{col}_missing"

    if missing_col in df.columns:

        total_missing = df[missing_col].sum()

        if total_missing > 0:
            print(f"{col}: {total_missing} missing values")


# ------------------------------------------------------------
# 2. Infinite values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. SENSOR COLUMNS WITH INFINITE VALUES")
print("=" * 70)

for col in sensor_cols:

    inf_col = f"{col}_inf"

    if inf_col in df.columns:

        total_inf = df[inf_col].sum()

        if total_inf > 0:
            print(f"{col}: {total_inf} infinite values")


# ------------------------------------------------------------
# 3. Timestamp gaps
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. RECORDINGS WITH TIMESTAMP GAPS > 0.01 SECONDS")
print("=" * 70)

gap_col = "timestamp_gaps_gt_0.01s"

if gap_col in df.columns:

    gaps = df[df[gap_col] > 0][
        ["file", "start_time", "end_time",
         "median_sampling_interval",
         "max_timestamp_gap",
         gap_col]
    ]

    if len(gaps) == 0:
        print("No recordings have gaps > 0.01 seconds.")
    else:
        print(gaps.to_string(index=False))


# ------------------------------------------------------------
# 4. Sensor ranges
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. GLOBAL SENSOR RANGES")
print("=" * 70)

for col in sensor_cols:

    min_col = f"{col}_min"
    max_col = f"{col}_max"

    if min_col not in df.columns:
        continue

    global_min = df[min_col].min()
    global_max = df[max_col].max()

    print(
        f"{col:>6} : "
        f"min = {global_min:.6g}, "
        f"max = {global_max:.6g}"
    )


# ------------------------------------------------------------
# 5. Largest absolute values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. RECORDINGS WITH EXTREME SENSOR VALUES")
print("=" * 70)

for col in sensor_cols:

    max_col = f"{col}_max"
    min_col = f"{col}_min"

    if max_col not in df.columns:
        continue

    max_idx = df[max_col].idxmax()
    min_idx = df[min_col].idxmin()

    max_value = df.loc[max_idx, max_col]
    min_value = df.loc[min_idx, min_col]

    print(f"\n{col}")
    print(
        f"  Highest : {max_value:.6g} "
        f"({df.loc[max_idx, 'file']})"
    )
    print(
        f"  Lowest  : {min_value:.6g} "
        f"({df.loc[min_idx, 'file']})"
    )


print("\n" + "=" * 70)
print("REVIEW COMPLETE")
print("=" * 70)