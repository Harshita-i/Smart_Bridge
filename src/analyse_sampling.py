import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\bridge_monitor\data\processed")

files = sorted(DATA_DIR.glob("*.csv"))

intervals = []

for file in files:

    df = pd.read_csv(file)

    ts = pd.to_datetime(
        df["ts"],
        format="mixed"
    )

    diff = ts.diff().dt.total_seconds().dropna()

    intervals.extend(diff.tolist())


intervals = np.array(intervals)

print("=" * 70)
print("SAMPLING INTERVAL ANALYSIS")
print("=" * 70)

print(f"Total intervals analysed: {len(intervals):,}")

print(f"\nMedian interval: {np.median(intervals):.6f} sec")
print(f"Mean interval:   {np.mean(intervals):.6f} sec")
print(f"Minimum:         {np.min(intervals):.6f} sec")
print(f"Maximum:         {np.max(intervals):.6f} sec")

print("\nMost common intervals:")

values, counts = np.unique(
    np.round(intervals, 6),
    return_counts=True
)

order = np.argsort(counts)[::-1]

for i in order[:10]:

    print(
        f"{values[i]:.6f} sec : "
        f"{counts[i]:,} occurrences"
    )


print("\nEstimated sampling frequency:")

median_interval = np.median(intervals)

print(
    f"{1 / median_interval:.2f} Hz"
)

print("=" * 70)