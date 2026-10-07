import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"C:\Users\immad\Downloads\DiB")

files = [
    f for f in RAW_DIR.glob("*.csv")
    if f.name not in [
        "dataset_summary.csv",
        "recording_analysis.csv",
        "recording_labels.csv"
    ]
]

results = []

for path in sorted(files):

    df = pd.read_csv(path)

    for fileid, group in df.groupby("fileid"):

        results.append({
            "filename": path.name,
            "fileid": fileid,
            "start_time": group["ts"].iloc[0],
            "end_time": group["ts"].iloc[-1],
            "rows": len(group),
            "duration_seconds": (
                pd.to_datetime(group["ts"].iloc[-1])
                - pd.to_datetime(group["ts"].iloc[0])
            ).total_seconds()
        })

result = pd.DataFrame(results)

result["start_time"] = pd.to_datetime(result["start_time"])
result = result.sort_values("start_time")

output = RAW_DIR / "fileid_analysis.csv"
result.to_csv(output, index=False)

print("=" * 80)
print("FILEID ANALYSIS")
print("=" * 80)

print(f"\nTotal individual file IDs: {len(result)}")

print("\nFirst 20:")
print(result.head(20).to_string(index=False))

print("\nLast 20:")
print(result.tail(20).to_string(index=False))

print("\n\nFILEID COUNTS PER CSV:")
print(
    result.groupby("filename")
    .size()
    .to_string()
)

print("\n\nSaved to:")
print(output)