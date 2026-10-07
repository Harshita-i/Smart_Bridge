import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\immad\Downloads\DiB")

# Sensors that showed suspicious ranges
CHECKS = {
    "ch_11": lambda x: (x.abs() > 10000),
    "ch_12": lambda x: (x.abs() > 10000),
    "ch_22": lambda x: (x.abs() > 100),
}

csv_files = sorted(
    p for p in DATA_DIR.glob("*.csv")
    if p.name in [
        "2023_02_20T03_10_53.csv",
        "2023_02_21T07_54_28.csv",
        "2023_03_02T22_43_19.csv",
        "2023_03_09T23_45_25.csv",
    ]
)

for file_path in csv_files:

    print("\n" + "=" * 80)
    print(file_path.name)
    print("=" * 80)

    df = pd.read_csv(file_path)

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], format="mixed")

    for col, condition in CHECKS.items():

        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        mask = condition(values)

        count = int(mask.sum())

        print(f"\n{col}: {count} suspicious samples")

        if count == 0:
            continue

        suspicious = df.loc[
            mask,
            [c for c in ["ts", "fileid", col] if c in df.columns]
        ]

        print("\nFirst 10 suspicious samples:")
        print(suspicious.head(10).to_string(index=False))

        print("\nLast 10 suspicious samples:")
        print(suspicious.tail(10).to_string(index=False))

        if "fileid" in suspicious.columns:
            print("\nSuspicious samples by fileid:")
            print(
                suspicious["fileid"]
                .value_counts()
                .sort_index()
                .to_string()
            )

        if "ts" in suspicious.columns:
            print("\nTime range:")
            print("First:", suspicious["ts"].min())
            print("Last :", suspicious["ts"].max())


print("\n" + "=" * 80)
print("EXTREME-VALUE INSPECTION COMPLETE")
print("=" * 80)