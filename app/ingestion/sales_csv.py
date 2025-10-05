import pandas as pd
from typing import List

REQUIRED_COLUMNS = [
    "date",
    "region",
    "revenue",
    "engagement",
]


def read_sales_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["revenue", "engagement"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["region"] = df["region"].astype(str)
    return df[REQUIRED_COLUMNS]
