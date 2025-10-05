import pandas as pd
from typing import List


REQUIRED_COLUMNS = [
    "source",  # review or chat
    "external_id",
    "text",
    "created_at",
    "user_location",
    "latitude",
    "longitude",
    "image_url",
]


def read_user_generated_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = None
    # standardize dtypes
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[REQUIRED_COLUMNS]
