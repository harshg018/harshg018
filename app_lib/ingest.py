from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["text"]
OPTIONAL_COLUMNS = ["brand", "timestamp", "lat", "lon", "source", "image_path"]


def read_events_csv(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    # Normalize columns
    lower_cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_cols)
    for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS]


def read_inbox_csvs(inbox_dir: str) -> pd.DataFrame:
    p = Path(inbox_dir)
    if not p.exists() or not p.is_dir():
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    frames: List[pd.DataFrame] = []
    for f in sorted(p.glob("*.csv")):
        try:
            frames.append(read_events_csv(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)
