from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_geo_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["latitude", "longitude", "sentiment_score"])
    geo = df[["latitude", "longitude", "sentiment_score"]].dropna()
    return geo


def compute_time_series(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "avg_sentiment", "count"])    
    ts = (
        df.assign(date=pd.to_datetime(df["created_at"]).dt.to_period(freq).dt.to_timestamp())
          .groupby("date")
          .agg(avg_sentiment=("sentiment_score", "mean"), count=("id", "count"))
          .reset_index()
    )
    return ts


def correlate_with_sales(posts: pd.DataFrame, sales: pd.DataFrame) -> Tuple[float, float]:
    if posts.empty or sales.empty:
        return float("nan"), float("nan")
    p = posts.copy()
    p["date"] = pd.to_datetime(p["created_at"]).dt.date
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.date
    merged = (
        p.groupby("date").agg(avg_sent=("sentiment_score", "mean"), engagement=("id", "count")).reset_index()
        .merge(s.groupby("date").agg(revenue=("revenue", "sum"), engagement_sales=("engagement", "sum")).reset_index(), on="date", how="inner")
    )
    if merged.empty:
        return float("nan"), float("nan")
    corr_sent_rev = merged["avg_sent"].corr(merged["revenue"])  # type: ignore[arg-type]
    corr_engage = merged["engagement"].corr(merged["engagement_sales"])  # type: ignore[arg-type]
    return float(corr_sent_rev), float(corr_engage)
