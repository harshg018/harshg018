from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.groupby(["brand"]).agg(
        avg_score=("sentiment_score", "mean"),
        pos_rate=("sentiment_score", lambda x: float(np.mean(np.array(x) > 0))),
        count=("brand", "count"),
    ).reset_index()
    return out


def bucketize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["bucket"] = pd.cut(
        df["sentiment_score"],
        bins=[-1.01, -0.2, 0.2, 1.01],
        labels=["negative", "neutral", "positive"],
    )
    return df


def timeseries_sentiment(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "brand", "avg_score", "count"]) 
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]) 
    out = (
        df.groupby([pd.Grouper(key="timestamp", freq=freq), "brand"]).agg(
            avg_score=("sentiment_score", "mean"),
            count=("brand", "count"),
        )
        .reset_index()
        .sort_values("timestamp")
    )
    return out


def correlate_with_metrics(sent_ts: pd.DataFrame, metrics: pd.DataFrame, on_col: str = "timestamp") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (merged_df, corr_table). metrics must include timestamp and numeric columns."""
    if sent_ts.empty or metrics.empty:
        return sent_ts, pd.DataFrame()
    m = metrics.copy()
    m[on_col] = pd.to_datetime(m[on_col], errors="coerce")
    sent = sent_ts.copy()
    sent[on_col] = pd.to_datetime(sent[on_col], errors="coerce")
    merged = pd.merge(sent, m, on=on_col, how="left")
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    if "avg_score" not in numeric_cols:
        return merged, pd.DataFrame()
    # Compute Pearson correlation of avg_score vs each metric
    rows = []
    for col in numeric_cols:
        if col in {"avg_score", "count"}:
            continue
        s1 = merged["avg_score"].astype(float)
        s2 = merged[col].astype(float)
        if len(s1.dropna()) >= 3 and len(s2.dropna()) >= 3:
            try:
                r = float(s1.corr(s2))
            except Exception:
                r = np.nan
            rows.append({"metric": col, "pearson_r": r})
    corr = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    return merged, corr
