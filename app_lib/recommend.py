from __future__ import annotations
import pandas as pd


def generate_recommendations(sent_ts: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Rule-based recommendations per brand based on trends and levels."""
    recs = []
    if agg is None or agg.empty:
        return pd.DataFrame(columns=["brand", "recommendation"])

    # Check trend: last 7 vs prior 7
    if sent_ts is not None and not sent_ts.empty:
        for brand, g in sent_ts.groupby("brand"):
            g = g.sort_values("timestamp")
            last7 = g.tail(7)["avg_score"].mean() if len(g) >= 7 else g["avg_score"].mean()
            prev7 = g.iloc[-14:-7]["avg_score"].mean() if len(g) >= 14 else last7
            level = agg.loc[agg["brand"] == brand, "avg_score"].values
            level = float(level[0]) if len(level) else 0.0

            if last7 < -0.1 and last7 < prev7:
                recs.append({
                    "brand": brand,
                    "recommendation": "Negative sentiment rising. Increase support responsiveness and address top issues in messaging.",
                })
            elif last7 > 0.4 and level > 0.3:
                recs.append({
                    "brand": brand,
                    "recommendation": "Positive momentum. Amplify user stories and run targeted acquisition campaigns.",
                })
            elif abs(last7 - prev7) < 0.05:
                recs.append({
                    "brand": brand,
                    "recommendation": "Stable sentiment. Maintain cadence and monitor for shifts by region and channel.",
                })

    # Fallback if no timeseries
    if not recs:
        for _, row in agg.iterrows():
            brand = row["brand"]
            avg = float(row["avg_score"]) if row["avg_score"] is not None else 0.0
            if avg < -0.1:
                msg = "Address recurring complaints; run make-good offers and clarify policies."
            elif avg > 0.3:
                msg = "Leverage advocates; launch referral incentives and creator partnerships."
            else:
                msg = "Hold steady; test positioning by segment and collect qualitative feedback."
            recs.append({"brand": brand, "recommendation": msg})

    return pd.DataFrame(recs)
