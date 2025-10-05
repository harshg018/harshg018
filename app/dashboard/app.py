import os
import io
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from app.db.session import init_db, SessionLocal
from app.db.models import Post, SalesMetric
from app.analysis.text_sentiment import TextSentimentAnalyzer
from app.analysis.image_emotion import ImageEmotionAnalyzer
from app.ingestion.twitter import search_tweets
from app.ingestion.csv_import import read_user_generated_csv
from sqlalchemy import select

st.set_page_config(page_title="Brand Reputation Monitor", layout="wide")

@st.cache_resource
def get_analyzers():
    return TextSentimentAnalyzer(), ImageEmotionAnalyzer()


def save_posts(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    text_analyzer, image_analyzer = get_analyzers()
    with SessionLocal() as db:
        new_count = 0
        for _, row in df.iterrows():
            sentiment = text_analyzer.analyze(str(row.get("text", "")))
            emotion_label = None
            emotion_score = None
            img_url = row.get("image_url")
            if isinstance(img_url, str) and len(img_url) > 5:
                emo = image_analyzer.analyze(img_url)
                emotion_label = str(emo.get("label"))
                emotion_score = float(emo.get("score", 0.0))
            post = Post(
                source=str(row.get("source", "twitter")),
                external_id=str(row.get("external_id", "")) if row.get("external_id") is not None else None,
                text=str(row.get("text", "")),
                image_url=img_url,
                created_at=row.get("created_at") if pd.notna(row.get("created_at")) else datetime.utcnow(),
                user_location=str(row.get("user_location", "")) if row.get("user_location") is not None else None,
                latitude=float(row.get("latitude")) if pd.notna(row.get("latitude")) else None,
                longitude=float(row.get("longitude")) if pd.notna(row.get("longitude")) else None,
                sentiment_label=str(sentiment.get("label")),
                sentiment_score=float(sentiment.get("score", 0.0)),
                emotion_label=emotion_label,
                emotion_score=emotion_score,
            )
            db.add(post)
            new_count += 1
        db.commit()
        return new_count


def load_posts(limit: int = 5000) -> pd.DataFrame:
    with SessionLocal() as db:
        stmt = select(Post).order_by(Post.created_at.desc()).limit(limit)
        rows = db.execute(stmt).scalars().all()
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "source": r.source,
                "external_id": r.external_id,
                "text": r.text,
                "image_url": r.image_url,
                "created_at": r.created_at,
                "user_location": r.user_location,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "sentiment_label": r.sentiment_label,
                "sentiment_score": r.sentiment_score,
                "emotion_label": r.emotion_label,
                "emotion_score": r.emotion_score,
            })
        return pd.DataFrame(data)


def load_sales() -> pd.DataFrame:
    with SessionLocal() as db:
        rows = db.execute(select(SalesMetric)).scalars().all()
        data = [{
            "id": r.id,
            "date": r.date,
            "region": r.region,
            "revenue": r.revenue,
            "engagement": r.engagement,
        } for r in rows]
        return pd.DataFrame(data)


from app.analytics.metrics import compute_geo_heatmap, compute_time_series, correlate_with_sales


def sidebar_controls():
    st.sidebar.header("Ingestion")
    query = st.sidebar.text_input("Twitter query", "brandname OR #brandname lang:en")
    limit = st.sidebar.slider("Tweet limit", 50, 1000, 200, step=50)
    fetch = st.sidebar.button("Fetch Tweets")

    st.sidebar.subheader("Upload Reviews/Chats CSV")
    up = st.sidebar.file_uploader("CSV with columns: source, external_id, text, created_at, user_location, latitude, longitude, image_url", type=["csv"])
    process_csv = st.sidebar.button("Process CSV")

    st.sidebar.header("Filters")
    days = st.sidebar.slider("Lookback days", 1, 90, 14)
    min_date = datetime.utcnow() - timedelta(days=days)

    return query, limit, fetch, up, process_csv, min_date


def render_overview(df: pd.DataFrame):
    st.subheader("Overview")
    if df.empty:
        st.info("No data yet. Use the sidebar to ingest tweets or upload CSVs.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", len(df))
    col2.metric("% Positive", f"{(df.sentiment_label=='positive').mean()*100:.1f}%")
    col3.metric("% Neutral", f"{(df.sentiment_label=='neutral').mean()*100:.1f}%")
    col4.metric("% Negative", f"{(df.sentiment_label=='negative').mean()*100:.1f}%")

    ts = compute_time_series(df)
    fig = px.line(ts, x="date", y="avg_sentiment", title="Average Sentiment Over Time")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Posts")
    st.dataframe(df[["created_at", "source", "text", "sentiment_label", "sentiment_score", "emotion_label"]].head(200))


def render_geo(df: pd.DataFrame):
    st.subheader("Sentiment Heatmap by Geography")
    geo = compute_geo_heatmap(df)
    if geo.empty:
        st.info("No geo-tagged posts available.")
        return
    fig = px.density_mapbox(
        geo.dropna(),
        lat="latitude",
        lon="longitude",
        z="sentiment_score",
        radius=10,
        center=dict(lat=0, lon=0),
        zoom=1,
        mapbox_style="open-street-map",
        title="Sentiment Intensity"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_images(df: pd.DataFrame):
    st.subheader("Image Emotions")
    imgs = df.dropna(subset=["image_url"]).head(30)
    if imgs.empty:
        st.info("No images detected.")
        return
    for _, row in imgs.iterrows():
        cols = st.columns([1,2])
        with cols[0]:
            st.image(row["image_url"], width=200)
        with cols[1]:
            st.write(f"Emotion: {row['emotion_label']} ({row.get('emotion_score', 0.0):.2f})")
            st.write(row["text"][:200] + ("..." if len(row["text"])>200 else ""))


def render_correlation(df: pd.DataFrame):
    st.subheader("Correlation with Sales & Engagement")
    sales = load_sales()
    c1, c2 = correlate_with_sales(df, sales)
    st.write(f"Correlation (Avg Sentiment vs Revenue): {c1:.3f}")
    st.write(f"Correlation (Posts vs Engagement): {c2:.3f}")


SUGGESTIONS = {
    "negative": [
        "Address common complaints in help center and proactive emails",
        "Increase support staffing for trending issue time slots",
        "Launch make-good offers or credits in affected regions",
    ],
    "neutral": [
        "Nudge reviews to tip toward positive via loyalty messaging",
        "Highlight product benefits in top channels",
    ],
    "positive": [
        "Amplify UGC with permission; run referral incentives",
        "Feature success stories and customer quotes",
    ],
}


def render_feedback(df: pd.DataFrame):
    st.subheader("Actionable Feedback")
    if df.empty:
        st.info("No data yet.")
        return
    latest_window = df[df["created_at"] > (pd.Timestamp.utcnow() - pd.Timedelta(days=7))]
    if latest_window.empty:
        latest_window = df
    dominant = latest_window["sentiment_label"].mode().iloc[0]
    st.write(f"Dominant sentiment in last week: {dominant}")
    for s in SUGGESTIONS.get(dominant, []):
        st.write(f"- {s}")


def main():
    init_db()
    query, limit, fetch, up, process_csv, min_date = sidebar_controls()

    if fetch and query:
        with st.spinner("Fetching tweets and analyzing..."):
            tweets = search_tweets(query, limit=limit)
            if not tweets.empty:
                tweets = tweets.rename(columns={
                    "id": "external_id",
                    "date": "created_at",
                    "content": "text",
                })
                tweets["source"] = "twitter"
                new_count = save_posts(tweets)
                st.success(f"Ingested {new_count} tweets")

    if process_csv and up is not None:
        with st.spinner("Processing CSV and analyzing..."):
            try:
                df = read_user_generated_csv(up)
                new_count = save_posts(df)
                st.success(f"Ingested {new_count} rows from CSV")
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")

    df = load_posts()
    if not df.empty:
        df = df[df["created_at"] >= min_date]

    tabs = st.tabs(["Overview", "Geo Heatmap", "Images", "Correlation", "Feedback"])

    with tabs[0]:
        render_overview(df)
    with tabs[1]:
        render_geo(df)
    with tabs[2]:
        render_images(df)
    with tabs[3]:
        render_correlation(df)
    with tabs[4]:
        render_feedback(df)


if __name__ == "__main__":
    main()
