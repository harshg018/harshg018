from datetime import datetime
from typing import Optional
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, DateTime, Text


class Base(DeclarativeBase):
    pass


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), index=True)  # twitter, review, chat
    external_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    text: Mapped[str] = mapped_column(Text)
    image_url: Mapped[Optional[str]] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True, default=datetime.utcnow)
    user_location: Mapped[Optional[str]] = mapped_column(String(256))
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)

    sentiment_label: Mapped[Optional[str]] = mapped_column(String(16), index=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    emotion_label: Mapped[Optional[str]] = mapped_column(String(32))
    emotion_score: Mapped[Optional[float]] = mapped_column(Float)


class SalesMetric(Base):
    __tablename__ = "sales_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    region: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    revenue: Mapped[float] = mapped_column(Float, default=0.0)
    engagement: Mapped[float] = mapped_column(Float, default=0.0)
