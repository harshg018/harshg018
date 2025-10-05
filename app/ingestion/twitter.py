from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

try:
    import snscrape.modules.twitter as sntwitter
    _SNSCRAPE = True
except Exception:
    _SNSCRAPE = False


@dataclass
class Tweet:
    id: str
    date: datetime
    content: str
    username: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    image_url: Optional[str] = None


def search_tweets(query: str, limit: int = 200) -> pd.DataFrame:
    if not _SNSCRAPE:
        return pd.DataFrame(columns=["id", "date", "content", "username", "latitude", "longitude", "image_url"])

    results: list[Tweet] = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        media_url = None
        if getattr(tweet, "media", None):
            try:
                if tweet.media:
                    first = tweet.media[0]
                    if getattr(first, "fullUrl", None):
                        media_url = first.fullUrl
            except Exception:
                pass
        lat = getattr(getattr(tweet, "coordinates", None), "latitude", None)
        lon = getattr(getattr(tweet, "coordinates", None), "longitude", None)
        results.append(
            Tweet(
                id=str(tweet.id),
                date=tweet.date,
                content=tweet.content or "",
                username=tweet.user.username if tweet.user else "",
                latitude=lat,
                longitude=lon,
                image_url=media_url,
            )
        )
    df = pd.DataFrame([t.__dict__ for t in results])
    return df
