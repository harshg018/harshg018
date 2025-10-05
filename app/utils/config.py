import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/brand_rep.db")
    hf_model: str = os.getenv("HF_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment")
    device: str = os.getenv("DEVICE", "cpu")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")


settings = Settings()
