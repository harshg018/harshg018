from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.utils.config import settings
from app.db.models import Base
import os


os.makedirs("data", exist_ok=True)
engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
