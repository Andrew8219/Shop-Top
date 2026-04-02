# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://shop:shop@localhost:5432/medialux"
)

engine = create_engine(DEFAULT_DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)
Base = declarative_base()