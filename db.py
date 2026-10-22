"""Подключение к PostgreSQL. Параметры берутся из окружения / файла .env."""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# Значение по умолчанию совпадает с docker-compose.yml (локальная разработка).
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://shop:shop@localhost:5432/medialux",
)

engine = create_engine(DATABASE_URL, echo=False, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)
Base = declarative_base()
