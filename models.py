from sqlalchemy import BigInteger, Boolean, Column, Date, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import TSVECTOR

from db import Base


class Product(Base):
    __tablename__ = "products"

    product_id = Column(BigInteger, primary_key=True)

    offers = relationship("ProductOffer", back_populates="product")


class Company(Base):
    __tablename__ = "companies"

    company_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)

    offers = relationship("ProductOffer", back_populates="company")


class Category(Base):
    __tablename__ = "categories"

    category_id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    parent_id = Column(
        BigInteger,
        ForeignKey("categories.category_id", ondelete="SET NULL"),
        nullable=True,
    )

    parent = relationship("Category", remote_side=[category_id], backref="children")
    offers = relationship("ProductOffer", back_populates="category")


class ProductOffer(Base):
    __tablename__ = "product_offers"

    offer_id = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id = Column(
        BigInteger,
        ForeignKey("products.product_id", ondelete="CASCADE"),
        nullable=False,
    )
    company_id = Column(
        Integer,
        ForeignKey("companies.company_id", ondelete="CASCADE"),
        nullable=False,
    )
    category_id = Column(
        BigInteger,
        ForeignKey("categories.category_id", ondelete="SET NULL"),
        nullable=True,
    )
    short_name = Column(Text, nullable=True)
    # Основная цена в рублях (для витрины и суммы в корзине); при NULL можно подставить цену из характеристик.
    price_rub = Column(Numeric(12, 2), nullable=True)

    product = relationship("Product", back_populates="offers")
    company = relationship("Company", back_populates="offers")
    category = relationship("Category", back_populates="offers")
    properties = relationship(
        "ProductProperty",
        back_populates="offer",
        cascade="all, delete-orphan",
    )


class PropertyName(Base):
    __tablename__ = "property_names"

    property_name_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)
    value_type = Column(String(32), nullable=True)

    properties = relationship("ProductProperty", back_populates="property_name")


class ProductProperty(Base):
    __tablename__ = "product_properties"

    offer_id = Column(
        BigInteger,
        ForeignKey("product_offers.offer_id", ondelete="CASCADE"),
        primary_key=True,
    )
    property_name_id = Column(
        Integer,
        ForeignKey("property_names.property_name_id", ondelete="CASCADE"),
        primary_key=True,
    )
    value = Column(Text, nullable=True)

    offer = relationship("ProductOffer", back_populates="properties")
    property_name = relationship("PropertyName", back_populates="properties")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(150), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    is_superuser = Column(Boolean, nullable=False, default=False)


class ShopOrder(Base):
    """Заказы витрины (оформление корзины можно подключить позже)."""

    __tablename__ = "shop_orders"

    order_id = Column(Integer, primary_key=True, autoincrement=True)
    total_amount = Column(Numeric(12, 2), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AnalyticsDailyVisit(Base):
    """Агрегированные визиты по календарным дням (для конверсии)."""

    __tablename__ = "analytics_daily_visits"

    stat_date = Column(Date, primary_key=True)
    visits = Column(Integer, nullable=False, default=0)


class SearchLog(Base):
    """Поисковые запросы и клики по выдаче (метрики, A/B по режиму поиска)."""

    __tablename__ = "search_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    clicked_offer_id = Column(BigInteger, nullable=True, index=True)
    results_count = Column(Integer, nullable=False)
    # hybrid | semantic | fulltext — для сравнения режимов по CTR
    search_mode = Column(String(16), nullable=False, server_default="hybrid")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class SearchIndex(Base):
    __tablename__ = "search_index"

    offer_id = Column(
        BigInteger,
        ForeignKey("product_offers.offer_id", ondelete="CASCADE"),
        primary_key=True,
    )

    # агрегированный текст, который индексируем (для отладки и возможных rebuild)
    text = Column(Text, nullable=False)

    # PostgreSQL full-text
    tsv = Column(TSVECTOR, nullable=True)

    # Эмбеддинг храним как JSON-строку (работает без pgvector).
    # Если pgvector extension доступен, build_job создаст отдельную колонку
    # `embedding_vec vector(384)` и будет заполнять её тоже.
    embedding = Column(Text, nullable=True)

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    offer = relationship("ProductOffer")
