"""
Простой скрипт для создания таблиц и импорта данных.
Запускается ОДИН РАЗ для инициализации БД.
"""

import os
from sqlalchemy import create_engine, func, Column, Integer, BigInteger, Text, Float, ForeignKey, Numeric
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func as sql_func

# Создаём Base и таблицы ПРЯМО ЗДЕСЬ (без импорта из models.py)
Base = declarative_base()


# === МОДЕЛИ (копия из models.py для избежания циклических импортов) ===

class Product(Base):
    __tablename__ = "products"
    product_id = Column(BigInteger, primary_key=True)


class Company(Base):
    __tablename__ = "companies"
    company_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)


class Category(Base):
    __tablename__ = "categories"
    category_id = Column(BigInteger, primary_key=True)
    name = Column(Text, nullable=False)
    parent_id = Column(BigInteger, nullable=True)


class ProductOffer(Base):
    __tablename__ = "product_offers"
    offer_id = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id = Column(BigInteger, nullable=False)
    company_id = Column(Integer, nullable=False)
    category_id = Column(BigInteger, nullable=True)
    short_name = Column(Text, nullable=True)
    price_rub = Column(Numeric(12, 2), nullable=True)


class PropertyName(Base):
    __tablename__ = "property_names"
    property_name_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)
    value_type = Column(Text, nullable=True)


class ProductProperty(Base):
    __tablename__ = "product_properties"
    offer_id = Column(BigInteger, primary_key=True)
    property_name_id = Column(Integer, primary_key=True)
    value = Column(Text, nullable=True)


# === НАСТРОЙКА БД ===

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://shop:shop@localhost:5432/medialux"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


# === СОЗДАНИЕ ТАБЛИЦ ===

def create_tables():
    print("🔄 Создание таблиц в базе данных...")
    Base.metadata.create_all(bind=engine)
    print("✅ Таблицы созданы!")


# === ИМПОРТ ДАННЫХ ===

DEMO_DATA = [
    {
        "short_name": "Samsung Galaxy A54",
        "category": "Электроника",
        "company": "Samsung",
        "properties": {"экран": "6.4 дюйма", "память": "128 ГБ", "камера": "50 МП"},
    },
    {
        "short_name": "Sony WH-1000XM5",
        "category": "Аудио",
        "company": "Sony",
        "properties": {"тип": "Полноразмерные", "подключение": "Bluetooth"},
    },
    {
        "short_name": "MacBook Air M2",
        "category": "Компьютеры",
        "company": "Apple",
        "properties": {"процессор": "Apple M2", "память": "8 ГБ"},
    },
    {
        "short_name": "De'Longhi Magnifica S",
        "category": "Техника для дома",
        "company": "De'Longhi",
        "properties": {"тип": "Автоматическая", "мощность": "1450 Вт"},
    },
    {
        "short_name": "Nike Air Max 270",
        "category": "Одежда и обувь",
        "company": "Nike",
        "properties": {"материал": "Текстиль", "сезон": "Лето"},
    },
]


def get_or_create(session, model, defaults=None, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    params = dict(kwargs, **(defaults or {}))
    instance = model(**params)
    session.add(instance)
    session.commit()
    session.refresh(instance)
    return instance


def get_next_id(session, model, id_column):
    max_id = session.query(func.max(id_column)).scalar() or 0
    return max_id + 1


def import_data():
    print("\n Импорт демо-данных...")
    db = SessionLocal()
    added = 0

    try:
        for item in DEMO_DATA:
            # Проверяем дубликаты
            exists = db.query(ProductOffer).filter(
                ProductOffer.short_name == item['short_name']
            ).first()

            if exists:
                print(f"   ⏭️ {item['short_name']} (уже есть)")
                continue

            # Создаём компанию и категорию
            company = get_or_create(db, Company, name=item['company'])
            category = get_or_create(db, Category, name=item['category'],
                                     defaults={'category_id': get_next_id(db, Category, Category.category_id)})

            # Создаём продукт
            product_id = get_next_id(db, Product, Product.product_id)
            product = Product(product_id=product_id)
            db.add(product)
            db.flush()

            # Создаём offer
            offer_id = get_next_id(db, ProductOffer, ProductOffer.offer_id)
            offer = ProductOffer(
                offer_id=offer_id,
                product_id=product_id,
                company_id=company.company_id,
                category_id=category.category_id,
                short_name=item['short_name']
            )
            db.add(offer)
            db.flush()

            # Создаём свойства
            for prop_name, prop_value in item['properties'].items():
                pname = get_or_create(db, PropertyName, name=prop_name,
                                      defaults={'value_type': 'text'})
                pp = ProductProperty(
                    offer_id=offer_id,
                    property_name_id=pname.property_name_id,
                    value=str(prop_value)
                )
                db.add(pp)

            added += 1
            print(f"   ✅ {item['short_name']}")

        db.commit()
        print(f"\n✅ ИМПОРТ ЗАВЕРШЁН! Добавлено: {added} товаров")

    except Exception as e:
        db.rollback()
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


# === ЗАПУСК ===

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ ДЛЯ ВКР")
    print("=" * 60)

    create_tables()
    import_data()

    print("\n💡 Теперь запустите сервер:")
    print("   uvicorn main:app --reload")
    print("\n💡 Проверьте Swagger:")
    print("   http://127.0.0.1:8000/docs")
    print("=" * 60)