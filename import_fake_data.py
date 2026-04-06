"""
Скрипт импорта тестовых данных для ВКР.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func

# Импорты - ВАЖНО: сначала db, потом models
from db import SessionLocal, engine
from models import Product, ProductOffer, Company, Category, PropertyName, ProductProperty, Base


# Создаем таблицы ТОЛЬКО после всех импортов
def create_tables():
    Base.metadata.create_all(bind=engine)


# =============================================================================
# ДЕМО-ДАННЫЕ
# =============================================================================

DEMO_DATA = [
    {
        "short_name": "Samsung Galaxy A54",
        "price": 29990.0,
        "category": "Электроника",
        "company": "Samsung",
        "properties": {
            "экран": "6.4 дюйма",
            "память": "128 ГБ",
            "камера": "50 МП",
        },
    },
    {
        "short_name": "Sony WH-1000XM5",
        "price": 34990.0,
        "category": "Аудио",
        "company": "Sony",
        "properties": {
            "тип": "Полноразмерные",
            "подключение": "Bluetooth",
        },
    },
    {
        "short_name": "MacBook Air M2",
        "price": 99990.0,
        "category": "Компьютеры",
        "company": "Apple",
        "properties": {
            "процессор": "Apple M2",
            "память": "8 ГБ",
        },
    },
]


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_or_create_company(session: Session, name: str):
    company = session.query(Company).filter(Company.name == name).first()
    if not company:
        company = Company(name=name)
        session.add(company)
        session.commit()
        session.refresh(company)
    return company


def get_or_create_category(session: Session, name: str):
    category = session.query(Category).filter(Category.name == name).first()
    if not category:
        max_id = session.query(func.max(Category.category_id)).scalar() or 0
        category = Category(category_id=max_id + 1, name=name)
        session.add(category)
        session.commit()
        session.refresh(category)
    return category


def get_or_create_property_name(session: Session, name: str):
    prop = session.query(PropertyName).filter(PropertyName.name == name).first()
    if not prop:
        prop = PropertyName(name=name, value_type="text")
        session.add(prop)
        session.commit()
        session.refresh(prop)
    return prop


def get_next_id(session, model):
    max_id = session.query(func.max(getattr(model, list(model.__table__.primary_key.columns)[0].name))).scalar() or 0
    return max_id + 1


# =============================================================================
# ИМПОРТ
# =============================================================================

def import_data(session: Session):
    added = 0
    for item in DEMO_DATA:
        # Проверяем дубликаты
        exists = session.query(ProductOffer).filter(
            ProductOffer.short_name == item['short_name']
        ).first()

        if exists:
            print(f"   ⏭️ Пропущено: {item['short_name']}")
            continue

        try:
            # Создаём связи
            company = get_or_create_company(session, item['company'])
            category = get_or_create_category(session, item['category'])

            # Создаём продукт
            product_id = get_next_id(session, Product)
            product = Product(product_id=product_id)
            session.add(product)
            session.flush()

            # Создаём offer
            offer_id = get_next_id(session, ProductOffer)
            offer = ProductOffer(
                offer_id=offer_id,
                product_id=product_id,
                company_id=company.company_id,
                category_id=category.category_id,
                short_name=item['short_name'],
                price_rub=item.get('price'),
            )
            session.add(offer)
            session.flush()

            # Создаём свойства
            for prop_name, prop_value in item['properties'].items():
                pname = get_or_create_property_name(session, prop_name)
                pp = ProductProperty(
                    offer_id=offer_id,
                    property_name_id=pname.property_name_id,
                    value=str(prop_value)
                )
                session.add(pp)

            p = item.get("price")
            if p is not None:
                pname = get_or_create_property_name(session, "Цена")
                pp = ProductProperty(
                    offer_id=offer_id,
                    property_name_id=pname.property_name_id,
                    value=f"{int(round(float(p))):,}".replace(",", "\u00a0") + "\u00a0₽",
                )
                session.add(pp)

            added += 1
            print(f"   ✅ {item['short_name']}")

        except Exception as e:
            session.rollback()
            print(f"   ❌ Ошибка {item['short_name']}: {e}")

    session.commit()
    return added


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("🚀 Импорт данных для ВКР...")
    print("=" * 50)

    # Создаём таблицы
    create_tables()

    db = SessionLocal()
    try:
        added = import_data(db)
        print("=" * 50)
        print(f"✅ Добавлено товаров: {added}")
        print("=" * 50)
    except Exception as e:
        db.rollback()
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()