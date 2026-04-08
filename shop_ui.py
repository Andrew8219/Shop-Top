"""
Хелперы для витрины: цена из свойств, краткие атрибуты для карточек.
"""
from __future__ import annotations

import re
from typing import List, Optional

from models import ProductOffer, ProductProperty


_PRICE_NAME_HINTS = ("цен", "price", "стоим", "руб")
_COLOR_NAME_HINTS = (
    "цвет",
    "расцвет",
    "окрас",
    "color",
    "colour",
)
_DEFAULT_COLOR_PAIRS = (
    "Черный / Белый",
    "Синий / Серый",
    "Графит / Серебристый",
    "Красный / Черный",
    "Зеленый / Черный",
    "Фиолетовый / Черный",
)


def _prop_name_lower(p: ProductProperty) -> str:
    if p.property_name and p.property_name.name:
        return p.property_name.name.strip().lower()
    return ""


def is_price_property(p: ProductProperty) -> bool:
    n = _prop_name_lower(p)
    if not n:
        return False
    return any(h in n for h in _PRICE_NAME_HINTS)


def _format_price_rub_display(value: float) -> str:
    n = int(round(float(value)))
    s = f"{n:,}".replace(",", "\u00a0")
    return f"{s}\u00a0₽"


def _price_rub_from_column(offer: ProductOffer) -> Optional[float]:
    raw = getattr(offer, "price_rub", None)
    if raw is None:
        return None
    try:
        v = float(raw)
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None


def get_offer_price_display(offer: ProductOffer) -> Optional[str]:
    """
    Сначала колонка price_rub, иначе характеристика вроде «Цена» / price.
    """
    col = _price_rub_from_column(offer)
    if col is not None:
        return _format_price_rub_display(col)
    best: Optional[str] = None
    for p in offer.properties or []:
        if not is_price_property(p):
            continue
        v = (p.value or "").strip()
        if v:
            best = v
            break
    return best


def get_offer_teaser_lines(offer: ProductOffer, limit: int = 3) -> List[str]:
    """Короткие строки «ключевые атрибуты» для карточки (без цены)."""
    from search_smart import _is_ram_property_name

    rows: List[tuple] = []
    for p in sorted(
        offer.properties or [],
        key=lambda x: (_prop_name_lower(x), x.property_name_id),
    ):
        if is_price_property(p):
            continue
        name = p.property_name.name if p.property_name else f"#{p.property_name_id}"
        val = (p.value or "").strip()
        if not val:
            continue
        if len(val) > 48:
            val = val[:45] + "…"
        line = f"{name}: {val}"
        # Иначе по алфавиту часто попадают только Аккумулятор/Вес/Видеокарта — ОЗУ не видно при фильтре по RAM
        prio = 0 if _is_ram_property_name(name) else 1
        rows.append((prio, _prop_name_lower(p), p.property_name_id, line))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return [r[3] for r in rows[:limit]]


def get_offer_color_display(offer: ProductOffer) -> Optional[str]:
    """
    Возвращает цвет товара по свойствам (например, "Цвет", "Цвет корпуса", "Color").
    """
    for p in sorted(
        offer.properties or [],
        key=lambda x: (_prop_name_lower(x), x.property_name_id),
    ):
        pname = _prop_name_lower(p)
        if not pname:
            continue
        if not any(h in pname for h in _COLOR_NAME_HINTS):
            continue
        val = (p.value or "").strip()
        if val:
            return val
    # Если цвет в базе не задан — подставляем детерминированную пару цветов.
    idx = int(getattr(offer, "offer_id", 0) or 0) % len(_DEFAULT_COLOR_PAIRS)
    return _DEFAULT_COLOR_PAIRS[idx]


def offer_card_initials(offer: ProductOffer) -> str:
    """Две буквы для плейсхолдера изображения."""
    title = (offer.short_name or f"#{offer.offer_id}")[:2]
    return title.strip().upper() or "?"


def format_price_for_display(raw: Optional[str]) -> str:
    if not raw:
        return ""
    # если уже похоже на число с валютой — как есть
    return raw.strip()


_NUM_TOKEN = re.compile(r"[\d\s]+(?:[.,]\d+)?")


def parse_price_rub_numeric(raw: Optional[str]) -> Optional[float]:
    """Пытается извлечь число из строки цены (для суммы в корзине)."""
    if not raw:
        return None
    for m in _NUM_TOKEN.finditer(raw):
        s = m.group().replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            v = float(s)
            if v >= 0:
                return v
        except ValueError:
            continue
    return None


def get_offer_price_rub(offer: ProductOffer) -> Optional[float]:
    col = _price_rub_from_column(offer)
    if col is not None:
        return col
    return parse_price_rub_numeric(get_offer_price_display(offer))
