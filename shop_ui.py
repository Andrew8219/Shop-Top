"""
Хелперы для витрины: цена из свойств, краткие атрибуты для карточек.
"""
from __future__ import annotations

import re
from typing import List, Optional

from models import ProductOffer, ProductProperty


_PRICE_NAME_HINTS = ("цен", "price", "стоим", "руб")


def _prop_name_lower(p: ProductProperty) -> str:
    if p.property_name and p.property_name.name:
        return p.property_name.name.strip().lower()
    return ""


def is_price_property(p: ProductProperty) -> bool:
    n = _prop_name_lower(p)
    if not n:
        return False
    return any(h in n for h in _PRICE_NAME_HINTS)


def get_offer_price_display(offer: ProductOffer) -> Optional[str]:
    """
    Возвращает строку цены, если в характеристиках есть поле вроде «Цена» / price.
    """
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
    lines: List[str] = []
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
        lines.append(f"{name}: {val}")
        if len(lines) >= limit:
            break
    return lines


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
    return parse_price_rub_numeric(get_offer_price_display(offer))
