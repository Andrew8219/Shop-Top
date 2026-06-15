"""
Заполняет нормализованные признаки железа в product_offers:
  - hw_gpu, hw_cpu, hw_ram_gb, hw_ssd_gb, hw_has_discrete_gpu

Запуск:
  DATABASE_URL=... python3 backfill_hw_fields.py

Рекомендуется запускать при остановленном uvicorn (чтобы не держать долгие транзакции).
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from db import SessionLocal
from models import Category, ProductOffer, ProductProperty
from search_smart import _parse_ram_gb_from_text, _parse_ssd_gb_from_text, _gpu_score_from_text


def _norm_text(s: str) -> str:
    s = (s or "").lower().replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_gpu_label(text: str) -> Optional[str]:
    s = _norm_text(text)
    m = re.search(r"\brtx\s*(\d{3,4})\b", s)
    if m:
        return f"rtx {m.group(1)}"
    m = re.search(r"\bgtx\s*(\d{3,4})\b", s)
    if m:
        return f"gtx {m.group(1)}"
    m = re.search(r"\brx\s*(\d{3,4})\b", s)
    if m:
        return f"rx {m.group(1)}"
    m = re.search(r"\barc\s*[a-z]?\s*(\d{3})\b", s)
    if m:
        return f"arc a{m.group(1)}"
    if any(k in s for k in ("iris xe", "intel uhd", "uhd graphics", "radeon vega", "vega", "integrated")):
        return "integrated"
    if "geforce" in s:
        return "geforce"
    if "radeon" in s:
        return "radeon"
    return None


def parse_cpu_label(text: str) -> Optional[str]:
    s = _norm_text(text)
    # Apple M
    if re.search(r"\bm[1234]\s*(pro|max|ultra)?\b", s):
        m = re.search(r"\bm([1234])\b", s)
        if m:
            suf = ""
            mm = re.search(r"\bm[1234]\s*(pro|max|ultra)\b", s)
            if mm:
                suf = " " + mm.group(1)
            return f"m{m.group(1)}{suf}".strip()
    # Intel i3/i5/i7/i9 + optional suffix
    m = re.search(r"\bi([3579])[-\\s]?([0-9]{4,5}[a-z]{0,2})\\b", s)
    if m:
        return f"i{m.group(1)}-{m.group(2)}"
    m = re.search(r"\bi([3579])\\b", s)
    if m:
        return f"i{m.group(1)}"
    # Ryzen
    m = re.search(r"\bryzen\\s*([3579])\\s*([0-9]{4})\\b", s)
    if m:
        return f"ryzen {m.group(1)} {m.group(2)}"
    m = re.search(r"\bryzen\\s*([3579])\\b", s)
    if m:
        return f"ryzen {m.group(1)}"
    return None


def _offer_hw_text(offer: ProductOffer) -> str:
    chunks = [offer.short_name or ""]
    for p in offer.properties or []:
        pname = (p.property_name.name if p.property_name else "") or ""
        pval = p.value or ""
        if not pname or not pval:
            continue
        pl = pname.lower()
        if any(k in pl for k in ("видеокарт", "gpu", "graphics", "процессор", "cpu", "оператив", "озу", "ram", "накопит", "ssd")):
            chunks.append(f"{pname}: {pval}")
    return " | ".join([c for c in chunks if c]).strip()


def compute_hw_fields(offer: ProductOffer) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int], Optional[bool]]:
    hw = _offer_hw_text(offer)
    gpu_label = parse_gpu_label(hw)
    cpu_label = parse_cpu_label(hw)
    ram_gb = _parse_ram_gb_from_text(hw)
    ssd_gb = _parse_ssd_gb_from_text(hw)
    gpu_score = _gpu_score_from_text(hw)
    has_discrete = None
    if gpu_label is not None:
        has_discrete = bool(gpu_score and gpu_score > 10 and gpu_label not in ("integrated",))
    return gpu_label, cpu_label, ram_gb, ssd_gb, has_discrete


def main() -> None:
    _ = os.getenv("DATABASE_URL", "").strip()
    db = SessionLocal()
    try:
        laptop_cat_id = db.execute(
            select(Category.category_id).where(Category.name.ilike("%ноутб%"))
        ).scalar()
        q = (
            select(ProductOffer)
            .options(
                selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
                selectinload(ProductOffer.category),
            )
        )
        if laptop_cat_id:
            q = q.where(ProductOffer.category_id == int(laptop_cat_id))

        offers = db.execute(q).scalars().unique().all()
        total = len(offers)
        updated = 0
        for idx, offer in enumerate(offers, 1):
            gpu, cpu, ram, ssd, has_discrete = compute_hw_fields(offer)
            # пишем только если есть что писать
            if gpu is not None:
                offer.hw_gpu = gpu
            if cpu is not None:
                offer.hw_cpu = cpu
            if ram is not None:
                offer.hw_ram_gb = int(ram)
            if ssd is not None:
                offer.hw_ssd_gb = int(ssd)
            if has_discrete is not None:
                offer.hw_has_discrete_gpu = bool(has_discrete)
            updated += 1
            if idx % 200 == 0:
                db.commit()
                print(f"committed {idx}/{total}")
        db.commit()
        print(f"done: updated {updated}/{total}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

