"""
Фасеты по текущей поисковой выдаче (дизъюнктивные счётчики по осям + TTL-кэш).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models import Category, Company, ProductOffer, ProductProperty, PropertyName
from search_smart import (
    _is_ram_property_name,
    _parse_ram_gb_from_property_value,
    filter_offer_ids_by_ram_gb,
)

# --- Константы из ТЗ ---
MIN_FACET_COUNT = 2
FACET_TTL_SEC = 60.0
_FACET_CACHE_MAX = 128
_facet_cache: Dict[Tuple[object, ...], Tuple[float, dict]] = {}

DIAGONAL_BUCKETS: Tuple[Tuple[str, str, Optional[float], Optional[float]], ...] = (
    ("small", 'до 13"', None, 13.0),
    ("medium", '13" – 15"', 13.0, 15.0),
    ("large", 'от 15"', 15.0, None),
)

# Ключ в URL / внутренний идентификатор (ГБ), подпись
SSD_FACET_DEFS: Tuple[Tuple[int, str], ...] = (
    (256, "256 ГБ"),
    (512, "512 ГБ"),
    (1024, "1 ТБ"),
    (2048, "2 ТБ"),
)

OS_FACET_DEFS: Tuple[Tuple[str, str], ...] = (
    ("windows", "Windows"),
    ("macos", "macOS"),
    ("android", "Android"),
    ("none", "Без ОС"),
)

# Популярность бренда (выше — выше в списке при равных условиях; «Noname» ниже Apple при большем count)
BRAND_POPULARITY_TIER: Dict[str, int] = {
    "Apple": 100,
    "Samsung": 98,
    "Google": 94,
    "Sony": 92,
    "Xiaomi": 90,
    "Huawei": 88,
    "OnePlus": 86,
    "OPPO": 84,
    "Vivo": 83,
    "Realme": 82,
    "Motorola": 80,
    "LG": 79,
    "Dell": 78,
    "HP": 77,
    "Lenovo": 76,
    "ASUS": 75,
    "Acer": 74,
    "MSI": 73,
    "Microsoft": 72,
    "Intel": 70,
    "AMD": 69,
    "Bose": 68,
    "Canon": 65,
    "Nikon": 65,
    "Noname": 5,
}


def _company_sort_key(row: dict) -> Tuple[int, int, str]:
    tier = BRAND_POPULARITY_TIER.get(row["name"], 50)
    return (-tier, -int(row["count"]), row["name"].lower())


def _facet_rows_min_count(rows: List[dict]) -> List[dict]:
    return [r for r in rows if int(r.get("count", 0)) >= MIN_FACET_COUNT]


@dataclass(frozen=True)
class FacetPriceBucketDyn:
    key: str
    label: str
    price_min: Optional[float]
    price_max: Optional[float]


def _fmt_rub(v: float) -> str:
    return f"{int(round(v)):,}".replace(",", " ")


def build_dynamic_price_buckets(prices: Sequence[float]) -> List[FacetPriceBucketDyn]:
    """Квантили 25/50/75 по ценам выдачи → 4 интервала."""
    arr = np.array([float(p) for p in prices if p is not None and float(p) >= 0], dtype=np.float64)
    if arr.size == 0:
        return []
    if arr.size < 4:
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mn >= mx:
            return [
                FacetPriceBucketDyn("all", f"до {_fmt_rub(mx)} ₽", None, mx),
            ]
        mid = (mn + mx) / 2.0
        return [
            FacetPriceBucketDyn("lo", f"до {_fmt_rub(mid)} ₽", None, mid),
            FacetPriceBucketDyn("hi", f"от {_fmt_rub(mid)} ₽", mid, None),
        ]
    q25, q50, q75 = (float(x) for x in np.percentile(arr, [25, 50, 75]))
    # слипание границ
    if q25 >= q50:
        q25 = float(np.min(arr))
    if q50 >= q75:
        q75 = float(np.max(arr))
    buckets: List[FacetPriceBucketDyn] = [
        FacetPriceBucketDyn("p0_q25", f"до {_fmt_rub(q25)} ₽", None, q25),
        FacetPriceBucketDyn(
            "q25_q50", f"{_fmt_rub(q25)} – {_fmt_rub(q50)} ₽", q25, q50
        ),
        FacetPriceBucketDyn(
            "q50_q75", f"{_fmt_rub(q50)} – {_fmt_rub(q75)} ₽", q50, q75
        ),
        FacetPriceBucketDyn("q75_plus", f"от {_fmt_rub(q75)} ₽", q75, None),
    ]
    return buckets


def _inch_pattern(val: str) -> Optional[float]:
    if not val:
        return None
    s = val.replace("\u00a0", " ").strip()
    m = re.search(
        r"(\d+[.,]?\d*)\s*(?:дюйм|дюйма|inch|\u2033|\"|''|″)",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        m = re.search(r"^(\d+[.,]?\d*)\s*$", s)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _is_diagonal_property(pname: str) -> bool:
    n = (pname or "").lower()
    if "разреш" in n:
        return False
    return "диагонал" in n or ("экран" in n and "разреш" not in n)


def _offer_screen_bucket(pairs: List[Tuple[str, str]]) -> Optional[str]:
    inches: List[float] = []
    for pname, val in pairs:
        if not _is_diagonal_property(pname):
            continue
        inch = _inch_pattern(val or "")
        if inch is not None and 0 < inch < 200:
            inches.append(inch)
    if not inches:
        return None
    x = max(inches)
    if x <= 13.0:
        return "small"
    if x <= 15.0:
        return "medium"
    return "large"


def _storage_gb_from_value(val: str) -> Optional[float]:
    if not val:
        return None
    s = (val or "").lower().replace("\u00a0", " ")
    tb = re.search(r"(\d+(?:[.,]\d+)?)\s*(тб|tb|терабайт)", s)
    if tb:
        try:
            return float(tb.group(1).replace(",", ".")) * 1024.0
        except ValueError:
            return None
    gb = re.search(r"(\d+(?:[.,]\d+)?)\s*(гб|gb|гиг)", s)
    if gb:
        try:
            return float(gb.group(1).replace(",", "."))
        except ValueError:
            return None
    return None


def _is_storage_property(pname: str) -> bool:
    n = (pname or "").lower()
    if _is_ram_property_name(pname):
        return False
    if "видеопам" in n or "vram" in n:
        return False
    return any(
        k in n
        for k in (
            "накопит",
            "ssd",
            "встроенн",
            "хранили",
            "storage",
            "емкост",
        )
    )


def _classify_ssd_bucket_gb(gb_val: float) -> Optional[int]:
    if gb_val < 384:
        return 256
    if gb_val < 896:
        return 512
    if gb_val < 1792:
        return 1024
    return 2048


def _offer_ssd_bucket_key(pairs: List[Tuple[str, str]]) -> Optional[int]:
    best: Optional[float] = None
    for pname, val in pairs:
        if not _is_storage_property(pname):
            continue
        g = _storage_gb_from_value(val or "")
        if g is None:
            continue
        if best is None or g > best:
            best = g
    if best is None:
        return None
    return _classify_ssd_bucket_gb(best)


def _offer_os_family(pairs: List[Tuple[str, str]]) -> Optional[str]:
    for pname, val in pairs:
        pl = (pname or "").lower()
        if not any(k in pl for k in ("ос", "операцион", "os", "system", "платформ")):
            continue
        v = (val or "").lower()
        if "без ос" in v or "no os" in v or "без операцион" in v:
            return "none"
        if "windows" in v:
            return "windows"
        if "macos" in v or "mac os" in v:
            return "macos"
        if "android" in v:
            return "android"
        if "ipad" in v or "ios" in v:
            return "macos"
    return None


def _load_offer_props(session: Session, offer_ids: Sequence[int]) -> Dict[int, List[Tuple[str, str]]]:
    if not offer_ids:
        return {}
    ids = list(map(int, offer_ids))
    stmt = (
        select(ProductOffer.offer_id, ProductProperty.value, PropertyName.name)
        .join(ProductProperty, ProductProperty.offer_id == ProductOffer.offer_id)
        .join(PropertyName, PropertyName.property_name_id == ProductProperty.property_name_id)
        .where(ProductOffer.offer_id.in_(ids))
    )
    by_offer: Dict[int, List[Tuple[str, str]]] = {}
    for oid, val, pname in session.execute(stmt).all():
        by_offer.setdefault(int(oid), []).append((pname or "", val or ""))
    return by_offer


def filter_offer_ids_by_screen_bucket(
    session: Session, offer_ids: Sequence[int], bucket: str
) -> List[int]:
    b = (bucket or "").strip().lower()
    allowed = {x[0] for x in DIAGONAL_BUCKETS}
    if b not in allowed or not offer_ids:
        return list(offer_ids)
    props = _load_offer_props(session, offer_ids)
    out: List[int] = []
    seen: set[int] = set()
    for oid in offer_ids:
        oi = int(oid)
        if oi in seen:
            continue
        seen.add(oi)
        got = _offer_screen_bucket(props.get(oi) or [])
        if got == b:
            out.append(oi)
    return out


def filter_offer_ids_by_ssd_bucket(
    session: Session, offer_ids: Sequence[int], ssd_gb: int
) -> List[int]:
    if ssd_gb not in {x[0] for x in SSD_FACET_DEFS} or not offer_ids:
        return list(offer_ids)
    props = _load_offer_props(session, offer_ids)
    out: List[int] = []
    seen: set[int] = set()
    for oid in offer_ids:
        oi = int(oid)
        if oi in seen:
            continue
        seen.add(oi)
        if _offer_ssd_bucket_key(props.get(oi) or []) == ssd_gb:
            out.append(oi)
    return out


def filter_offer_ids_by_os_family(
    session: Session, offer_ids: Sequence[int], family: str
) -> List[int]:
    f = (family or "").strip().lower()
    allowed = {x[0] for x in OS_FACET_DEFS}
    if f not in allowed or not offer_ids:
        return list(offer_ids)
    props = _load_offer_props(session, offer_ids)
    out: List[int] = []
    seen: set[int] = set()
    for oid in offer_ids:
        oi = int(oid)
        if oi in seen:
            continue
        seen.add(oi)
        if _offer_os_family(props.get(oi) or []) == f:
            out.append(oi)
    return out


def _normalize_gpu_label(raw: str) -> Optional[str]:
    s = (raw or "").lower().replace("\u00a0", " ")
    patterns = (
        r"(rtx\s*\d{3,4}(?:\s*ti)?)",
        r"(gtx\s*\d{3,4}(?:\s*ti)?)",
        r"(rx\s*\d{3,4}(?:\s*xt)?)",
        r"(arc\s*[a-z]?\d{3})",
    )
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return re.sub(r"\s+", " ", m.group(1).upper()).strip()
    if "iris xe" in s:
        return "Intel Iris Xe"
    if "uhd" in s:
        return "Intel UHD"
    if "radeon" in s:
        return "Radeon"
    if "geforce" in s:
        return "GeForce"
    return None


def _offer_gpu_label(pairs: List[Tuple[str, str]]) -> Optional[str]:
    labels: List[str] = []
    for pname, val in pairs:
        pl = (pname or "").lower()
        if any(k in pl for k in ("видеокарт", "gpu", "graphics", "графическ", "видеочип")):
            lab = _normalize_gpu_label(val or "")
            if lab:
                labels.append(lab)
    return labels[0] if labels else None


def filter_offer_ids_by_gpu_value(
    session: Session, offer_ids: Sequence[int], gpu_value: str
) -> List[int]:
    v = (gpu_value or "").strip().lower()
    if not v or not offer_ids:
        return list(offer_ids)
    props = _load_offer_props(session, offer_ids)
    out: List[int] = []
    seen: set[int] = set()
    for oid in offer_ids:
        oi = int(oid)
        if oi in seen:
            continue
        seen.add(oi)
        got = _offer_gpu_label(props.get(oi) or [])
        if got and got.lower() == v:
            out.append(oi)
    return out


def _apply_cross_facets(
    session: Session,
    ids: List[int],
    *,
    ram_gb_filter: Optional[int],
    screen_bucket: Optional[str],
    ssd_gb_filter: Optional[int],
    os_family: Optional[str],
    gpu_filter: Optional[str],
    skip_ram: bool = False,
    skip_screen: bool = False,
    skip_ssd: bool = False,
    skip_os: bool = False,
    skip_gpu: bool = False,
) -> List[int]:
    out = list(ids)
    if ram_gb_filter is not None and not skip_ram:
        out = filter_offer_ids_by_ram_gb(session, out, ram_gb_filter)
    if screen_bucket and not skip_screen:
        out = filter_offer_ids_by_screen_bucket(session, out, screen_bucket)
    if ssd_gb_filter is not None and not skip_ssd:
        out = filter_offer_ids_by_ssd_bucket(session, out, ssd_gb_filter)
    if os_family and not skip_os:
        out = filter_offer_ids_by_os_family(session, out, os_family)
    if gpu_filter and not skip_gpu:
        out = filter_offer_ids_by_gpu_value(session, out, gpu_filter)
    return out


def _prune_facet_cache(now: float) -> None:
    if len(_facet_cache) <= _FACET_CACHE_MAX:
        return
    # удаляем самые старые половину при переполнении
    items = sorted(_facet_cache.items(), key=lambda kv: kv[1][0])
    for k, _ in items[: len(items) // 2]:
        del _facet_cache[k]


def compute_search_facets(
    db: Session,
    ids_soft: Sequence[int],
    *,
    company_id: Optional[int],
    category_id: Optional[int],
    price_min: Optional[float],
    price_max: Optional[float],
    filter_offer_ids_fn: Callable[..., List[int]],
    ram_gb_filter: Optional[int] = None,
    screen_bucket: Optional[str] = None,
    ssd_gb_filter: Optional[int] = None,
    os_family: Optional[str] = None,
    gpu_filter: Optional[str] = None,
) -> dict:
    """
    ram_gb_filter / screen_bucket / ssd_gb_filter / os_family — активные фильтры из URL
    (дизъюнктивные счётчики по своей оси).
    """
    key = (
        tuple(int(x) for x in ids_soft),
        company_id,
        category_id,
        price_min,
        price_max,
        ram_gb_filter,
        screen_bucket,
        ssd_gb_filter,
        os_family,
        gpu_filter,
    )
    now = time.monotonic()
    hit = _facet_cache.get(key)
    if hit is not None and now - hit[0] < FACET_TTL_SEC:
        return hit[1]

    empty = {
        "companies": [],
        "categories": [],
        "price_buckets": [],
        "ram_gb": [],
        "ram_all_count": 0,
        "screen_buckets": [],
        "screen_all_count": 0,
        "ssd_buckets": [],
        "ssd_all_count": 0,
        "os_families": [],
        "os_all_count": 0,
        "gpu_values": [],
        "gpu_all_count": 0,
    }
    if not ids_soft:
        _facet_cache[key] = (now, empty)
        _prune_facet_cache(now)
        return empty

    def filt(
        ids: Sequence[int],
        comp: Optional[int],
        cat: Optional[int],
        pmin: Optional[float],
        pmax: Optional[float],
    ) -> List[int]:
        return filter_offer_ids_fn(db, ids, comp, cat, pmin, pmax)

    data = _compute_search_facets_uncached(
        db,
        ids_soft,
        company_id=company_id,
        category_id=category_id,
        price_min=price_min,
        price_max=price_max,
        filt=filt,
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
    )
    _facet_cache[key] = (now, data)
    _prune_facet_cache(now)
    return data


def _compute_search_facets_uncached(
    db: Session,
    ids_soft: Sequence[int],
    *,
    company_id: Optional[int],
    category_id: Optional[int],
    price_min: Optional[float],
    price_max: Optional[float],
    filt: Callable[..., List[int]],
    ram_gb_filter: Optional[int],
    screen_bucket: Optional[str],
    ssd_gb_filter: Optional[int],
    os_family: Optional[str],
    gpu_filter: Optional[str],
) -> dict:
    cat = category_id if category_id and category_id > 0 else None
    comp = company_id if company_id and company_id > 0 else None

    # --- Бренд (без фильтра по компании) + остальные оси
    ids_brand = filt(ids_soft, None, cat, price_min, price_max)
    ids_brand = _apply_cross_facets(
        db,
        ids_brand,
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
    )
    companies: List[dict] = []
    if ids_brand:
        stmt_c = (
            select(ProductOffer.company_id, func.count())
            .where(ProductOffer.offer_id.in_(list(map(int, ids_brand))))
            .group_by(ProductOffer.company_id)
        )
        raw_c = db.execute(stmt_c).all()
        comp_names = {
            int(r.company_id): r.name
            for r in db.execute(select(Company.company_id, Company.name)).all()
        }
        companies = [
            {
                "id": int(cid),
                "name": comp_names.get(int(cid), f"#{cid}"),
                "count": int(cnt),
            }
            for cid, cnt in raw_c
            if cid is not None
        ]
        companies.sort(key=_company_sort_key)
        companies = _facet_rows_min_count(companies)

    # --- Категория
    ids_cat = filt(ids_soft, comp, None, price_min, price_max)
    ids_cat = _apply_cross_facets(
        db,
        ids_cat,
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
    )
    categories: List[dict] = []
    if ids_cat:
        stmt_t = (
            select(ProductOffer.category_id, func.count())
            .where(ProductOffer.offer_id.in_(list(map(int, ids_cat))))
            .group_by(ProductOffer.category_id)
        )
        raw_t = db.execute(stmt_t).all()
        cat_names = {
            int(r.category_id): r.name
            for r in db.execute(select(Category.category_id, Category.name)).all()
        }
        for cid, cnt in raw_t:
            if cid is None:
                continue
            categories.append(
                {
                    "id": int(cid),
                    "name": cat_names.get(int(cid), f"#{cid}"),
                    "count": int(cnt),
                }
            )
        categories.sort(key=lambda x: (-x["count"], x["name"].lower()))
        categories = _facet_rows_min_count(categories)

    # --- Цена (квантили по выдаче без ценового фильтра URL)
    ids_price = filt(ids_soft, comp, cat, None, None)
    ids_price = _apply_cross_facets(
        db,
        ids_price,
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
    )
    price_buckets: List[dict] = []
    if ids_price:
        stmt_p = select(ProductOffer.offer_id, ProductOffer.price_rub).where(
            ProductOffer.offer_id.in_(list(map(int, ids_price))),
            ProductOffer.price_rub.isnot(None),
        )
        rows_p = db.execute(stmt_p).all()
        prices = [float(pr) for _, pr in rows_p if pr is not None]
        dyn = build_dynamic_price_buckets(prices)
        for b in dyn:
            n = 0
            for _, pr in rows_p:
                if pr is None:
                    continue
                v = float(pr)
                # полуоткрытые интервалы: (price_min, price_max], первый — (−∞, q25]
                if b.price_min is not None and v <= b.price_min:
                    continue
                if b.price_max is not None and v > b.price_max:
                    continue
                n += 1
            if n >= MIN_FACET_COUNT:
                price_buckets.append(
                    {
                        "key": b.key,
                        "label": b.label,
                        "price_min": b.price_min,
                        "price_max": b.price_max,
                        "count": n,
                    }
                )

    # --- RAM
    ids_ram_basis = filt(ids_soft, comp, cat, price_min, price_max)
    ids_ram_basis = _apply_cross_facets(
        db,
        ids_ram_basis,
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
        skip_ram=True,
    )
    ram_counts: Dict[int, int] = {}
    if ids_ram_basis:
        stmt_r = (
            select(ProductOffer.offer_id, ProductProperty.value, PropertyName.name)
            .join(ProductProperty, ProductProperty.offer_id == ProductOffer.offer_id)
            .join(PropertyName, PropertyName.property_name_id == ProductProperty.property_name_id)
            .where(ProductOffer.offer_id.in_(list(map(int, ids_ram_basis))))
        )
        by_offer: Dict[int, List[Tuple[str, str]]] = {}
        for oid, val, pname in db.execute(stmt_r).all():
            by_offer.setdefault(int(oid), []).append((pname or "", val or ""))
        for oid in dict.fromkeys(int(x) for x in ids_ram_basis):
            pairs = by_offer.get(int(oid))
            if not pairs:
                continue
            gb = _ram_gb_for_offer_from_pairs(pairs)
            if gb is None or gb <= 0:
                continue
            ram_counts[gb] = ram_counts.get(gb, 0) + 1
        ram_gb = [
            {"gb": g, "label": f"{g} ГБ", "count": c}
            for g, c in sorted(ram_counts.items(), key=lambda x: x[0])
            if c >= MIN_FACET_COUNT
        ][:12]
    else:
        ram_gb = []

    ram_all_count = (
        len(dict.fromkeys(int(x) for x in ids_ram_basis)) if ids_ram_basis else 0
    )

    # --- Диагональ
    ids_scr = _apply_cross_facets(
        db,
        filt(ids_soft, comp, cat, price_min, price_max),
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
        skip_screen=True,
    )
    screen_buckets: List[dict] = []
    if ids_scr:
        props = _load_offer_props(db, ids_scr)
        sb_counts: Dict[str, int] = {b[0]: 0 for b in DIAGONAL_BUCKETS}
        for oid in dict.fromkeys(int(x) for x in ids_scr):
            b = _offer_screen_bucket(props.get(int(oid)) or [])
            if b:
                sb_counts[b] = sb_counts.get(b, 0) + 1
        for key, label, _lo, _hi in DIAGONAL_BUCKETS:
            c = sb_counts.get(key, 0)
            if c >= MIN_FACET_COUNT:
                screen_buckets.append({"key": key, "label": label, "count": c})
    screen_all_count = len(dict.fromkeys(int(x) for x in ids_scr)) if ids_scr else 0

    # --- SSD
    ids_ssd = _apply_cross_facets(
        db,
        filt(ids_soft, comp, cat, price_min, price_max),
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
        skip_ssd=True,
    )
    ssd_buckets: List[dict] = []
    if ids_ssd:
        props = _load_offer_props(db, ids_ssd)
        ssd_counts: Dict[int, int] = {k: 0 for k, _ in SSD_FACET_DEFS}
        for oid in dict.fromkeys(int(x) for x in ids_ssd):
            k = _offer_ssd_bucket_key(props.get(int(oid)) or [])
            if k is not None:
                ssd_counts[k] = ssd_counts.get(k, 0) + 1
        for gb_key, label in SSD_FACET_DEFS:
            c = ssd_counts.get(gb_key, 0)
            if c >= MIN_FACET_COUNT:
                ssd_buckets.append({"gb": gb_key, "label": label, "count": c})
    ssd_all_count = len(dict.fromkeys(int(x) for x in ids_ssd)) if ids_ssd else 0

    # --- ОС
    ids_os = _apply_cross_facets(
        db,
        filt(ids_soft, comp, cat, price_min, price_max),
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
        skip_os=True,
    )
    os_families: List[dict] = []
    if ids_os:
        props = _load_offer_props(db, ids_os)
        oc: Dict[str, int] = {k: 0 for k, _ in OS_FACET_DEFS}
        for oid in dict.fromkeys(int(x) for x in ids_os):
            f = _offer_os_family(props.get(int(oid)) or [])
            if f:
                oc[f] = oc.get(f, 0) + 1
        for fk, flabel in OS_FACET_DEFS:
            c = oc.get(fk, 0)
            if c >= MIN_FACET_COUNT:
                os_families.append({"key": fk, "label": flabel, "count": c})
    os_all_count = len(dict.fromkeys(int(x) for x in ids_os)) if ids_os else 0

    # --- GPU
    ids_gpu = _apply_cross_facets(
        db,
        filt(ids_soft, comp, cat, price_min, price_max),
        ram_gb_filter=ram_gb_filter,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_gb_filter,
        os_family=os_family,
        gpu_filter=gpu_filter,
        skip_gpu=True,
    )
    gpu_values: List[dict] = []
    if ids_gpu:
        props = _load_offer_props(db, ids_gpu)
        gc: Dict[str, int] = {}
        for oid in dict.fromkeys(int(x) for x in ids_gpu):
            gl = _offer_gpu_label(props.get(int(oid)) or [])
            if gl:
                gc[gl] = gc.get(gl, 0) + 1
        for label, c in sorted(gc.items(), key=lambda x: (-x[1], x[0].lower()))[:12]:
            if c >= MIN_FACET_COUNT:
                gpu_values.append({"value": label, "label": label, "count": c})
    gpu_all_count = len(dict.fromkeys(int(x) for x in ids_gpu)) if ids_gpu else 0

    return {
        "companies": companies,
        "categories": categories,
        "price_buckets": price_buckets,
        "ram_gb": ram_gb,
        "ram_all_count": ram_all_count,
        "screen_buckets": screen_buckets,
        "screen_all_count": screen_all_count,
        "ssd_buckets": ssd_buckets,
        "ssd_all_count": ssd_all_count,
        "os_families": os_families,
        "os_all_count": os_all_count,
        "gpu_values": gpu_values,
        "gpu_all_count": gpu_all_count,
    }


def _ram_gb_for_offer_from_pairs(pairs: List[Tuple[str, str]]) -> Optional[int]:
    best: Optional[float] = None
    for pname, val in pairs:
        if not _is_ram_property_name(pname):
            continue
        gb = _parse_ram_gb_from_property_value(val)
        if gb is None:
            continue
        if best is None or gb > best:
            best = gb
    if best is None:
        return None
    return int(round(best))
