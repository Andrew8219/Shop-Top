"""
Умный поиск: разбор цены/RAM из текста, fuzzy-исправление брендов/категорий, подсказки.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# --- цена из естественного языка (как в main._parse_human_price_filters) ---


def parse_human_price_filters(q: str) -> Tuple[Optional[float], Optional[float]]:
    text_q = (q or "").lower()
    if not text_q:
        return None, None

    def _num_to_value(raw: str, suffix: str) -> Optional[float]:
        s = (raw or "").replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            v = float(s)
        except ValueError:
            return None
        suf = (suffix or "").lower()
        if "млн" in suf or suf == "m":
            v *= 1_000_000
        elif ("к" in suf) or ("k" in suf) or ("тыс" in suf):
            v *= 1_000
        return v if v >= 0 else None

    num = r"(\d+(?:[.,]\d+)?)"
    suffix = r"\s*(к|k|тыс(?:яч)?|тыс\.|млн|m)?"
    max_match = re.search(rf"(?:до|не\s*дороже|дешевле)\s*{num}{suffix}", text_q, flags=re.IGNORECASE)
    min_match = re.search(rf"(?:от|не\s*дешевле|дороже)\s*{num}{suffix}", text_q, flags=re.IGNORECASE)
    # «за 30к», «около 25 тыс» — бюджетный потолок, как «до»
    za_match = re.search(
        rf"(?:за|примерно|около)\s*{num}{suffix}", text_q, flags=re.IGNORECASE
    )
    price_max = _num_to_value(max_match.group(1), max_match.group(2) or "") if max_match else None
    price_min = _num_to_value(min_match.group(1), min_match.group(2) or "") if min_match else None
    za_cap = _num_to_value(za_match.group(1), za_match.group(2) or "") if za_match else None
    # «за 30к» / «около 30 тыс» — ориентир бюджета, не жёсткий «до 30000» (иначе демо-цены 500–99k почти всё отсекают).
    if za_cap is not None:
        za_relaxed = float(round(za_cap * 1.5))
        if price_max is None:
            price_max = za_relaxed
        else:
            price_max = min(price_max, za_cap)
    if price_min is not None and price_max is not None and price_min > price_max:
        return price_max, price_min
    return price_min, price_max


# --- RAM ---

_RAM_PHRASE_RE = re.compile(
    r"(?P<blk>"
    r"(?:\d+(?:[.,]\d+)?)\s*(?:гб|gb|гб\.|гиг)\s*(?:озу|ram|ddr\d*)?"
    r"|(?:озу|ram|ddr\d*)\s*[:]?\s*(?:\d+(?:[.,]\d+)?)\s*(?:гб|gb|гб\.)?"
    r"|(?:\d+(?:[.,]\d+)?)\s*(?:озу|ram)\b"
    r")",
    flags=re.IGNORECASE,
)


_DIAGONAL_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:дюйм(?:а|ов)?|inch|\u2033|''|\")",
    flags=re.IGNORECASE,
)


_SSD_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:гб|gb|тб|tb)\s*ssd\b",
    flags=re.IGNORECASE,
)


def _parse_ram_gb_from_text(text: str) -> Optional[int]:
    t = (text or "").lower()
    for m in _RAM_PHRASE_RE.finditer(t):
        chunk = m.group(0)
        num_m = re.search(r"(\d+(?:[.,]\d+)?)", chunk)
        if not num_m:
            continue
        try:
            v = float(num_m.group(1).replace(",", "."))
        except ValueError:
            continue
        if v <= 0 or v > 512:
            continue
        return int(round(v))
    # «16 гб» без ключевых слов RAM — только если рядом есть маркер памяти
    loose = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:гб|gb)\b(?=[^\n]{0,12}(?:озу|ram|оператив|памят))",
        t,
        flags=re.IGNORECASE,
    )
    if loose:
        try:
            v = float(loose.group(1).replace(",", "."))
            if 0 < v <= 512:
                return int(round(v))
        except ValueError:
            pass
    return None


def _parse_diagonal_from_text(text: str) -> Optional[float]:
    t = (text or "").lower()
    m = _DIAGONAL_RE.search(t)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", "."))
    except ValueError:
        return None
    if 3.0 <= v <= 120.0:
        return v
    return None


def _parse_ssd_gb_from_text(text: str) -> Optional[int]:
    t = (text or "").lower()
    m = _SSD_RE.search(t)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", "."))
    except ValueError:
        return None
    unit_chunk = m.group(0)
    if "тб" in unit_chunk or "tb" in unit_chunk:
        v *= 1024.0
    if v <= 0:
        return None
    # Приводим к фасетным бакетам.
    if v < 384:
        return 256
    if v < 896:
        return 512
    if v < 1792:
        return 1024
    return 2048


def _strip_price_phrases(text: str) -> str:
    t = text
    num = r"(\d+(?:[.,]\d+)?)"
    suffix = r"\s*(к|k|тыс(?:яч)?|тыс\.|млн|m)?"
    for pat in (
        rf"(?:до|не\s*дороже|дешевле)\s*{num}{suffix}",
        rf"(?:от|не\s*дешевле|дороже)\s*{num}{suffix}",
        rf"(?:за|примерно|около)\s*{num}{suffix}",
    ):
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    return t


def _strip_currency_noise(text: str) -> str:
    return re.sub(
        r"\b(?:руб|рублей|рубля|р\.|₽)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )


@dataclass(frozen=True)
class ParsedSearchQuery:
    """Результат разбора запроса для гибридного поиска и SQL-фильтров."""

    clean_q: str
    price_min: Optional[float]
    price_max: Optional[float]
    ram_gb: Optional[int]
    diagonal_inch: Optional[float]
    ssd_gb: Optional[int]


def parse_query_intent(q: str) -> ParsedSearchQuery:
    from search_intent_lexicon import strip_intent_phrases

    raw = strip_intent_phrases((q or "").strip())
    pmin, pmax = parse_human_price_filters(raw)
    ram_gb = _parse_ram_gb_from_text(raw)
    diagonal_inch = _parse_diagonal_from_text(raw)
    ssd_gb = _parse_ssd_gb_from_text(raw)

    t = raw
    t = _RAM_PHRASE_RE.sub(" ", t)
    t = _DIAGONAL_RE.sub(" ", t)
    t = _SSD_RE.sub(" ", t)
    t = _strip_price_phrases(t)
    t = _strip_currency_noise(t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        t = raw
    return ParsedSearchQuery(
        clean_q=t,
        price_min=pmin,
        price_max=pmax,
        ram_gb=ram_gb,
        diagonal_inch=diagonal_inch,
        ssd_gb=ssd_gb,
    )


# --- fuzzy бренды / категории ---

FUZZY_SCORE_CUTOFF = 88

# Латиница / транслит → русское слово (двойной проход _apply_word_aliases подхватит ноутбук→Ноутбуки).
_TRANSLIT_QUERY_ALIASES: Dict[str, str] = {
    "noytbuk": "ноутбук",
    "noutbuk": "ноутбук",
    "laptop": "ноутбук",
    "smartfon": "смартфон",
    "smartphone": "смартфон",
    "planshet": "планшет",
    "tablet": "планшет",
}

# Частые русские написания → как в БД (латиница)
_BRAND_CATEGORY_ALIASES: Dict[str, str] = {
    "самсунг": "Samsung",
    "галакси": "Galaxy",
    "ксяоми": "Xiaomi",
    "сяоми": "Xiaomi",
    "хуавей": "Huawei",
    "эппл": "Apple",
    "асус": "ASUS",
    "асер": "Acer",
    "леново": "Lenovo",
    "нокиа": "Nokia",
    "сони": "Sony",
    "гугл": "Google",
    "ванплас": "OnePlus",
    "оппо": "OPPO",
    "виво": "Vivo",
    "риалми": "Realme",
    "мото": "Motorola",
    "моторола": "Motorola",
    "макбук": "MacBook",
    "макбука": "MacBook",
    "макбуком": "MacBook",
    "макбуки": "MacBook",
    "макбуков": "MacBook",
    "ноутбук": "Ноутбуки",
    "смартфон": "Смартфоны",
    "телефон": "Смартфоны",
    "монитор": "Мониторы",
    "телевизор": "Телевизоры",
}


def _apply_word_aliases(q: str) -> str:
    def once(text: str) -> str:
        def repl_word(m: re.Match) -> str:
            w = m.group(0)
            low = w.lower()
            rep = _TRANSLIT_QUERY_ALIASES.get(low) or _BRAND_CATEGORY_ALIASES.get(low)
            return rep if rep else w

        return re.sub(r"\b\w+\b", repl_word, text, flags=re.UNICODE)

    return once(once(q))


def _flatten_entity_tokens(names: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for name in names:
        s = (name or "").strip()
        if len(s) < 2:
            continue
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
        for part in re.split(r"[^\w]+", s, flags=re.UNICODE):
            p = part.strip()
            if len(p) >= 3 and p.lower() not in seen:
                seen.add(p.lower())
                out.append(p)
    return out


def fuzzy_correct_entities(
    q: str,
    company_names: Sequence[str],
    category_names: Sequence[str],
    *,
    score_cutoff: int = FUZZY_SCORE_CUTOFF,
) -> Tuple[str, bool]:
    """
    Подмена отдельных слов запроса на ближайшие бренды/категории (RapidFuzz).
    Возвращает (исправленная_строка, было_ли_изменение).
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return q, False

    text = (q or "").strip()
    if not text:
        return text, False

    choices = _flatten_entity_tokens(list(company_names) + list(category_names))
    if not choices:
        return text, False

    changed = False
    out_parts: List[str] = []

    for m in re.finditer(r"\S+", text):
        token = m.group(0)
        alnum = re.sub(r"^\W+|\W+$", "", token, flags=re.UNICODE)
        if len(alnum) < 4:
            out_parts.append(token)
            continue
        match = process.extractOne(
            alnum,
            choices,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
        )
        if match:
            best, score, _ = match
            if score >= score_cutoff and best.lower() != alnum.lower():
                # сохраняем пунктуацию на концах токена
                prefix = token[: token.index(alnum)] if alnum in token else ""
                suffix = token[token.index(alnum) + len(alnum) :] if alnum in token else ""
                out_parts.append(prefix + best + suffix)
                changed = True
            else:
                out_parts.append(token)
        else:
            out_parts.append(token)

    corrected = re.sub(r"\s+", " ", " ".join(out_parts)).strip()
    return corrected, changed


# --- расширение запроса (синонимы для первичного retrieval) ---

_QUERY_SYNONYMS: Dict[str, str] = {
    "телефон": "смартфон мобильный сотовый",
    "смартфон": "телефон мобильный",
    "мобильный": "телефон смартфон",
    "тихий": "бесшумный малошумный низкий уровень шума quiet",
    "бесшумный": "тихий малошумный",
    "ноутбук": "лэптоп портативный компьютер",
    "лэптоп": "ноутбук портативный",
    "монитор": "дисплей экран",
    "телевизор": "тв tv экран",
    "планшет": "tablet pad",
    "наушники": "гарнитура аудио",
    "камера": "фотоаппарат съёмка",
    "мышь": "mouse компьютерная мышь",
    "клавиатура": "keyboard клавиши",
    "принтер": "печать мфу",
    "ssd": "накопитель твердотельный",
    "hdd": "жёсткий диск накопитель",
    "игровой": "gaming game для игр",
    "мощный": "производительный быстрый топовый",
    "дешёвый": "недорогой бюджетный экономичный",
    "дёшевый": "недорогой бюджетный экономичный",
    "пылесос": "пылесосы робот вертикальный беспроводной всасывание мощность Dyson Xiaomi",
    "крутой": "мощный производительный топовый",
    "крутая": "мощный производительный",
    "сильный": "мощный производительный высокая мощность",
    "сильная": "мощный производительный",
}

from search_intent_lexicon import INTENT_QUERY_SYNONYMS as _INTENT_QUERY_SYNONYMS

for _k, _v in _INTENT_QUERY_SYNONYMS.items():
    _QUERY_SYNONYMS.setdefault(_k, _v)


def expand_search_query(q: str, *, max_extra_tokens: int = 20) -> str:
    """
    Добавляет к запросу синонимы по словарю (для bi-encoder / FTS).
    Не меняет исходные слова; мягкий фильтр по токенам использует строку без расширения.
    """
    if os.getenv("SEARCH_QUERY_EXPANSION", "1").strip().lower() in ("0", "false", "no", "off"):
        return (q or "").strip()
    text = (q or "").strip()
    if not text:
        return text

    words = re.findall(r"\w+", text, flags=re.UNICODE)
    seen = {w.casefold() for w in words}
    extra: List[str] = []
    for w in words:
        low = w.casefold()
        pack = _QUERY_SYNONYMS.get(low)
        if not pack:
            continue
        for tok in pack.split():
            tl = tok.casefold()
            if tl not in seen:
                seen.add(tl)
                extra.append(tok)
            if len(extra) >= max_extra_tokens:
                break
        if len(extra) >= max_extra_tokens:
            break
    if not extra:
        return text
    return f"{text} {' '.join(extra)}"


def prepare_search_query(
    q: str, company_names: Sequence[str], category_names: Sequence[str]
) -> Tuple[str, ParsedSearchQuery, bool, str, bool, str]:
    """
    Алиасы → parse intent → fuzzy по clean_q.
    Возвращает (search_q, parsed_intent, refined, search_q_retrieval, fuzzy_changed, fuzzy_display_q).
    fuzzy_display_q — строка после fuzzy (до compact), для подписи «Исправлен запрос»; пустая, если fuzzy не срабатывал.
    search_q — компактная строка для UI и мягкого токен-фильтра (без расширения синонимами).
    search_q_retrieval — после expand_search_query (синонимы) и compact; для FTS и эмбеддингов.
    """
    from search_intent_lexicon import strip_intent_phrases

    raw = strip_intent_phrases((q or "").strip())
    aliased = _apply_word_aliases(raw)
    alias_changed = aliased.casefold() != raw.casefold()
    parsed = parse_query_intent(aliased)
    corrected, fuzzy_changed = fuzzy_correct_entities(parsed.clean_q, company_names, category_names)
    search_q_raw = corrected.strip() or parsed.clean_q.strip() or raw

    from search_index import compact_query_for_retrieval

    compacted = compact_query_for_retrieval(search_q_raw)
    search_q = compacted.strip() if compacted.strip() else search_q_raw

    expanded = expand_search_query(search_q_raw)
    retrieval_compacted = compact_query_for_retrieval(expanded)
    search_q_retrieval = (
        retrieval_compacted.strip() if retrieval_compacted.strip() else search_q
    )

    refined = alias_changed or fuzzy_changed
    fuzzy_display_q = search_q_raw if fuzzy_changed else ""
    return search_q, parsed, refined, search_q_retrieval, fuzzy_changed, fuzzy_display_q


# --- подсказки (кэш) ---

_SUGGEST_TS = 0.0
_SUGGEST_LIST: List[str] = []
_SUGGEST_TTL_SEC = 300.0


def load_suggest_strings(session: "Session", *, max_titles: int = 4000) -> List[str]:
    from sqlalchemy import func, select

    from models import Category, Company, ProductOffer

    companies = [r[0] for r in session.execute(select(Company.name)).all() if r[0]]
    categories = [r[0] for r in session.execute(select(Category.name)).all() if r[0]]
    titles = session.execute(
        select(ProductOffer.short_name)
        .where(ProductOffer.short_name.isnot(None))
        .where(func.length(ProductOffer.short_name) > 2)
        .limit(max_titles)
    ).scalars().all()

    seen: Set[str] = set()
    out: List[str] = []
    for s in companies + categories + list(titles):
        t = (s or "").strip()
        if not t or len(t) > 200:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    out.sort(key=lambda x: (-len(x), x.casefold()))
    return out


def get_cached_suggest_strings(session: "Session") -> List[str]:
    global _SUGGEST_TS, _SUGGEST_LIST
    now = time.monotonic()
    if now - _SUGGEST_TS < _SUGGEST_TTL_SEC and _SUGGEST_LIST:
        return _SUGGEST_LIST
    _SUGGEST_LIST = load_suggest_strings(session)
    _SUGGEST_TS = now
    return _SUGGEST_LIST


def suggest_completions(prefix: str, choices: Sequence[str], *, limit: int = 10) -> List[str]:
    """
    Возвращает подсказки для автодополнения.
    Приоритет: точный префикс слова → подстрока → fuzzy (RapidFuzz WRatio ≥ 72).
    """
    p = (prefix or "").strip()
    if len(p) < 2:
        return []
    ch = list(choices)
    if not ch:
        return []

    p_low = p.casefold()
    p_words = p_low.split()

    # Tier-1: начинается с prefix (или первое слово совпадает)
    tier1: List[str] = []
    # Tier-2: содержит prefix как подстроку
    tier2: List[str] = []
    seen: Set[str] = set()

    for c in ch:
        c_low = c.casefold()
        key = c_low
        if key in seen:
            continue
        if c_low.startswith(p_low):
            seen.add(key)
            tier1.append(c)
        elif any(w.startswith(p_words[0]) for w in c_low.split()):
            seen.add(key)
            tier1.append(c)
        elif p_low in c_low:
            seen.add(key)
            tier2.append(c)

    # Tier-3: fuzzy (rapidfuzz), skip already found
    tier3: List[str] = []
    try:
        from rapidfuzz import fuzz, process

        pool = [c for c in ch if c.casefold() not in seen]
        matches = process.extract(p, pool, scorer=fuzz.WRatio, limit=limit * 4)
        for name, score, _ in matches:
            if score < 72:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            tier3.append(name)
    except ImportError:
        pass

    result = tier1 + tier2 + tier3
    return result[:limit]


# --- фильтр по RAM в SQL/Python ---

def _is_ram_property_name(name: str) -> bool:
    n = (name or "").lower()
    if "видеопам" in n or "vram" in n:
        return False
    if "оператив" in n or "озу" in n:
        return True
    if "ram" in n and "gram" not in n:
        return True
    return False


def _parse_ram_gb_from_property_value(val: str) -> Optional[float]:
    if not val:
        return None
    s = val.lower().replace("\u00a0", " ")
    # Явные МБ
    mb = re.search(r"(\d+(?:[.,]\d+)?)\s*(мб|mb|мегабайт)", s)
    if mb:
        try:
            return float(mb.group(1).replace(",", ".")) / 1024.0
        except ValueError:
            return None
    # ГБ / Gb
    gb = re.search(r"(\d+(?:[.,]\d+)?)\s*(гб|gb|гиг)", s)
    if gb:
        try:
            return float(gb.group(1).replace(",", "."))
        except ValueError:
            return None
    # Только число в начале «16» / «16.0»
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*$", s)
    if m:
        try:
            v = float(m.group(1).replace(",", "."))
            return v if 0 < v <= 512 else None
        except ValueError:
            return None
    return None


def filter_offer_ids_by_ram_gb(
    session: "Session",
    offer_ids: Sequence[int],
    ram_gb: int,
    *,
    tolerance: float = 1.0,
) -> List[int]:
    """
    Оставляет офферы, у которых в характеристиках указан объём RAM ≈ ram_gb ГБ.
    """
    from sqlalchemy import select

    from models import ProductOffer, ProductProperty, PropertyName

    if not offer_ids or ram_gb <= 0:
        return list(offer_ids)

    ids = list(map(int, offer_ids))
    stmt = (
        select(ProductOffer.offer_id, ProductProperty.value, PropertyName.name)
        .join(ProductProperty, ProductProperty.offer_id == ProductOffer.offer_id)
        .join(PropertyName, PropertyName.property_name_id == ProductProperty.property_name_id)
        .where(ProductOffer.offer_id.in_(ids))
    )
    rows = session.execute(stmt).all()

    by_offer: Dict[int, List[Tuple[str, str]]] = {}
    for oid, val, pname in rows:
        if not _is_ram_property_name(pname or ""):
            continue
        by_offer.setdefault(int(oid), []).append((pname or "", val or ""))

    # Как в фасете ОЗУ: берём максимум по строкам с именем «оперативная / ОЗУ / RAM»,
    # а не «любая строка в допуске» — иначе счётчик на чипе и выдача расходятся.
    out: List[int] = []
    seen: Set[int] = set()
    for oid in ids:
        if oid in seen:
            continue
        pairs = by_offer.get(oid)
        if not pairs:
            continue
        mx: Optional[float] = None
        for _, val in pairs:
            gb = _parse_ram_gb_from_property_value(val)
            if gb is not None:
                mx = gb if mx is None else max(mx, gb)
        if mx is not None and abs(mx - float(ram_gb)) <= tolerance:
            seen.add(oid)
            out.append(oid)
    return out


def _is_gpu_property_name(name: str) -> bool:
    n = (name or "").lower()
    return any(
        k in n
        for k in ("видеокарт", "gpu", "graphics", "графическ", "видеочип", "video card")
    )


def filter_offer_ids_by_videocard(
    session: "Session",
    offer_ids: Sequence[int],
    videocard_query: str,
) -> List[int]:
    """
    Фильтр по параметру видеокарты (например, "4070", "rtx 4070", "radeon").
    Ищет совпадения в short_name и в GPU-свойствах карточки.
    """
    from sqlalchemy import select

    from models import ProductOffer, ProductProperty, PropertyName

    q = (videocard_query or "").strip().lower()
    if not offer_ids or not q:
        return list(offer_ids)
    tokens = [t for t in re.findall(r"\w+", q, flags=re.UNICODE) if len(t) >= 2 or t.isdigit()]
    if not tokens:
        return list(offer_ids)

    ids = list(map(int, offer_ids))
    rows = session.execute(
        select(ProductOffer.offer_id, ProductOffer.short_name, PropertyName.name, ProductProperty.value)
        .join(ProductProperty, ProductProperty.offer_id == ProductOffer.offer_id, isouter=True)
        .join(PropertyName, PropertyName.property_name_id == ProductProperty.property_name_id, isouter=True)
        .where(ProductOffer.offer_id.in_(ids))
    ).all()

    by_offer: Dict[int, List[Tuple[str, str, str]]] = {}
    for oid, short_name, pname, pval in rows:
        by_offer.setdefault(int(oid), []).append((short_name or "", pname or "", pval or ""))

    out: List[int] = []
    seen: Set[int] = set()
    for oid in ids:
        if oid in seen:
            continue
        seen.add(oid)
        triples = by_offer.get(oid) or []
        if not triples:
            continue
        # Всегда учитываем short_name + только профильные GPU-свойства.
        chunks: List[str] = []
        first_short = (triples[0][0] or "").strip()
        if first_short:
            chunks.append(first_short)
        for _sn, pname, pval in triples:
            if _is_gpu_property_name(pname):
                chunks.append(f"{pname}: {pval}")
        hay = " | ".join(chunks).lower()
        if hay and all(tok in hay for tok in tokens):
            out.append(oid)
    return out


def parse_vacuum_suction_watts(offer) -> int:
    """Парсит «Мощность всасывания: N Вт» (или похожее имя свойства)."""
    best = 0
    for p in offer.properties or []:
        pname = (p.property_name.name if p.property_name else "") or ""
        pl = pname.lower()
        if "всасыван" not in pl and "мощност" not in pl:
            continue
        val = p.value or ""
        m = re.search(r"(\d+)\s*вт", val, flags=re.IGNORECASE)
        if m:
            w = int(m.group(1))
            if w > best:
                best = w
    return best


def premium_power_intent(q: str) -> bool:
    """Запрос намекает на топовые/мощные характеристики."""
    t = (q or "").lower()
    if re.search(
        r"(крут[а-я]{0,4}|мощн[а-я]{0,4}|сильн[а-я]{0,5}|топов[а-я]{0,4}|"
        r"навороч[а-я]{0,6}|лучш[а-я]{0,5}|премиум|производительн[а-я]{0,6}|"
        r"максимальн[а-я]{0,6})",
        t,
    ):
        return True
    if re.search(r"(^|[^\w])топ([^\w]|$)", t, flags=re.UNICODE):
        return True
    return False


def vacuum_product_intent(q: str) -> bool:
    return "пылесос" in (q or "").lower()


def prioritize_vacuum_offers_by_suction(
    session: "Session",
    offer_ids: Sequence[int],
    raw_query: str,
) -> List[int]:
    """
    Для запросов вроде «крутой пылесос с мощным всасыванием» — сначала модели
    с большей мощностью всасывания (по полю в карточке), остальные — в исходном порядке.
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from models import ProductOffer, ProductProperty

    if not offer_ids:
        return []
    if not vacuum_product_intent(raw_query) or not premium_power_intent(raw_query):
        return list(map(int, offer_ids))

    ids = list(dict.fromkeys(int(x) for x in offer_ids))
    stmt = (
        select(ProductOffer)
        .options(
            selectinload(ProductOffer.category),
            selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
        )
        .where(ProductOffer.offer_id.in_(ids))
    )
    rows = session.execute(stmt).scalars().all()
    by_id = {int(o.offer_id): o for o in rows}

    vacuum_ids: List[int] = []
    other_ids: List[int] = []
    for oid in ids:
        o = by_id.get(oid)
        if o is None:
            continue
        cname = ((o.category.name if o.category else "") or "").lower()
        if "пылесос" in cname:
            vacuum_ids.append(oid)
        else:
            other_ids.append(oid)

    if not vacuum_ids:
        return list(map(int, offer_ids))

    vacuum_ids.sort(key=lambda i: -parse_vacuum_suction_watts(by_id[i]))
    return vacuum_ids + other_ids


def _parse_battery_mah(offer) -> int:
    best = 0
    for p in offer.properties or []:
        pname = (p.property_name.name if p.property_name else "") or ""
        pl = pname.lower()
        if not any(k in pl for k in ("батар", "аккум", "емкост", "mah", "мач")):
            continue
        v = (p.value or "").lower().replace("\u00a0", " ")
        m = re.search(r"(\d{3,6})\s*(?:ма[чч]|mah)", v, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"^\s*(\d{3,6})\s*$", v)
        if not m:
            continue
        got = int(m.group(1))
        if got > best:
            best = got
    return best


def _parse_display_inch(offer) -> float:
    best = 0.0
    for p in offer.properties or []:
        pname = (p.property_name.name if p.property_name else "") or ""
        pl = pname.lower()
        if "разреш" in pl:
            continue
        if "диагонал" not in pl and "экран" not in pl and "display" not in pl:
            continue
        v = (p.value or "").replace("\u00a0", " ")
        m = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:дюйм(?:а|ов)?|inch|\u2033|\"|''|″)",
            v,
            flags=re.IGNORECASE,
        )
        if not m:
            m = re.search(r"^\s*(\d+(?:[.,]\d+)?)\s*$", v)
        if not m:
            continue
        try:
            inch = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        if 3.0 <= inch <= 120.0 and inch > best:
            best = inch
    return best


def _long_battery_intent(q: str) -> bool:
    t = (q or "").lower()
    return bool(
        re.search(
            r"(батар[а-я]{0,6}|аккум[а-я]{0,6}|долго\s*держ|автономн[а-я]{0,6}|"
            r"долгая\s*работа|длительн[а-я]{0,6}\s*работ)",
            t,
        )
    )


def _big_display_intent(q: str) -> bool:
    t = (q or "").lower()
    return bool(
        re.search(
            r"(больш[а-я]{0,4}\s*экран|больш[а-я]{0,4}\s*диагонал|широк[а-я]{0,5}\s*экран|"
            r"крупн[а-я]{0,5}\s*экран)",
            t,
        )
    )


def _is_phone_or_wearable(offer) -> bool:
    cname = ((offer.category.name if offer.category else "") or "").lower()
    return any(k in cname for k in ("смартфон", "телефон", "планшет", "часы", "браслет"))


def _is_big_screen_category(offer) -> bool:
    cname = ((offer.category.name if offer.category else "") or "").lower()
    return any(k in cname for k in ("телевиз", "монитор", "ноутбук", "планшет"))


def prioritize_by_battery(
    session: "Session",
    offer_ids: Sequence[int],
    raw_query: str,
) -> List[int]:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from models import ProductOffer, ProductProperty

    if not offer_ids or not _long_battery_intent(raw_query):
        return list(map(int, offer_ids))

    ids = list(dict.fromkeys(int(x) for x in offer_ids))
    rows = session.execute(
        select(ProductOffer)
        .options(
            selectinload(ProductOffer.category),
            selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
        )
        .where(ProductOffer.offer_id.in_(ids))
    ).scalars().all()
    by_id = {int(o.offer_id): o for o in rows}
    group_ids: List[int] = []
    other_ids: List[int] = []
    for oid in ids:
        o = by_id.get(oid)
        if o is None:
            continue
        if _is_phone_or_wearable(o):
            group_ids.append(oid)
        else:
            other_ids.append(oid)
    group_ids.sort(key=lambda i: -_parse_battery_mah(by_id[i]))
    return group_ids + other_ids


def prioritize_by_display(
    session: "Session",
    offer_ids: Sequence[int],
    raw_query: str,
) -> List[int]:
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from models import ProductOffer, ProductProperty

    if not offer_ids or not _big_display_intent(raw_query):
        return list(map(int, offer_ids))

    ids = list(dict.fromkeys(int(x) for x in offer_ids))
    rows = session.execute(
        select(ProductOffer)
        .options(
            selectinload(ProductOffer.category),
            selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
        )
        .where(ProductOffer.offer_id.in_(ids))
    ).scalars().all()
    by_id = {int(o.offer_id): o for o in rows}
    group_ids: List[int] = []
    other_ids: List[int] = []
    for oid in ids:
        o = by_id.get(oid)
        if o is None:
            continue
        if _is_big_screen_category(o):
            group_ids.append(oid)
        else:
            other_ids.append(oid)
    group_ids.sort(key=lambda i: -_parse_display_inch(by_id[i]))
    return group_ids + other_ids


def log_search_query(
    session: "Session",
    query: str,
    results_count: int,
    *,
    search_mode: str = "hybrid",
) -> Optional[int]:
    from models import SearchLog

    try:
        row = SearchLog(
            query=(query or "").strip(),
            results_count=max(0, int(results_count)),
            search_mode=(search_mode or "hybrid").strip() or "hybrid",
            clicked_offer_id=None,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return int(row.id)
    except Exception:
        session.rollback()
        return None
