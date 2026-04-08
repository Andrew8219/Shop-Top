# Включает: нормализацию, clustering, fallback, безопасный pipeline

import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Dict, List

from sqlalchemy import select, update, func
from sqlalchemy.orm import Session

from models import PropertyName, ProductProperty, ProductOffer, Category

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    AgglomerativeClustering = None
    TfidfVectorizer = None

LOG = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]

# =========================
# REGEX / CONST
# =========================

NUMERIC_RE = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?)\s*([^\d]+)?\s*$")

UNIT_SYNONYMS = {
    "kg": "кг", "кг.": "кг",
    "g": "г", "гр": "г",
    "mm": "мм", "см": "см", "cm": "см",
    "gb": "ГБ", "мб": "МБ", "mb": "МБ",
    "tb": "ТБ", "ghz": "ГГц", "mhz": "МГц",
    "mah": "мАч", "mp": "МП",
}

_UNIT_PATTERNS = [
    (re.compile(r'(\d[\d.,]*)\s*(gb|гб)', re.I), r'\1 ГБ'),
    (re.compile(r'(\d[\d.,]*)\s*(mb|мб)', re.I), r'\1 МБ'),
    (re.compile(r'(\d[\d.,]*)\s*(ghz|ггц)', re.I), r'\1 ГГц'),
    (re.compile(r'(\d[\d.,]*)\s*(mhz|мгц)', re.I), r'\1 МГц'),
    (re.compile(r'(\d[\d.,]*)\s*(kg|кг)', re.I), r'\1 кг'),
]

# =========================
# NORMALIZATION
# =========================

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_units_regex(s: str) -> str:
    for pattern, replacement in _UNIT_PATTERNS:
        s = pattern.sub(replacement, s)
    return s.strip()


def normalize_basic_value(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = normalize_whitespace(s)
    s = normalize_units_regex(s)
    return s


def parse_numeric(value: str) -> Optional[Tuple[float, Optional[str]]]:
    m = NUMERIC_RE.match(value)
    if not m:
        return None

    num_str, unit = m.groups()

    try:
        num = float(num_str.replace(",", "."))
    except ValueError:
        return None

    if unit:
        u_norm = unit.lower().replace(" ", "").replace("·", "")
        unit = UNIT_SYNONYMS.get(u_norm, unit)

    return num, unit

# =========================
# BOOLEAN
# =========================

BOOL_TRUE = {"да", "true", "yes", "1"}
BOOL_FALSE = {"нет", "false", "no", "0"}


def normalize_bool(v: str) -> str:
    l = v.lower().strip()
    if l in BOOL_TRUE:
        return "Да"
    if l in BOOL_FALSE:
        return "Нет"
    return v

# =========================
# CLUSTERING
# =========================

class TfidfEmbedder:
    def __init__(self):
        if TfidfVectorizer is None:
            raise RuntimeError("sklearn required")
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))

    def fit(self, texts):
        self.vec.fit(texts)

    def transform(self, texts):
        return self.vec.transform(texts).toarray().astype("float32")


def cluster_values(values, counts, embedder, threshold=0.35):
    if len(values) <= 1:
        return {v: v for v in values}

    X = embedder.transform(values)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )

    labels = clustering.fit_predict(X)

    groups = defaultdict(list)
    for value, label in zip(values, labels):  # <-- исправлено
        groups[label].append(value)

    result = {}
    for vals in groups.values():
        best = sorted(vals, key=lambda x: (-counts[x], len(x)))[0]
        for val in vals:
            result[val] = best

    return result

# =========================
# DB PIPELINE
# =========================

def collect(session: Session):
    values = defaultdict(Counter)
    stmt = select(PropertyName.name, ProductProperty.value).join(ProductProperty)

    for name, val in session.execute(stmt):
        if not val:
            continue
        norm = normalize_basic_value(val)
        values[name][norm] += 1

    return values


def build_mapping(session: Session, progress_cb: Optional[ProgressCallback] = None):
    if progress_cb:
        progress_cb("collect", 0, 1)
    values = collect(session)
    if progress_cb:
        progress_cb("collect", 1, 1)

    all_vals = [v for c in values.values() for v in c.keys()]

    embedder = None
    if all_vals and TfidfVectorizer:
        embedder = TfidfEmbedder()
        embedder.fit(all_vals)

    result = {}
    props_list = list(values.items())
    total_props = len(props_list) or 1

    for i, (prop, counter) in enumerate(props_list):
        vals = list(counter.keys())

        if embedder:
            mapping = cluster_values(vals, counter, embedder)
        else:
            mapping = {v: v for v in vals}

        result[prop] = mapping
        if progress_cb:
            progress_cb("mapping", i + 1, total_props)

    return result


def apply_mapping(
    session: Session,
    mapping,
    progress_cb: Optional[ProgressCallback] = None,
):
    name_to_id = {
        name: pid for pid, name in session.execute(
            select(PropertyName.property_name_id, PropertyName.name)
        )
    }

    ops: List[Tuple[int, str, str]] = []
    for prop, m in mapping.items():
        pid = name_to_id.get(prop)
        if not pid:
            continue
        for raw, canon in m.items():
            if raw == canon:
                continue
            ops.append((pid, raw, canon))

    total = 0
    n_ops = len(ops) or 1
    for idx, (pid, raw, canon) in enumerate(ops, start=1):
        stmt = (
            update(ProductProperty)
            .where(ProductProperty.property_name_id == pid)
            .where(ProductProperty.value == raw)
            .values(value=canon)
        )
        result = session.execute(stmt)
        rows_affected = result.rowcount if hasattr(result, "rowcount") else 0
        total += rows_affected
        if progress_cb:
            progress_cb("apply", idx, n_ops)

    session.commit()
    return total

# =========================
# GROUPING (SAFE)
# =========================

def _fallback_grouping(session: Session):
    rows = session.execute(select(PropertyName.property_name_id, PropertyName.name)).all()

    return {pid: name for pid, name in rows}


def build_property_name_groups(
    session: Session,
    use_llm=True,
    llm_model_dir=None,
    distance_threshold: float = 0.6,
    progress_cb: Optional[ProgressCallback] = None,
):
    """
    distance_threshold зарезервирован под будущую кластеризацию имён свойств;
    в текущем fallback/LLM-заглушке не используется.
    """
    _ = distance_threshold

    if progress_cb:
        progress_cb("group", 0, 1)

    if use_llm and llm_model_dir:
        try:
            out = build_property_name_groups_llm(session, llm_model_dir)
            if progress_cb:
                progress_cb("group", 1, 1)
            return out
        except Exception as e:
            LOG.warning("LLM failed: %s", e)

    out = _fallback_grouping(session)
    if progress_cb:
        progress_cb("group", 1, 1)
    return out


_CLS_TOKENIZER = None
_CLS_MODEL = None
_CLS_RESOLVED_DIR: Optional[str] = None


def _resolve_prop_groups_model_dir(model_dir: str) -> Path:
    p = Path(model_dir)
    if not p.is_absolute():
        base = Path(__file__).resolve().parent
        p = (base / model_dir).resolve()
    return p


def _prop_groups_has_weights(d: Path) -> bool:
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def _id2label_str(model, class_id: int) -> str:
    raw = getattr(model.config, "id2label", None) or {}
    v = raw.get(class_id)
    if v is None:
        v = raw.get(str(class_id))
    return str(v) if v is not None else str(int(class_id))


def _load_prop_groups_classifier(model_dir: Path):
    global _CLS_TOKENIZER, _CLS_MODEL, _CLS_RESOLVED_DIR
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Нужны пакеты torch и transformers (см. requirements.txt)"
        ) from e

    key = str(model_dir.resolve())
    if _CLS_RESOLVED_DIR == key and _CLS_MODEL is not None and _CLS_TOKENIZER is not None:
        return _CLS_TOKENIZER, _CLS_MODEL

    if not model_dir.is_dir():
        raise RuntimeError(f"Каталог модели не найден: {model_dir}")

    if not _prop_groups_has_weights(model_dir):
        raise RuntimeError(
            f"В {model_dir} нет весов классификатора (model.safetensors или pytorch_model.bin). "
            "Обучите модель: python train_prop_groups_llm.py"
        )

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _CLS_TOKENIZER = tok
    _CLS_MODEL = model
    _CLS_RESOLVED_DIR = key
    return tok, model


def build_property_name_groups_llm(session: Session, model_dir: str) -> Dict[int, str]:
    """
    Классификация имён свойств в группы дообученной rubert-tiny2
    (см. train_prop_groups_llm.py). Возвращает property_name_id -> название группы.
    """
    import torch

    override = os.getenv("PROP_GROUPS_LLM_DIR", "").strip()
    resolved = _resolve_prop_groups_model_dir(override or model_dir)
    tokenizer, model = _load_prop_groups_classifier(resolved)
    device = next(model.parameters()).device

    rows = session.execute(select(PropertyName.property_name_id, PropertyName.name)).all()
    if not rows:
        return {}

    batch_size = int(os.getenv("PROP_GROUPS_LLM_BATCH", "32"))
    max_length = 64

    pids: List[int] = []
    texts: List[str] = []
    for pid, name in rows:
        pids.append(int(pid))
        t = (name or "").strip()
        texts.append(t if t else "?")

    out: Dict[int, str] = {}
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            pred = logits.argmax(dim=-1).tolist()
            for j, class_id in enumerate(pred):
                pid = pids[i + j]
                out[pid] = _id2label_str(model, int(class_id))

    LOG.info(
        "Mini-LLM групп: модель %s, свойств: %d",
        resolved,
        len(out),
    )
    return out


# =========================
# MERGE DUPLICATE PROPERTY NAMES
# =========================

from sqlalchemy import delete


def merge_duplicate_property_names(
    session: Session,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[int, int]:
    """Объединяет дублирующиеся названия свойств.

    Возвращает (число объединённых дубликатов, число удалённых строк property_names).
    """

    stmt = select(PropertyName.property_name_id, PropertyName.name)
    rows = session.execute(stmt).all()

    name_groups = defaultdict(list)
    for pid, name in rows:
        normalized_name = normalize_whitespace(name.lower())
        name_groups[normalized_name].append((pid, name))

    keys = list(name_groups.keys())
    total_groups = len(keys) or 1

    merged_count = 0
    for gi, normalized_name in enumerate(keys):
        group = name_groups[normalized_name]
        if len(group) > 1:
            canonical_pid, canonical_name = sorted(group, key=lambda x: x[1])[0]

            for pid, name in group[1:]:
                if pid != canonical_pid:
                    update_stmt = (
                        update(ProductProperty)
                        .where(ProductProperty.property_name_id == pid)
                        .values(property_name_id=canonical_pid)
                    )
                    session.execute(update_stmt)

                    delete_stmt = (
                        delete(PropertyName)
                        .where(PropertyName.property_name_id == pid)
                    )
                    session.execute(delete_stmt)

                    merged_count += 1
        if progress_cb:
            progress_cb("merge", gi + 1, total_groups)

    session.commit()
    LOG.info("Объединено %s дублирующихся свойств", merged_count)
    # каждое объединение удаляет одну строку property_names
    return merged_count, merged_count


# =========================
# TASK 2/3 METRICS + CATEGORY-AWARE NORMALIZATION
# =========================

def collect_by_category(session: Session) -> Dict[str, Dict[str, Counter]]:
    """
    category_name -> property_name -> Counter(normalized_value -> count)
    """
    values: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    stmt = (
        select(Category.name, PropertyName.name, ProductProperty.value)
        .select_from(ProductProperty)
        .join(PropertyName, PropertyName.property_name_id == ProductProperty.property_name_id)
        .join(ProductOffer, ProductOffer.offer_id == ProductProperty.offer_id)
        .join(Category, Category.category_id == ProductOffer.category_id, isouter=True)
    )
    for cat_name, prop_name, val in session.execute(stmt):
        if not val or not prop_name:
            continue
        c = cat_name or "Без категории"
        values[c][prop_name][normalize_basic_value(val)] += 1
    return values


def build_mapping_by_category(session: Session) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    category_name -> property_name -> {raw_norm: canonical}
    """
    grouped = collect_by_category(session)
    all_vals: List[str] = []
    for props in grouped.values():
        for counter in props.values():
            all_vals.extend(counter.keys())

    embedder = None
    if all_vals and TfidfVectorizer:
        embedder = TfidfEmbedder()
        embedder.fit(all_vals)

    result: Dict[str, Dict[str, Dict[str, str]]] = {}
    for cat_name, props in grouped.items():
        cat_map: Dict[str, Dict[str, str]] = {}
        for prop_name, counter in props.items():
            vals = list(counter.keys())
            if embedder and len(vals) > 1:
                m = cluster_values(vals, counter, embedder)
            else:
                m = {v: v for v in vals}
            cat_map[prop_name] = m
        result[cat_name] = cat_map
    return result


def apply_mapping_by_category(session: Session, mapping_by_category: Dict[str, Dict[str, Dict[str, str]]]) -> int:
    """
    Применяет category-aware mapping.
    """
    total = 0
    rows = session.execute(
        select(PropertyName.property_name_id, PropertyName.name)
    ).all()
    name_to_id = {name: pid for pid, name in rows}

    for cat_name, cat_map in mapping_by_category.items():
        for prop_name, raw_map in cat_map.items():
            pid = name_to_id.get(prop_name)
            if not pid:
                continue
            for raw, canon in raw_map.items():
                if raw == canon:
                    continue
                stmt = (
                    update(ProductProperty)
                    .where(ProductProperty.property_name_id == pid)
                    .where(
                        ProductProperty.offer_id.in_(
                            select(ProductOffer.offer_id).join(Category, isouter=True).where(
                                (Category.name == cat_name) if cat_name != "Без категории" else (ProductOffer.category_id.is_(None))
                            )
                        )
                    )
                    .where(ProductProperty.value == raw)
                    .values(value=canon)
                )
                res = session.execute(stmt)
                total += res.rowcount if hasattr(res, "rowcount") and res.rowcount else 0
    session.commit()
    return total


def evaluate_filter_quality(session: Session) -> Dict[str, float]:
    """
    Метрики качества набора характеристик для фильтров (Task 2):
    - coverage: доля офферов с >=1 свойством
    - duplicates_ratio: доля дублей имён свойств (по lower/trim)
    - consistency: средняя доля моды по значениям свойства (взвешенная)
    """
    total_offers = session.execute(select(func.count()).select_from(ProductOffer)).scalar() or 0
    offers_with_props = session.execute(
        select(func.count(func.distinct(ProductProperty.offer_id)))
    ).scalar() or 0
    coverage = (offers_with_props / total_offers) if total_offers else 0.0

    prop_rows = session.execute(select(PropertyName.name)).all()
    names = [normalize_whitespace((n or "").lower()) for (n,) in prop_rows if n]
    unique_names = len(set(names))
    total_names = len(names)
    duplicates_ratio = (1.0 - (unique_names / total_names)) if total_names else 0.0

    values = collect(session)
    weighted_mode_sum = 0.0
    weighted_total = 0
    for _prop, counter in values.items():
        total = sum(counter.values())
        if total <= 0:
            continue
        mode_count = max(counter.values())
        weighted_mode_sum += mode_count
        weighted_total += total
    consistency = (weighted_mode_sum / weighted_total) if weighted_total else 0.0

    return {
        "coverage": round(float(coverage), 4),
        "duplicates_ratio": round(float(duplicates_ratio), 4),
        "consistency": round(float(consistency), 4),
    }


def evaluate_category_consistency(session: Session) -> List[Dict[str, float]]:
    """
    Метрики качества нормализации по категориям (Task 3):
    - offers: количество офферов в категории
    - avg_values_per_property: среднее число уникальных значений на свойство (меньше лучше)
    - weighted_mode_ratio: доля моды по всем category/property (больше лучше)
    """
    grouped = collect_by_category(session)

    offer_counts = dict(
        session.execute(
            select(Category.name, func.count(ProductOffer.offer_id))
            .select_from(ProductOffer)
            .join(Category, Category.category_id == ProductOffer.category_id, isouter=True)
            .group_by(Category.name)
        ).all()
    )

    result: List[Dict[str, float]] = []
    for cat_name, props in grouped.items():
        prop_unique_counts = []
        mode_sum = 0
        total_sum = 0
        for _prop, counter in props.items():
            prop_unique_counts.append(len(counter))
            t = sum(counter.values())
            if t > 0:
                mode_sum += max(counter.values())
                total_sum += t
        avg_values = (sum(prop_unique_counts) / len(prop_unique_counts)) if prop_unique_counts else 0.0
        mode_ratio = (mode_sum / total_sum) if total_sum else 0.0
        result.append(
            {
                "category": cat_name,
                "offers": float(offer_counts.get(cat_name, 0) or 0),
                "avg_values_per_property": round(float(avg_values), 4),
                "weighted_mode_ratio": round(float(mode_ratio), 4),
            }
        )
    result.sort(key=lambda x: x["offers"], reverse=True)
    return result