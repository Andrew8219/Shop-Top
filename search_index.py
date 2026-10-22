from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session, selectinload

from models import ProductOffer, ProductProperty, SearchIndex

LOG = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_DIM = 384
DEFAULT_CROSS_ENCODER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Путь к дообученному bi-encoder (если задан — используется вместо дефолтного).
# Установите в .env: SEARCH_BI_ENCODER_MODEL=models/bi_encoder_finetuned
_BI_ENCODER_MODEL: str = os.getenv("SEARCH_BI_ENCODER_MODEL", "").strip() or DEFAULT_MODEL_NAME
# Путь к cross-encoder (если задан — переопределяет дефолтный).
# Установите в .env: SEARCH_CROSS_ENCODER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
_CROSS_ENCODER_MODEL: str = (
    os.getenv("SEARCH_CROSS_ENCODER_MODEL", "").strip() or DEFAULT_CROSS_ENCODER_NAME
)

# Размер чанка при индексировании — чтобы не грузить все офферы в память сразу
_INDEX_CHUNK_SIZE = int(os.getenv("SEARCH_INDEX_CHUNK_SIZE", "500"))


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _model_cache_dir() -> Optional[str]:
    # Явный путь к кэшу моделей (если не задан — библиотека использует дефолтный HF cache).
    p = os.getenv("HF_HOME", "").strip() or os.getenv("SENTENCE_TRANSFORMERS_HOME", "").strip()
    return p or None


@dataclass(frozen=True)
class SearchHit:
    offer_id: int
    score: float
    source: str  # "semantic" | "text" | "hybrid"


def _faiss_paths(root: Path) -> Tuple[Path, Path]:
    return root / "faiss.index", root / "faiss_offer_ids.json"


def _get_store_dir() -> Path:
    p = os.getenv("SEARCH_INDEX_DIR", "search_artifacts")
    return Path(p)


# =====================
#   FAISS IN-MEMORY CACHE
# =====================

_faiss_cache_lock = threading.Lock()
_faiss_index = None   # faiss.Index | None
_faiss_ids: Optional[List[int]] = None


def _get_faiss_cached() -> Tuple[Optional[object], Optional[List[int]]]:
    """Возвращает закешированный FAISS-индекс (читает с диска только при первом вызове)."""
    global _faiss_index, _faiss_ids
    with _faiss_cache_lock:
        if _faiss_index is None:
            _faiss_index, _faiss_ids = load_faiss()
        return _faiss_index, _faiss_ids


def invalidate_faiss_cache() -> None:
    """Сбрасывает кэш FAISS. Вызывается после перестройки индекса."""
    global _faiss_index, _faiss_ids
    with _faiss_cache_lock:
        _faiss_index = None
        _faiss_ids = None
    LOG.info("FAISS cache invalidated")


# =====================
#   pgvector helpers
# =====================

def ensure_pgvector_extension(session: Session) -> None:
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.commit()


def ensure_pgvector_column(session: Session, dim: int = DEFAULT_DIM) -> None:
    session.execute(
        text(f"ALTER TABLE search_index ADD COLUMN IF NOT EXISTS embedding_vec vector({dim})")
    )
    session.commit()


def ensure_gin_index(session: Session) -> None:
    """Создаёт GIN-индекс на tsv если его нет — ускоряет полнотекстовый поиск в разы."""
    session.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_search_index_tsv "
            "ON search_index USING GIN(tsv)"
        )
    )
    session.commit()
    LOG.info("GIN index on search_index.tsv ensured")


# =====================
#   TEXT BUILDING
# =====================

def build_offer_text(offer: ProductOffer) -> str:
    parts: List[str] = []
    if offer.short_name:
        parts.append(offer.short_name)
    if offer.company and offer.company.name:
        parts.append(offer.company.name)
    if offer.category and offer.category.name:
        parts.append(offer.category.name)

    pr = getattr(offer, "price_rub", None)
    if pr is not None:
        try:
            parts.append(f"цена {float(pr):.0f} руб")
        except (TypeError, ValueError):
            pass

    for prop in offer.properties or []:
        if prop.property_name and prop.property_name.name:
            pname = prop.property_name.name
        else:
            pname = ""
        pval = prop.value or ""
        if pname and pval:
            parts.append(f"{pname}: {pval}")
        elif pval:
            parts.append(pval)

    return " | ".join([p.strip() for p in parts if p and p.strip()])


# =====================
#   EMBEDDINGS
# =====================

@lru_cache(maxsize=2)
def _lazy_load_embedder(model_name: str = DEFAULT_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    cache_dir = _model_cache_dir()
    offline_only = (
        _env_truthy("SEARCH_OFFLINE_ONLY")
        or _env_truthy("HF_OFFLINE", "1")
        or _env_truthy("HF_HUB_OFFLINE")
        or _env_truthy("TRANSFORMERS_OFFLINE")
    )
    try:
        return SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            local_files_only=offline_only,
        )
    except Exception as e:
        if offline_only:
            raise RuntimeError(
                f"Model '{model_name}' not found in local cache while offline mode is enabled. "
                "Run once with internet to warm up the cache."
            ) from e
        LOG.warning("Model load failed online, retrying from local cache only: %s", e)
        return SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            local_files_only=True,
        )


@lru_cache(maxsize=1)
def _lazy_load_cross_encoder(model_name: str = ""):
    """
    Лениво загружает cross-encoder модель. Возвращает None если модель недоступна
    (не скачана и режим офлайн, или нет зависимостей).
    """
    effective_name = model_name or _CROSS_ENCODER_MODEL
    offline_only = (
        _env_truthy("SEARCH_OFFLINE_ONLY")
        or _env_truthy("HF_OFFLINE", "1")
        or _env_truthy("HF_HUB_OFFLINE")
        or _env_truthy("TRANSFORMERS_OFFLINE")
    )
    try:
        from sentence_transformers.cross_encoder import CrossEncoder

        return CrossEncoder(
            effective_name,
            max_length=512,
            device="cpu",
            local_files_only=offline_only,
        )
    except Exception as e:
        LOG.info(
            "Cross-encoder '%s' не загружен (%s). "
            "Запустите: python -c \"from sentence_transformers.cross_encoder import CrossEncoder; CrossEncoder('%s')\" "
            "для скачивания модели. Будет использован bi-encoder fallback.",
            effective_name,
            e,
            effective_name,
        )
        return None


def embed_passages(texts: Sequence[str], model_name: str = "") -> np.ndarray:
    model = _lazy_load_embedder(model_name or _BI_ENCODER_MODEL)
    passages = [f"passage: {t}" for t in texts]
    emb = model.encode(
        passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) >= 200,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_query(q: str, model_name: str = "") -> np.ndarray:
    model = _lazy_load_embedder(model_name or _BI_ENCODER_MODEL)
    emb = model.encode(
        [f"query: {q}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)[0]


# =====================
#   FAISS
# =====================

def build_faiss_index(embeddings: np.ndarray, offer_ids: Sequence[int]):
    import faiss

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, list(map(int, offer_ids))


def save_faiss(index, offer_ids: Sequence[int], out_dir: Path) -> None:
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path, map_path = _faiss_paths(out_dir)
    faiss.write_index(index, str(idx_path))
    map_path.write_text(
        json.dumps(list(map(int, offer_ids)), ensure_ascii=False), encoding="utf-8"
    )


def load_faiss(out_dir: Optional[Path] = None):
    import faiss

    out_dir = out_dir or _get_store_dir()
    idx_path, map_path = _faiss_paths(out_dir)
    if not idx_path.exists() or not map_path.exists():
        return None, None
    index = faiss.read_index(str(idx_path))
    offer_ids = json.loads(map_path.read_text(encoding="utf-8"))
    return index, offer_ids


def semantic_search_faiss(q: str, top_k: int = 50) -> List[Tuple[int, int]]:
    """Возвращает список (offer_id, rank). Индекс берётся из кэша в памяти."""
    index, offer_ids = _get_faiss_cached()
    if index is None or not offer_ids:
        return []
    qv = np.asarray([embed_query(q)], dtype=np.float32)
    scores, idxs = index.search(qv, top_k)
    result: List[Tuple[int, int]] = []
    rank = 1
    for i in idxs[0].tolist():
        if i < 0:
            continue
        result.append((int(offer_ids[i]), rank))
        rank += 1
    return result


def semantic_search_pgvector(session: Session, q: str, top_k: int = 50) -> List[Tuple[int, int]]:
    qv = embed_query(q).tolist()
    rows = session.execute(
        text(
            "SELECT offer_id "
            "FROM search_index "
            "WHERE embedding_vec IS NOT NULL "
            "ORDER BY (embedding_vec <#> :qv) ASC "
            "LIMIT :k"
        ),
        {"qv": qv, "k": int(top_k)},
    ).all()
    return [(int(oid), idx + 1) for idx, (oid,) in enumerate(rows)]


def _fts_query(session: Session, q: str, top_k: int) -> List[int]:
    tsq = func.websearch_to_tsquery("russian", q)
    rank = func.ts_rank_cd(SearchIndex.tsv, tsq)
    stmt = (
        select(SearchIndex.offer_id)
        .where(SearchIndex.tsv.op("@@")(tsq))
        .order_by(rank.desc())
        .limit(top_k)
    )
    return [int(oid) for (oid,) in session.execute(stmt).all()]


def text_search(session: Session, q: str, top_k: int = 50) -> List[Tuple[int, int]]:
    """
    Полнотекстовый поиск PostgreSQL. Сначала строгий AND по всем словам;
    если ничего не нашлось (например, запрос расширен синонимами, которых нет
    в карточке) — OR по словам с ранжированием по числу совпадений (ts_rank_cd).
    """
    ids = _fts_query(session, q, top_k)
    if not ids:
        words = re.findall(r"\w+", q or "", flags=re.UNICODE)
        if len(words) > 1:
            ids = _fts_query(session, " or ".join(words), top_k)
    return [(oid, idx + 1) for idx, oid in enumerate(ids)]


# =====================
#   RRF MERGE
# =====================

def rrf_merge(
    semantic: Sequence[Tuple[int, int]],
    text_ranks: Sequence[Tuple[int, int]],
    *,
    k: int = 60,
    limit: int = 50,
) -> List[int]:
    scores: Dict[int, float] = {}
    for oid, r in semantic:
        scores[oid] = scores.get(oid, 0.0) + 1.0 / (k + r)
    for oid, r in text_ranks:
        scores[oid] = scores.get(oid, 0.0) + 1.0 / (k + r)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [oid for oid, _ in ordered[:limit]]


def merge_search_rankings(
    mode: str,
    semantic: Sequence[Tuple[int, int]],
    text_ranks: Sequence[Tuple[int, int]],
    *,
    k: int = 60,
    limit: int = 500,
) -> List[int]:
    """
    Объединяет семантический и полнотекстовый поиск в зависимости от режима.
    mode: 'hybrid' | 'semantic' | 'fulltext'
    """
    if mode == "semantic":
        return [oid for oid, _ in sorted(semantic, key=lambda x: x[1])[:limit]]
    if mode == "fulltext":
        return [oid for oid, _ in sorted(text_ranks, key=lambda x: x[1])[:limit]]
    # hybrid (default): RRF
    return rrf_merge(semantic, text_ranks, k=k, limit=limit)


# =====================
#   TOKEN FILTER
# =====================

_STOPWORDS_SEARCH = frozenset(
    {
        "и", "в", "во", "не", "на", "я", "с", "со", "как", "а", "то", "все",
        "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
        "бы", "по", "из", "от", "до", "при", "мне", "мы", "он",
        "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "is", "it",
    }
)


@lru_cache(maxsize=1)
def _retrieval_fluff() -> frozenset:
    """Разговорные/субъективные слова из словаря намерений (их нет в карточках)."""
    try:
        from search_intent_lexicon import INTENT_RETRIEVAL_FLUFF

        return frozenset(INTENT_RETRIEVAL_FLUFF)
    except Exception:
        return frozenset()


def tokenize_search_query(q: str) -> List[str]:
    """Слова и числа из запроса (нижний регистр), без короткого мусора и стоп-слов."""
    q = (q or "").strip().lower()
    if not q:
        return []
    parts = re.findall(r"\w+", q, flags=re.UNICODE)
    fluff = _retrieval_fluff()
    out: List[str] = []
    for p in parts:
        if p in _STOPWORDS_SEARCH or p in fluff:
            continue
        if len(p) < 2 and not p.isdigit():
            continue
        if len(p) > 64:
            continue
        out.append(p)
    return out


def compact_query_for_retrieval(q: str) -> str:
    """
    Убирает стоп-слова и шумовые токены из запроса перед передачей в модель эмбеддингов.
    Возвращает строку из значимых токенов через пробел.
    """
    return " ".join(tokenize_search_query(q))


def _compact_alnum(s: str) -> str:
    return re.sub(r"[\s\-_\.]+", "", s, flags=re.UNICODE)


def token_matches_in_offer_text(token: str, hay_lower: str) -> bool:
    """
    Числа — только как отдельное число (не подстрока внутри 194).
    Слова — подстрока в hay или совпадение без пробелов/дефисов (GF63 ≈ GF-63).
    """
    if not token:
        return True
    if token.isdigit():
        return (
            re.search(r"(^|[^\d])" + re.escape(token) + r"([^\d]|$)", hay_lower) is not None
        )
    if token in hay_lower:
        return True
    # Русская словоформа: «камерой» ≈ «камера» (совпадение по основе без окончания).
    if len(token) >= 6 and re.fullmatch(r"[а-яё]+", token):
        for cut in (1, 2, 3):
            stem = token[:-cut]
            if len(stem) >= 4 and stem in hay_lower:
                return True
    if len(token) >= 2:
        ct = _compact_alnum(token)
        if ct and ct in _compact_alnum(hay_lower):
            return True
    return False


def _required_token_matches(n_tokens: int) -> int:
    """
    Сколько токенов должно совпасть:
    - 1-3 токена: все (строгий AND)
    - 4-5 токенов: n-1 (1 промах допускается)
    - 6+ токенов: n-2 (2 промаха допускаются)
    Это делает поиск более терпимым для длинных запросов.
    """
    if n_tokens <= 3:
        return n_tokens
    if n_tokens <= 5:
        return n_tokens - 1
    return n_tokens - 2


def filter_offer_ids_by_query_tokens(
    session: Session,
    offer_ids: Sequence[int],
    q: str,
) -> Tuple[List[int], Dict[int, ProductOffer]]:
    """
    Оставляет только офферы, текст которых содержит достаточно токенов из запроса.
    Возвращает (отфильтрованные id в исходном порядке, словарь offer_id → оффер).
    Словарь можно передать в fetch_offers_in_order для повторного использования.
    """
    tokens = tokenize_search_query(q)
    if not tokens or not offer_ids:
        return list(offer_ids), {}

    stmt = (
        select(ProductOffer)
        .options(
            selectinload(ProductOffer.company),
            selectinload(ProductOffer.category),
            selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
        )
        .where(ProductOffer.offer_id.in_(list(map(int, offer_ids))))
    )
    offers = session.execute(stmt).scalars().all()
    by_id: Dict[int, ProductOffer] = {int(o.offer_id): o for o in offers}

    required = _required_token_matches(len(tokens))
    out: List[int] = []
    for oid in offer_ids:
        o = by_id.get(int(oid))
        if o is None:
            continue
        hay = build_offer_text(o).lower()
        matched = sum(1 for t in tokens if token_matches_in_offer_text(t, hay))
        if matched >= required:
            out.append(int(oid))
    return out, by_id


def fetch_offers_in_order(
    session: Session,
    offer_ids: Sequence[int],
    cache: Optional[Dict[int, ProductOffer]] = None,
) -> List[ProductOffer]:
    """
    Возвращает офферы в порядке offer_ids.
    Если передан cache (от filter_offer_ids_by_query_tokens), подгружает только недостающие.
    """
    if not offer_ids:
        return []

    by_id: Dict[int, ProductOffer] = dict(cache) if cache else {}
    missing = [oid for oid in offer_ids if oid not in by_id]

    if missing:
        stmt = (
            select(ProductOffer)
            .options(
                selectinload(ProductOffer.company),
                selectinload(ProductOffer.category),
                selectinload(ProductOffer.product),
                selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
            )
            .where(ProductOffer.offer_id.in_(list(map(int, missing))))
        )
        for o in session.execute(stmt).scalars().all():
            by_id[int(o.offer_id)] = o

    return [by_id[oid] for oid in offer_ids if oid in by_id]


def filter_offer_ids_by_query_tokens_soft(
    session: Session,
    offer_ids: Sequence[int],
    q: str,
) -> List[int]:
    """
    Обёртка над filter_offer_ids_by_query_tokens, возвращает только список ids.
    Используется в поисковом пайплайне — cache не нужен на этом шаге.
    """
    filtered, _ = filter_offer_ids_by_query_tokens(session, offer_ids, q)
    return filtered


def rerank_offer_ids_cross_encoder(
    session: Session,
    q: str,
    offer_ids: Sequence[int],
    *,
    head_size: int = 50,
) -> List[int]:
    """
    Переставляет топ-head_size результатов по релевантности запросу.

    Стратегия (в порядке приоритета):
    1. Cross-encoder: joint scoring пары (запрос, текст_товара). Значительно точнее bi-encoder.
       Требует модель 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1' в локальном кэше HF.
    2. Bi-encoder fallback: dot product эмбеддинга запроса и хранимых эмбеддингов офферов.
       Работает всегда если индекс построен.

    Остальные результаты (tail) остаются в исходном порядке после переставленного топа.
    """
    if not offer_ids:
        return []

    head = list(offer_ids[:head_size])
    tail = list(offer_ids[head_size:])

    if not head:
        return list(offer_ids)

    # --- попытка 1: cross-encoder ---
    ce = _lazy_load_cross_encoder()
    if ce is not None:
        try:
            rows = session.execute(
                select(SearchIndex.offer_id, SearchIndex.text)
                .where(SearchIndex.offer_id.in_(head))
                .where(SearchIndex.text.isnot(None))
            ).all()
            if rows:
                oid_list = [int(r[0]) for r in rows]
                pairs = [(q, r[1]) for r in rows]
                scores_arr = ce.predict(pairs, show_progress_bar=False)
                scored: Dict[int, float] = {oid: float(s) for oid, s in zip(oid_list, scores_arr)}
                reranked = sorted(head, key=lambda oid: scored.get(oid, -999.0), reverse=True)
                LOG.debug("rerank: cross-encoder scored %d pairs", len(pairs))
                return reranked + tail
        except Exception:
            LOG.debug("rerank: cross-encoder scoring failed, falling back to bi-encoder", exc_info=True)

    # --- попытка 2: bi-encoder (dot product из хранимых эмбеддингов) ---
    try:
        rows = session.execute(
            select(SearchIndex.offer_id, SearchIndex.embedding)
            .where(SearchIndex.offer_id.in_(head))
            .where(SearchIndex.embedding.isnot(None))
        ).all()
    except Exception:
        LOG.debug("rerank: не удалось загрузить эмбеддинги", exc_info=True)
        return list(offer_ids)

    if not rows:
        return list(offer_ids)

    try:
        qv = embed_query(q)
        bi_scored: Dict[int, float] = {}
        for oid, emb_json in rows:
            try:
                emb = np.array(json.loads(emb_json), dtype=np.float32)
                bi_scored[int(oid)] = float(np.dot(qv, emb))
            except Exception:
                pass
        reranked = sorted(head, key=lambda oid: bi_scored.get(oid, -1.0), reverse=True)
        return reranked + tail
    except Exception:
        LOG.debug("rerank: bi-encoder scoring failed", exc_info=True)
        return list(offer_ids)


# =====================
#   BUILD INDEX
# =====================

def build_search_index(
    session: Session,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    rebuild: bool = True,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, int]:
    """
    Собирает search_index + FAISS артефакты.
    Офферы обрабатываются чанками по _INDEX_CHUNK_SIZE — избегаем OOM на больших каталогах.

    progress_cb(stage, current, total): stage — collect | embed | faiss (для UI джоба).
    """
    if rebuild:
        session.execute(delete(SearchIndex))
        session.commit()

    # Получаем все offer_id — легковесно
    all_ids: List[int] = list(
        session.execute(
            select(ProductOffer.offer_id).order_by(ProductOffer.offer_id)
        ).scalars().all()
    )
    if progress_cb:
        try:
            progress_cb("collect", 1, 1)
        except Exception:
            LOG.debug("progress_cb(collect) failed", exc_info=True)
    if not all_ids:
        return {"offers_indexed": 0}

    # Пытаемся включить pgvector (если нет прав — продолжаем без него)
    pgvector_ok = True
    try:
        ensure_pgvector_extension(session)
    except Exception:
        session.rollback()
        pgvector_ok = False

    # Если модель эмбеддингов недоступна (нет sentence-transformers / весов),
    # строим только полнотекстовый индекс — витрина и FTS-поиск продолжают работать.
    semantic_ok = True
    try:
        _lazy_load_embedder(model_name or _BI_ENCODER_MODEL)
    except Exception as exc:
        LOG.warning("Модель эмбеддингов недоступна (%s): строится только полнотекстовый индекс", exc)
        semantic_ok = False

    all_embeddings_list: List[np.ndarray] = []
    all_indexed_ids: List[int] = []

    total_chunks = (len(all_ids) + _INDEX_CHUNK_SIZE - 1) // _INDEX_CHUNK_SIZE
    LOG.info("Индексирование %d офферов чанками по %d (%d чанков)", len(all_ids), _INDEX_CHUNK_SIZE, total_chunks)

    for chunk_num, chunk_start in enumerate(range(0, len(all_ids), _INDEX_CHUNK_SIZE)):
        if progress_cb:
            try:
                progress_cb("embed", chunk_num + 1, max(1, total_chunks))
            except Exception:
                LOG.debug("progress_cb(embed) failed", exc_info=True)
        chunk_ids = all_ids[chunk_start: chunk_start + _INDEX_CHUNK_SIZE]

        stmt = (
            select(ProductOffer)
            .options(
                selectinload(ProductOffer.company),
                selectinload(ProductOffer.category),
                selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
            )
            .where(ProductOffer.offer_id.in_(chunk_ids))
            .order_by(ProductOffer.offer_id)
        )
        chunk_offers = session.execute(stmt).scalars().all()

        chunk_texts: List[str] = []
        chunk_ids_valid: List[int] = []
        for o in chunk_offers:
            txt = build_offer_text(o)
            if txt:
                chunk_texts.append(txt)
                chunk_ids_valid.append(int(o.offer_id))

        if not chunk_texts:
            continue

        LOG.info("Чанк %d/%d: %d офферов", chunk_num + 1, total_chunks, len(chunk_texts))
        chunk_embeddings = (
            embed_passages(chunk_texts, model_name=model_name) if semantic_ok else None
        )

        # Записываем в search_index
        for i, oid in enumerate(chunk_ids_valid):
            row = SearchIndex(
                offer_id=oid,
                text=chunk_texts[i],
                tsv=None,
                embedding=(
                    json.dumps(chunk_embeddings[i].tolist(), ensure_ascii=False)
                    if chunk_embeddings is not None
                    else None
                ),
            )
            session.add(row)
        session.commit()

        if chunk_embeddings is not None:
            all_embeddings_list.append(chunk_embeddings)
        all_indexed_ids.extend(chunk_ids_valid)

        # Освобождаем объекты офферов из памяти сессии
        session.expunge_all()

    if not all_indexed_ids:
        return {"offers_indexed": 0}

    # tsvector считаем внутри БД одним UPDATE
    session.execute(
        text(
            "UPDATE search_index "
            "SET tsv = to_tsvector('russian', coalesce(text,'')) "
            "WHERE tsv IS NULL"
        )
    )
    session.commit()

    # GIN-индекс — создаём/обновляем после заполнения данных
    try:
        ensure_gin_index(session)
    except Exception:
        LOG.warning("Не удалось создать GIN-индекс (возможно, нет прав)", exc_info=True)
        session.rollback()

    if not semantic_ok:
        invalidate_faiss_cache()
        LOG.info("Индексирование завершено (только FTS): %d офферов", len(all_indexed_ids))
        return {"offers_indexed": len(all_indexed_ids), "pgvector_ok": 0, "semantic": 0}

    # Собираем все эмбеддинги для FAISS
    all_embeddings = np.vstack(all_embeddings_list)

    if progress_cb:
        try:
            progress_cb("faiss", 0, 1)
        except Exception:
            LOG.debug("progress_cb(faiss) failed", exc_info=True)

    # pgvector: batch-update вместо цикла по одной строке
    if pgvector_ok:
        try:
            ensure_pgvector_column(session, dim=int(all_embeddings.shape[1]))
            batch_params = [
                {"v": all_embeddings[i].tolist(), "oid": int(all_indexed_ids[i])}
                for i in range(len(all_indexed_ids))
            ]
            session.execute(
                text("UPDATE search_index SET embedding_vec = :v WHERE offer_id = :oid"),
                batch_params,
            )
            session.commit()
        except Exception:
            LOG.warning("pgvector batch update не удался", exc_info=True)
            session.rollback()
            pgvector_ok = False

    # Строим и сохраняем FAISS-индекс
    faiss_index, faiss_offer_ids = build_faiss_index(all_embeddings, all_indexed_ids)
    save_faiss(faiss_index, faiss_offer_ids, _get_store_dir())

    if progress_cb:
        try:
            progress_cb("faiss", 1, 1)
        except Exception:
            LOG.debug("progress_cb(faiss done) failed", exc_info=True)

    # Сбрасываем in-memory кэш — при следующем поиске загрузится новый индекс
    invalidate_faiss_cache()

    LOG.info("Индексирование завершено: %d офферов", len(all_indexed_ids))
    return {
        "offers_indexed": len(all_indexed_ids),
        "pgvector_ok": 1 if pgvector_ok else 0,
    }


def parse_query_intent(q: str):
    """
    Разбор намерения запроса (цена, RAM, SSD, диагональ и т.д.) для поиска.
    Реализация в search_smart; экспорт здесь — рядом с токенизацией/индексом.
    """
    from search_smart import parse_query_intent as _parse_query_intent_impl

    return _parse_query_intent_impl(q)
