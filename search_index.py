from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session, selectinload

from models import ProductOffer, ProductProperty, SearchIndex


DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_DIM = 384


@dataclass(frozen=True)
class SearchHit:
    offer_id: int
    score: float
    source: str  # "semantic" | "text" | "hybrid"


def _faiss_paths(root: Path) -> Tuple[Path, Path]:
    return root / "faiss.index", root / "faiss_offer_ids.json"


def _get_store_dir() -> Path:
    # можно переопределить через env (удобно для деплоя)
    p = os.getenv("SEARCH_INDEX_DIR", "search_artifacts")
    return Path(p)


def ensure_pgvector_extension(session: Session) -> None:
    # Безопасно, если расширение уже есть.
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.commit()


def ensure_pgvector_column(session: Session, dim: int = DEFAULT_DIM) -> None:
    # Создаём колонку под pgvector только если расширение реально доступно.
    session.execute(text(f"ALTER TABLE search_index ADD COLUMN IF NOT EXISTS embedding_vec vector({dim})"))
    session.commit()


def build_offer_text(offer: ProductOffer) -> str:
    parts: List[str] = []
    if offer.short_name:
        parts.append(offer.short_name)
    if offer.company and offer.company.name:
        parts.append(offer.company.name)
    if offer.category and offer.category.name:
        parts.append(offer.category.name)

    # свойства: "Название: значение"
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


@lru_cache(maxsize=2)
def _lazy_load_embedder(model_name: str = DEFAULT_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    # E5 рекомендует normalize_embeddings=True и similarity via dot/inner product
    return SentenceTransformer(model_name)


def embed_passages(texts: Sequence[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    model = _lazy_load_embedder(model_name)
    passages = [f"passage: {t}" for t in texts]
    emb = model.encode(
        passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) >= 200,
    )
    emb = np.asarray(emb, dtype=np.float32)
    return emb


def embed_query(q: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    model = _lazy_load_embedder(model_name)
    emb = model.encode(
        [f"query: {q}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    emb = np.asarray(emb, dtype=np.float32)
    return emb[0]


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
    map_path.write_text(json.dumps(list(map(int, offer_ids)), ensure_ascii=False), encoding="utf-8")


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
    """
    Возвращает список (offer_id, rank), rank начиная с 1.
    """
    index, offer_ids = load_faiss()
    if index is None or not offer_ids:
        return []
    qv = embed_query(q)
    qv = np.asarray([qv], dtype=np.float32)
    scores, idxs = index.search(qv, top_k)
    idxs = idxs[0].tolist()
    result: List[Tuple[int, int]] = []
    rank = 1
    for i in idxs:
        if i < 0:
            continue
        oid = int(offer_ids[i])
        result.append((oid, rank))
        rank += 1
    return result


def semantic_search_pgvector(session: Session, q: str, top_k: int = 50) -> List[Tuple[int, int]]:
    """
    Требует установленного pgvector extension и корректного типа в колонке embedding.
    """
    qv = embed_query(q).tolist()
    # Используем raw SQL, т.к. модель хранит embedding как TEXT (для совместимости без pgvector).
    rows = session.execute(
        text(
            "SELECT offer_id "
            "FROM search_index "
            "WHERE embedding_vec IS NOT NULL "
            "ORDER BY (embedding_vec <=> :qv) ASC "
            "LIMIT :k"
        ),
        {"qv": qv, "k": int(top_k)},
    ).all()
    return [(int(oid), idx + 1) for idx, (oid,) in enumerate(rows)]


def text_search(session: Session, q: str, top_k: int = 50) -> List[Tuple[int, int]]:
    tsq = func.websearch_to_tsquery("russian", q)
    rank = func.ts_rank_cd(SearchIndex.tsv, tsq)
    stmt = (
        select(SearchIndex.offer_id)
        .where(SearchIndex.tsv.op("@@")(tsq))
        .order_by(rank.desc())
        .limit(top_k)
    )
    rows = session.execute(stmt).all()
    return [(int(oid), idx + 1) for idx, (oid,) in enumerate(rows)]


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


def fetch_offers_in_order(session: Session, offer_ids: Sequence[int]) -> List[ProductOffer]:
    if not offer_ids:
        return []
    stmt = (
        select(ProductOffer)
        .options(
            selectinload(ProductOffer.company),
            selectinload(ProductOffer.category),
            selectinload(ProductOffer.product),
        )
        .where(ProductOffer.offer_id.in_(list(map(int, offer_ids))))
    )
    offers = session.execute(stmt).scalars().all()
    by_id = {int(o.offer_id): o for o in offers}
    return [by_id[oid] for oid in offer_ids if oid in by_id]


def build_search_index(
    session: Session,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    rebuild: bool = True,
) -> Dict[str, int]:
    """
    Собирает search_index + FAISS артефакты.
    """
    if rebuild:
        session.execute(delete(SearchIndex))
        session.commit()

    # Подтягиваем всё нужное
    stmt = (
        select(ProductOffer)
        .options(
            selectinload(ProductOffer.company),
            selectinload(ProductOffer.category),
            selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
        )
        .order_by(ProductOffer.offer_id)
    )
    offers = session.execute(stmt).scalars().all()

    texts: List[str] = []
    offer_ids: List[int] = []
    for o in offers:
        txt = build_offer_text(o)
        if not txt:
            continue
        texts.append(txt)
        offer_ids.append(int(o.offer_id))

    if not texts:
        return {"offers_indexed": 0}

    # Full-text: сохраняем tsvector (через SQL function)
    # Embeddings: считаем пакетно
    embeddings = embed_passages(texts, model_name=model_name)

    # Пытаемся включить pgvector extension (если нет прав — не упадём)
    pgvector_ok = True
    try:
        ensure_pgvector_extension(session)
        ensure_pgvector_column(session, dim=int(embeddings.shape[1]))
    except Exception:
        session.rollback()
        pgvector_ok = False

    # Запись в search_index
    for i, oid in enumerate(offer_ids):
        row = SearchIndex(
            offer_id=oid,
            text=texts[i],
            # to_tsvector лучше на стороне DB — сохраняем сразу выражением update ниже
            tsv=None,
            embedding=json.dumps(embeddings[i].tolist(), ensure_ascii=False),
        )
        session.add(row)
    session.commit()

    # tsvector считаем внутри БД для сохранённых строк
    session.execute(
        text(
            "UPDATE search_index "
            "SET tsv = to_tsvector('russian', coalesce(text,'')) "
            "WHERE tsv IS NULL"
        )
    )
    session.commit()

    # Если pgvector включён — заполняем embedding_vec
    if pgvector_ok:
        for i, oid in enumerate(offer_ids):
            session.execute(
                text("UPDATE search_index SET embedding_vec = :v WHERE offer_id = :oid"),
                {"v": embeddings[i].tolist(), "oid": int(oid)},
            )
        session.commit()

    # FAISS артефакты
    faiss_index, faiss_offer_ids = build_faiss_index(embeddings, offer_ids)
    save_faiss(faiss_index, faiss_offer_ids, _get_store_dir())

    return {
        "offers_indexed": len(offer_ids),
        "pgvector_ok": 1 if pgvector_ok else 0,
    }

