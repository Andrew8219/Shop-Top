"""
Скрипт дообучения (fine-tuning) bi-encoder для поиска по каталогу.

Что делает:
  1. Извлекает тексты товаров из search_index (или строит на лету из product_offers).
  2. Генерирует синтетические запросы из реальных поисковых логов (search_logs)
     и из свойств товаров (название, бренд, характеристики).
  3. Собирает пары (запрос, релевантный_текст, [hard_negatives]) через BM25-поиск.
  4. Дообучает SentenceTransformer (intfloat/multilingual-e5-small) с помощью
     MultipleNegativesRankingLoss (MNRL).
  5. Сохраняет модель в models/bi_encoder_finetuned/.

После дообучения укажите путь в .env:
  BI_ENCODER_MODEL=models/bi_encoder_finetuned

И измените DEFAULT_MODEL_NAME в search_index.py на новый путь, или задайте переменную:
  SEARCH_BI_ENCODER_MODEL=models/bi_encoder_finetuned

Запуск:
  python train_search_model.py [--epochs 3] [--batch 16] [--limit-offers 5000]

Требования: torch, sentence-transformers >= 3.0, rank-bm25 (pip install rank-bm25)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("train_search")

# ── DB ────────────────────────────────────────────────────────────────────────

def _get_session():
    from db import SessionLocal
    return SessionLocal()


# ── Сборка корпуса ────────────────────────────────────────────────────────────

def _load_offer_texts(session, limit: int) -> List[Tuple[int, str]]:
    """Возвращает [(offer_id, text)] из search_index. Если пусто — строит на лету."""
    from sqlalchemy import select, func
    from models import SearchIndex

    rows = session.execute(
        select(SearchIndex.offer_id, SearchIndex.text)
        .where(SearchIndex.text.isnot(None))
        .where(func.length(SearchIndex.text) > 10)
        .limit(limit)
    ).all()

    if rows:
        LOG.info("Загружено %d текстов из search_index", len(rows))
        return [(int(r[0]), r[1]) for r in rows]

    # Fallback: строим из product_offers
    LOG.warning("search_index пуст — строю тексты из product_offers (медленнее)")
    from models import ProductOffer, ProductProperty
    from sqlalchemy.orm import selectinload
    from search_index import build_offer_text

    offers = session.execute(
        select(ProductOffer)
        .options(
            selectinload(ProductOffer.company),
            selectinload(ProductOffer.category),
            selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
        )
        .limit(limit)
    ).scalars().all()

    result = []
    for o in offers:
        txt = build_offer_text(o)
        if txt:
            result.append((int(o.offer_id), txt))
    LOG.info("Построено %d текстов из product_offers", len(result))
    return result


def _load_search_logs(session, min_results: int = 1) -> List[str]:
    """Загружает реальные запросы пользователей из search_logs."""
    from sqlalchemy import select
    from models import SearchLog

    rows = session.execute(
        select(SearchLog.query)
        .where(SearchLog.results_count >= min_results)
        .where(SearchLog.query.isnot(None))
    ).scalars().all()
    seen = set()
    out = []
    for q in rows:
        q = (q or "").strip()
        if not q or len(q) < 2 or q in seen:
            continue
        seen.add(q)
        out.append(q)
    LOG.info("Загружено %d уникальных запросов из search_logs", len(out))
    return out


# ── Генерация синтетических запросов из текста товара ─────────────────────────

_BRAND_CAT_RE = re.compile(r"[|]")
_NUM_RE = re.compile(r"\d+")


def _synthetic_queries_for_text(text: str, *, max_queries: int = 4) -> List[str]:
    """
    Генерирует несколько вариантов запросов из текста товара.
    Стратегии:
      - первые 1-2 значимых слова (имитирует быстрый поиск)
      - числовые характеристики + слово-сосед (16 гб, 512 ssd, 15.6 дюйм)
      - полное название первой части (до первого |)
    """
    queries: List[str] = []
    parts = [p.strip() for p in _BRAND_CAT_RE.split(text) if p.strip()]

    if parts:
        # Часть 1: до 6 слов заголовка
        title_words = parts[0].split()
        if len(title_words) >= 2:
            queries.append(" ".join(title_words[:4]))
        if len(title_words) >= 4:
            queries.append(" ".join(title_words[:2]))

    # Часть 2: числовые характеристики (напр. "16 гб", "512 ssd")
    for part in parts[2:]:  # пропускаем название и бренд
        part_low = part.lower()
        # RAM
        m = re.search(r"(\d+)\s*(?:гб|gb)\s*(?:озу|ram|oперативн)?", part_low)
        if m:
            queries.append(f"{m.group(1)} гб озу")
        # SSD
        m = re.search(r"(\d+)\s*(?:гб|gb|тб|tb)\s*ssd", part_low)
        if m:
            queries.append(f"ssd {m.group(1)}")
        # Диагональ
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:дюйм|inch|\")", part_low)
        if m:
            val = m.group(1).replace(",", ".")
            queries.append(f"{val} дюйм")

    return list(dict.fromkeys(q for q in queries if len(q) >= 3))[:max_queries]


# ── BM25 hard negatives ───────────────────────────────────────────────────────

def _build_bm25(corpus_texts: List[str]):
    """Строит BM25-индекс по корпусу. Возвращает (bm25_index, tokenized_corpus)."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        LOG.warning("rank-bm25 не установлен (pip install rank-bm25). BM25 hard negatives отключены.")
        return None, None

    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    LOG.info("BM25-индекс построен для %d документов", len(tokenized))
    return bm25, tokenized


def _get_hard_negatives(
    bm25,
    query: str,
    relevant_idx: int,
    all_texts: List[str],
    *,
    n_negatives: int = 5,
    skip_top: int = 3,
) -> List[str]:
    """
    Возвращает hard negatives: документы, которые BM25 считает похожими,
    но не совпадают с релевантным. Это самые сложные для модели примеры.
    """
    if bm25 is None:
        return []
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    # Берём топ-20, пропускаем сам релевантный и первые skip_top (возможно релевантные)
    top_idxs = sorted(range(len(scores)), key=lambda i: -scores[i])
    negatives = []
    for idx in top_idxs:
        if idx == relevant_idx:
            continue
        if len(negatives) < skip_top:
            negatives.append(idx)  # временно буфер
            continue
        negatives.append(idx)
        if len(negatives) >= skip_top + n_negatives:
            break
    # Берём только n_negatives после пропуска
    hard_neg_idxs = negatives[skip_top: skip_top + n_negatives]
    return [all_texts[i] for i in hard_neg_idxs]


# ── Сборка датасета ───────────────────────────────────────────────────────────

def build_training_pairs(
    offer_texts: List[Tuple[int, str]],
    search_log_queries: List[str],
    *,
    n_hard_neg: int = 3,
    max_synth_per_offer: int = 3,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, List[str]]]]:
    """
    Возвращает:
      simple_pairs  — [(query, positive_text)] для MNRL без hard negatives
      triplets      — [(query, positive_text, [neg1, neg2, ...])] для MNRL с hard neg
    """
    corpus_texts = [t for _, t in offer_texts]
    id_to_idx: Dict[int, int] = {oid: i for i, (oid, _) in enumerate(offer_texts)}
    idx_to_text: Dict[int, str] = {i: t for i, (_, t) in enumerate(offer_texts)}

    LOG.info("Строю BM25-индекс...")
    bm25, _ = _build_bm25(corpus_texts)

    LOG.info("Генерирую синтетические запросы...")
    simple_pairs: List[Tuple[str, str]] = []
    triplets: List[Tuple[str, str, List[str]]] = []

    for idx, (oid, text) in enumerate(offer_texts):
        synth_qs = _synthetic_queries_for_text(text, max_queries=max_synth_per_offer)
        for q in synth_qs:
            hard_negs = _get_hard_negatives(bm25, q, idx, corpus_texts, n_negatives=n_hard_neg)
            if hard_negs:
                triplets.append((q, text, hard_negs))
            else:
                simple_pairs.append((q, text))

        if idx % 500 == 0 and idx > 0:
            LOG.info("  %d / %d офферов обработано", idx, len(offer_texts))

    # Реальные запросы из логов → FAISS/semantic search → топ-1 как positive
    # (используем embed_query + FAISS)
    if search_log_queries:
        LOG.info("Добавляю пары из %d реальных запросов...", len(search_log_queries))
        try:
            from search_index import embed_query, _get_faiss_cached
            faiss_index, faiss_ids = _get_faiss_cached()
            if faiss_index is not None and faiss_ids:
                import numpy as np
                for q in search_log_queries[:2000]:  # не больше 2000
                    qv = np.asarray([embed_query(q)], dtype=np.float32)
                    scores, idxs = faiss_index.search(qv, 3)
                    for rank, fi_raw in enumerate(idxs[0].tolist()):
                        # Pyright may infer tolist() elements as int | list; keep only scalar ids.
                        if isinstance(fi_raw, list):
                            continue
                        fi = int(fi_raw)
                        if fi < 0:
                            continue
                        oid_raw = faiss_ids[fi]
                        if isinstance(oid_raw, list):
                            continue
                        oid = int(oid_raw)
                        text_idx = id_to_idx.get(oid)
                        if text_idx is None:
                            continue
                        pos_text = idx_to_text[text_idx]
                        hard_negs = _get_hard_negatives(
                            bm25, q, text_idx, corpus_texts, n_negatives=n_hard_neg
                        )
                        if hard_negs:
                            triplets.append((q, pos_text, hard_negs))
                        else:
                            simple_pairs.append((q, pos_text))
                        break  # только топ-1
        except Exception as e:
            LOG.warning("Не удалось добавить пары из логов: %s", e)

    LOG.info(
        "Датасет: %d простых пар + %d триплетов = %d всего",
        len(simple_pairs),
        len(triplets),
        len(simple_pairs) + len(triplets),
    )
    return simple_pairs, triplets


# ── Обучение ──────────────────────────────────────────────────────────────────

def train(
    *,
    model_name: str = "intfloat/multilingual-e5-small",
    output_dir: str = "models/bi_encoder_finetuned",
    epochs: int = 3,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
    limit_offers: int = 5000,
    n_hard_neg: int = 3,
    eval_fraction: float = 0.05,
):
    import torch
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    LOG.info("Загружаю модель: %s", model_name)
    model = SentenceTransformer(model_name, device="cpu")

    session = _get_session()
    try:
        offer_texts = _load_offer_texts(session, limit=limit_offers)
        log_queries = _load_search_logs(session)
    finally:
        session.close()

    if not offer_texts:
        LOG.error("Нет данных для обучения. Запустите сначала индексацию (/admin/search/build).")
        sys.exit(1)

    random.shuffle(offer_texts)

    simple_pairs, triplets = build_training_pairs(
        offer_texts,
        log_queries,
        n_hard_neg=n_hard_neg,
    )

    # Формируем InputExample для MNRL.
    # MNRL принимает [anchor, positive, neg1, neg2, ...] или просто [anchor, positive].
    examples: List[InputExample] = []

    for q, pos in simple_pairs:
        examples.append(InputExample(texts=[f"query: {q}", f"passage: {pos}"]))

    for q, pos, negs in triplets:
        texts = [f"query: {q}", f"passage: {pos}"] + [f"passage: {n}" for n in negs]
        examples.append(InputExample(texts=texts))

    if not examples:
        LOG.error("Пустой датасет. Проверьте данные.")
        sys.exit(1)

    # Train / eval split
    random.shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_fraction))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    LOG.info(
        "Train: %d примеров, Eval: %d примеров, Epochs: %d, Batch: %d",
        len(train_examples),
        len(eval_examples),
        epochs,
        batch_size,
    )

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    loss_fn = MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_loader) * epochs * warmup_ratio)

    # Evaluator: InformationRetrievalEvaluator на eval_examples
    evaluator = None
    if eval_examples:
        try:
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

            eval_pairs = [(ex.texts[0], ex.texts[1], 1.0) for ex in eval_examples]
            e_queries = [p[0] for p in eval_pairs]
            e_docs = [p[1] for p in eval_pairs]
            e_scores = [p[2] for p in eval_pairs]
            evaluator = EmbeddingSimilarityEvaluator(e_queries, e_docs, e_scores)
        except Exception as e:
            LOG.warning("Не удалось создать evaluator: %s", e)

    LOG.info("Начинаю обучение...")
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=max(100, len(train_loader) // 2),
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    LOG.info("Модель сохранена в: %s", output_dir)
    LOG.info("")
    LOG.info("Следующие шаги:")
    LOG.info("  1. Добавьте в .env: SEARCH_BI_ENCODER_MODEL=%s", output_dir)
    LOG.info("  2. Перестройте поисковый индекс через /admin/search/build")
    LOG.info("  3. Перезапустите сервер")


# ── Генерация тренировочных данных для cross-encoder ─────────────────────────

def export_cross_encoder_data(
    output_jsonl: str = "data/cross_encoder_train.jsonl",
    *,
    limit_offers: int = 3000,
    n_negatives: int = 3,
):
    """
    Генерирует JSONL для дообучения cross-encoder (опционально).
    Формат: {"query": "...", "positive": "...", "negatives": ["...", ...]}
    """
    session = _get_session()
    try:
        offer_texts = _load_offer_texts(session, limit=limit_offers)
        log_queries = _load_search_logs(session)
    finally:
        session.close()

    random.shuffle(offer_texts)
    corpus_texts = [t for _, t in offer_texts]
    bm25, _ = _build_bm25(corpus_texts)

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        # Синтетические запросы
        for idx, (_, text) in enumerate(offer_texts):
            for q in _synthetic_queries_for_text(text, max_queries=2):
                negs = _get_hard_negatives(bm25, q, idx, corpus_texts, n_negatives=n_negatives)
                row = {"query": q, "positive": text, "negatives": negs}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        # Реальные запросы
        for q in log_queries[:1000]:
            # Без FAISS — просто BM25 top-1 как positive
            tokens = q.lower().split()
            if bm25:
                scores = bm25.get_scores(tokens)
                top_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
                pos_text = corpus_texts[top_idx]
                negs = _get_hard_negatives(bm25, q, top_idx, corpus_texts, n_negatives=n_negatives)
                row = {"query": q, "positive": pos_text, "negatives": negs}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    LOG.info("Экспортировано %d записей в %s", count, out_path)
    LOG.info("Для обучения cross-encoder используйте sentence-transformers CrossEncoder.train()")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune bi-encoder для поиска по каталогу MediaLux"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "export-ce"],
        default="train",
        help=(
            "train — дообучение bi-encoder (по умолчанию); "
            "export-ce — экспорт данных для cross-encoder"
        ),
    )
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--output", default="models/bi_encoder_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit-offers", type=int, default=5000,
                        help="Максимальное число товаров для обучения")
    parser.add_argument("--hard-neg", type=int, default=3,
                        help="Кол-во BM25 hard negatives на пример")
    parser.add_argument("--ce-output", default="data/cross_encoder_train.jsonl",
                        help="Путь к JSONL для cross-encoder (только при --mode export-ce)")
    args = parser.parse_args()

    if args.mode == "train":
        train(
            model_name=args.model,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch,
            limit_offers=args.limit_offers,
            n_hard_neg=args.hard_neg,
        )
    elif args.mode == "export-ce":
        export_cross_encoder_data(
            output_jsonl=args.ce_output,
            limit_offers=args.limit_offers,
            n_negatives=args.hard_neg,
        )


if __name__ == "__main__":
    main()
