from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from db import SessionLocal
from normalization import (
    apply_mapping_by_category,
    build_mapping_by_category,
    evaluate_category_consistency,
    evaluate_filter_quality,
)
from search_index import (
    filter_offer_ids_by_query_tokens,
    rrf_merge,
    semantic_search_faiss,
    semantic_search_pgvector,
    text_search,
)


def _recall_at_k(pred: Sequence[int], rel: set[int], k: int) -> float:
    if not rel:
        return 0.0
    hit = len(set(pred[:k]).intersection(rel))
    return hit / len(rel)


def _mrr_at_k(pred: Sequence[int], rel: set[int], k: int) -> float:
    for idx, oid in enumerate(pred[:k], start=1):
        if oid in rel:
            return 1.0 / idx
    return 0.0


def _ndcg_at_k(pred: Sequence[int], rel: set[int], k: int) -> float:
    import math

    dcg = 0.0
    for i, oid in enumerate(pred[:k], start=1):
        rel_i = 1.0 if oid in rel else 0.0
        if rel_i > 0:
            dcg += rel_i / math.log2(i + 1)
    ideal_hits = min(k, len(rel))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _ids_from_rank_pairs(pairs: Sequence[tuple[int, int]]) -> List[int]:
    return [oid for oid, _ in sorted(pairs, key=lambda x: x[1])]


def _evaluate_task5(dataset_path: Path, project_root: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Задача 7: метрики поиска по размеченному набору + детализация по запросам.

    Возвращает (усреднённые метрики по режимам, блок task7 для JSON/отчёта).
    """
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    db = SessionLocal()
    mode_names = (
        "semantic_only",
        "text_only",
        "hybrid_rrf_nofilter",
        "hybrid_rrf",
    )
    ks = [5, 10, 20]
    totals: Dict[str, Dict[str, float]] = {
        mode: {f"Recall@{k}": 0.0 for k in ks}
        | {f"MRR@{k}": 0.0 for k in ks}
        | {f"nDCG@{k}": 0.0 for k in ks}
        for mode in mode_names
    }
    per_query: List[Dict[str, Any]] = []

    try:
        for item in rows:
            q = item["query"]
            rel = set(int(x) for x in item["relevant_offer_ids"])

            sem = semantic_search_faiss(q, top_k=100)
            if not sem:
                try:
                    sem = semantic_search_pgvector(db, q, top_k=100)
                except Exception:
                    sem = []
            txt = text_search(db, q, top_k=100)
            hyb_raw = rrf_merge(sem, txt, k=60, limit=100)
            hyb_filtered = filter_offer_ids_by_query_tokens(db, list(hyb_raw), q)

            pred_sem = _ids_from_rank_pairs(sem)
            pred_txt = _ids_from_rank_pairs(txt)
            pred_hyb_raw = list(hyb_raw)
            pred_hyb = hyb_filtered

            per_mode_preds = {
                "semantic_only": pred_sem,
                "text_only": pred_txt,
                "hybrid_rrf_nofilter": pred_hyb_raw,
                "hybrid_rrf": pred_hyb,
            }

            row_metrics: Dict[str, Dict[str, float]] = {}
            for mode, pred in per_mode_preds.items():
                row_metrics[mode] = {}
                for k in ks:
                    totals[mode][f"Recall@{k}"] += _recall_at_k(pred, rel, k)
                    totals[mode][f"MRR@{k}"] += _mrr_at_k(pred, rel, k)
                    totals[mode][f"nDCG@{k}"] += _ndcg_at_k(pred, rel, k)
                    row_metrics[mode][f"Recall@{k}"] = round(_recall_at_k(pred, rel, k), 4)
                    row_metrics[mode][f"MRR@{k}"] = round(_mrr_at_k(pred, rel, k), 4)
                    row_metrics[mode][f"nDCG@{k}"] = round(_ndcg_at_k(pred, rel, k), 4)

            per_query.append(
                {
                    "id": item.get("id", ""),
                    "query": q,
                    "num_relevant": len(rel),
                    "metrics": row_metrics,
                }
            )

        n = len(rows) or 1
        for mode in totals:
            for metric in totals[mode]:
                totals[mode][metric] = round(totals[mode][metric] / n, 4)

        try:
            ds_rel = str(dataset_path.resolve().relative_to(project_root.resolve()))
        except ValueError:
            ds_rel = str(dataset_path)
        task7_block: Dict[str, Any] = {
            "title": "Задача 7 — оценка эффективности поиска",
            "methodology_file": "evaluation/task7_methodology.md",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "path": ds_rel.replace("\\", "/"),
                "num_queries": len(rows),
            },
            "experiment": {
                "embedding_model": "intfloat/multilingual-e5-small",
                "semantic_top_k": 100,
                "text_top_k": 100,
                "rrf_k": 60,
                "rrf_candidate_limit": 100,
                "post_filter_tokens": True,
                "modes": {
                    "semantic_only": "FAISS/pgvector, только семантика",
                    "text_only": "PostgreSQL tsvector + websearch_to_tsquery",
                    "hybrid_rrf_nofilter": "RRF(semantic, text), без постфильтра по токенам",
                    "hybrid_rrf": "RRF + постфильтр (все токены запроса в тексте оффера)",
                },
            },
            "per_query": per_query,
        }
        return totals, task7_block
    finally:
        db.close()


def _evaluate_task23(*, dry_run: bool = False) -> Dict[str, object]:
    db = SessionLocal()
    try:
        # Метрики качества набора фильтров (Task 2)
        task2 = evaluate_filter_quality(db)

        # Category-aware mapping (Task 3) + метрики
        before = evaluate_category_consistency(db)
        if dry_run:
            # Не меняем БД; метрики Task 3 — только по текущему состоянию.
            after = copy.deepcopy(before)
            updated = 0
        else:
            mapping = build_mapping_by_category(db)
            updated = apply_mapping_by_category(db, mapping)
            after = evaluate_category_consistency(db)

        return {
            "dry_run": dry_run,
            "task2_filter_quality": task2,
            "task3_category_before": before[:15],
            "task3_category_after": after[:15],
            "task3_values_updated": updated,
        }
    finally:
        db.close()


def _best_mode_for_metric(totals: Dict[str, Dict[str, float]], metric: str) -> str:
    return max(totals.keys(), key=lambda mode: totals[mode][metric])


def _task7_conclusion_lines(totals: Dict[str, Dict[str, float]]) -> List[str]:
    lines: List[str] = []
    for label, metric in [
        ("Recall@10", "Recall@10"),
        ("MRR@10", "MRR@10"),
        ("nDCG@10", "nDCG@10"),
    ]:
        best = _best_mode_for_metric(totals, metric)
        lines.append(
            f"- По усреднённому **{label}** лучший режим: **`{best}`** "
            f"({totals[best][metric]})."
        )
    h = totals["hybrid_rrf"]["Recall@10"]
    hn = totals["hybrid_rrf_nofilter"]["Recall@10"]
    if h > hn:
        lines.append(
            f"- Постфильтр по токенам **улучшил** средний Recall@10 гибрида: "
            f"{hn} → {h} (сравнение `hybrid_rrf_nofilter` и `hybrid_rrf`)."
        )
    elif h < hn:
        lines.append(
            f"- На этом наборе постфильтр **снизил** средний Recall@10: "
            f"{hn} → {h} — имеет смысл проверить разметку и состав запросов."
        )
    else:
        lines.append(
            "- Средний Recall@10 у `hybrid_rrf` и `hybrid_rrf_nofilter` совпал — постфильтр не меняет итог на агрегате."
        )
    sem = totals["semantic_only"]["Recall@10"]
    txt = totals["text_only"]["Recall@10"]
    if sem > txt:
        lines.append(
            f"- Семантика даёт более высокий средний Recall@10, чем чистый FTS ({sem} vs {txt})."
        )
    elif txt > sem:
        lines.append(
            f"- Полнотекст даёт более высокий средний Recall@10, чем чистая семантика ({txt} vs {sem})."
        )
    return lines


def _to_markdown(
    task5: Dict[str, Dict[str, float]],
    task23: Dict[str, object] | None,
    task7: Dict[str, Any],
) -> str:
    lines: List[str] = []
    dry = bool(task23 and task23.get("dry_run"))
    lines.append("# Отчёт ВКР: задача 7 (оценка) + метрики задач 2 и 3")
    lines.append("")
    lines.append("## Задача 7. Оценка эффективности поиска на тестовых данных")
    lines.append("")
    lines.append(
        "Методология (метрики, базовые линии, воспроизведение): см. файл "
        "`evaluation/task7_methodology.md`."
    )
    exp = task7.get("experiment", {})
    ds = task7.get("dataset", {})
    lines.append("")
    lines.append("### Параметры эксперимента")
    lines.append("")
    lines.append(f"- Датасет: `{ds.get('path', '')}` ({ds.get('num_queries', '?')} запросов).")
    lines.append(f"- Время формирования отчёта (UTC): `{task7.get('generated_at_utc', '')}`.")
    lines.append(f"- Модель эмбеддингов: `{exp.get('embedding_model', '')}`.")
    lines.append(
        f"- RRF: k={exp.get('rrf_k')}, до {exp.get('rrf_candidate_limit')} кандидатов; "
        f"топ семантики/FTS: {exp.get('semantic_top_k')}."
    )
    lines.append("")
    lines.append("### Сравнение базовых подходов (усреднение macro по запросам)")
    lines.append("")
    lines.append("| Режим | Recall@5 | Recall@10 | Recall@20 | MRR@5 | MRR@10 | MRR@20 | nDCG@5 | nDCG@10 | nDCG@20 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in [
        "semantic_only",
        "text_only",
        "hybrid_rrf_nofilter",
        "hybrid_rrf",
    ]:
        m = task5[mode]
        lines.append(
            f"| {mode} | {m['Recall@5']} | {m['Recall@10']} | {m['Recall@20']} | "
            f"{m['MRR@5']} | {m['MRR@10']} | {m['MRR@20']} | "
            f"{m['nDCG@5']} | {m['nDCG@10']} | {m['nDCG@20']} |"
        )
    lines.append("")
    lines.append("### Краткий сравнительный анализ (автоматически по таблице)")
    lines.append("")
    lines.extend(_task7_conclusion_lines(task5))
    lines.append("")
    lines.append(
        "Пометрики по каждому запросу — в JSON (`task7.per_query` в файле `vkr_metrics*.json`)."
    )

    lines.append("")
    lines.append("---")
    lines.append("")
    if task23 is None or task23.get("skipped"):
        lines.append("## Приложение A")
        lines.append("")
        lines.append(
            "*Раздел не формировался: запуск с флагом `--task7-only` (только оценка поиска).*"
        )
        return "\n".join(lines) + "\n"

    lines.append("## Приложение A. Задача 2 и 3 (качество данных и нормализация)")
    if dry:
        lines.append("")
        lines.append(
            "> **Режим `--dry-run`:** к базе не применялась внутрикатегорийная нормализация (Task 3). "
            "Метрики Task 2 и состояние категорий — по текущим данным БД; колонки «после» для Task 3 совпадают с «до»."
        )
    lines.append("")
    lines.append("### Задача 2: Качество набора характеристик для фильтров")
    lines.append("")
    t2 = task23["task2_filter_quality"]
    lines.append("| Метрика | Значение |")
    lines.append("|---|---:|")
    lines.append(f"| coverage | {t2['coverage']} |")
    lines.append(f"| duplicates_ratio | {t2['duplicates_ratio']} |")
    lines.append(f"| consistency | {t2['consistency']} |")
    lines.append("")
    lines.append("Критерии интерпретации:")
    lines.append("- coverage: чем выше, тем лучше (доля офферов с заполненными характеристиками)")
    lines.append("- duplicates_ratio: чем ниже, тем лучше (меньше дублей имён свойств)")
    lines.append("- consistency: чем выше, тем лучше (устойчивость/единообразие значений)")

    lines.append("")
    lines.append("### Задача 3: Внутрикатегорийная нормализация")
    lines.append("")
    if dry:
        lines.append(
            "Обновлено значений при category-aware нормализации: **не применялось (`--dry-run`)**"
        )
    else:
        lines.append(
            f"Обновлено значений при category-aware нормализации: **{task23['task3_values_updated']}**"
        )
    lines.append("")
    lines.append("| Категория | offers | avg_values_per_property (до) | avg_values_per_property (после) | weighted_mode_ratio (до) | weighted_mode_ratio (после) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    before = {x["category"]: x for x in task23["task3_category_before"]}
    after = {x["category"]: x for x in task23["task3_category_after"]}
    cats = sorted(set(before.keys()).intersection(after.keys()))
    for c in cats:
        b = before[c]
        a = after[c]
        lines.append(
            f"| {c} | {int(a['offers'])} | {b['avg_values_per_property']} | {a['avg_values_per_property']} | "
            f"{b['weighted_mode_ratio']} | {a['weighted_mode_ratio']} |"
        )

    lines.append("")
    lines.append("**Итог по приложению A:** метрики Task 2 и динамика Task 3 отражают качество "
                "набора характеристик и эффект внутрикатегорийной нормализации на текущей БД.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Задача 7: оценка поиска на размеченном наборе + отчёт по Task 2/3 "
            "(качество фильтров, нормализация по категориям)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Не применять category-aware нормализацию к БД (Task 3). "
            "Поиск и Task 2 читают текущее состояние БД; отчёт: vkr_metrics_dry_run.*"
        ),
    )
    parser.add_argument(
        "--task7-only",
        action="store_true",
        help="Только оценка поиска (Task 7): не считать Task 2/3 и не трогать нормализацию БД.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset = root / "evaluation" / "task5_queries.json"
    reports_dir = root / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    task5, task7 = _evaluate_task5(dataset, root)
    if args.task7_only:
        task23 = None
    else:
        task23 = _evaluate_task23(dry_run=args.dry_run)

    payload: Dict[str, Any] = {
        "task5": task5,
        "task7": task7,
        "task23": task23,
    }
    suffix = "_dry_run" if args.dry_run else ""
    json_path = reports_dir / f"vkr_metrics{suffix}.json"
    md_path = reports_dir / f"vkr_metrics{suffix}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(task5, task23, task7), encoding="utf-8")
    print("Saved:")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
