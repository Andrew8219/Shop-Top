import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence

from db import SessionLocal
from normalization import (
    apply_mapping_by_category,
    build_mapping_by_category,
    evaluate_category_consistency,
    evaluate_filter_quality,
)
from search_index import rrf_merge, semantic_search_faiss, semantic_search_pgvector, text_search


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


def _evaluate_task5(dataset_path: Path) -> Dict[str, Dict[str, float]]:
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    db = SessionLocal()
    try:
        modes = {
            "semantic_only": [],
            "text_only": [],
            "hybrid_rrf": [],
        }

        ks = [5, 10, 20]
        totals = {
            mode: {f"Recall@{k}": 0.0 for k in ks} | {f"MRR@{k}": 0.0 for k in ks} | {f"nDCG@{k}": 0.0 for k in ks}
            for mode in modes
        }

        for item in rows:
            q = item["query"]
            rel = set(item["relevant_offer_ids"])

            sem = semantic_search_faiss(q, top_k=100)
            if not sem:
                try:
                    sem = semantic_search_pgvector(db, q, top_k=100)
                except Exception:
                    sem = []
            txt = text_search(db, q, top_k=100)
            hyb_ids = rrf_merge(sem, txt, k=60, limit=100)

            pred_sem = _ids_from_rank_pairs(sem)
            pred_txt = _ids_from_rank_pairs(txt)
            pred_hyb = hyb_ids

            per_mode = {
                "semantic_only": pred_sem,
                "text_only": pred_txt,
                "hybrid_rrf": pred_hyb,
            }

            for mode, pred in per_mode.items():
                for k in ks:
                    totals[mode][f"Recall@{k}"] += _recall_at_k(pred, rel, k)
                    totals[mode][f"MRR@{k}"] += _mrr_at_k(pred, rel, k)
                    totals[mode][f"nDCG@{k}"] += _ndcg_at_k(pred, rel, k)

        n = len(rows) or 1
        for mode in totals:
            for metric in totals[mode]:
                totals[mode][metric] = round(totals[mode][metric] / n, 4)
        return totals
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


def _to_markdown(task5: Dict[str, Dict[str, float]], task23: Dict[str, object]) -> str:
    lines: List[str] = []
    dry = bool(task23.get("dry_run"))
    lines.append("# Отчет по задачам 5, 2 и 3")
    if dry:
        lines.append("")
        lines.append(
            "> **Режим `--dry-run`:** к базе не применялась внутрикатегорийная нормализация (Task 3). "
            "Метрики Task 2 и состояние категорий — по текущим данным БД; колонки «после» для Task 3 совпадают с «до»."
        )
    lines.append("")
    lines.append("## Задача 5: Интеллектуальный поиск (сравнение режимов)")
    lines.append("")
    lines.append("| Режим | Recall@5 | Recall@10 | Recall@20 | MRR@5 | MRR@10 | MRR@20 | nDCG@5 | nDCG@10 | nDCG@20 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ["semantic_only", "text_only", "hybrid_rrf"]:
        m = task5[mode]
        lines.append(
            f"| {mode} | {m['Recall@5']} | {m['Recall@10']} | {m['Recall@20']} | "
            f"{m['MRR@5']} | {m['MRR@10']} | {m['MRR@20']} | "
            f"{m['nDCG@5']} | {m['nDCG@10']} | {m['nDCG@20']} |"
        )

    lines.append("")
    lines.append("## Задача 2: Качество набора характеристик для фильтров")
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
    lines.append("## Задача 3: Внутрикатегорийная нормализация")
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
    lines.append("Вывод для ВКР:")
    lines.append("- Для задачи 5 отчет содержит сравнение semantic-only, text-only и hybrid(RRF).")
    lines.append("- Для задачи 2 формализованы критерии качества набора фильтров.")
    lines.append("- Для задачи 3 добавлена явная category-aware нормализация и метрики по категориям.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка метрик ВКР (поиск, качество фильтров, нормализация по категориям)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Не применять category-aware нормализацию к БД (Task 3). "
            "Task 5 и Task 2 читают текущее состояние БД; отчёт сохраняется как vkr_metrics_dry_run.*"
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset = root / "evaluation" / "task5_queries.json"
    reports_dir = root / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    task5 = _evaluate_task5(dataset)
    task23 = _evaluate_task23(dry_run=args.dry_run)

    payload = {"task5": task5, "task23": task23}
    suffix = "_dry_run" if args.dry_run else ""
    json_path = reports_dir / f"vkr_metrics{suffix}.json"
    md_path = reports_dir / f"vkr_metrics{suffix}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(task5, task23), encoding="utf-8")
    print("Saved:")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
