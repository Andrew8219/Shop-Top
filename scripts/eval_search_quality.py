"""
Оценка качества поиска по логам (search_logs).

Цели:
  - быстро увидеть % пустых выдач
  - топ запросов с нулевой выдачей
  - среднее/медиана размера выдачи

Запуск:
  DATABASE_URL=... python3 eval_search_quality.py --days 7 --limit 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from statistics import median

from sqlalchemy import desc, func, select

from db import SessionLocal
from models import SearchLog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=max(1, args.days))
    db = SessionLocal()
    try:
        total = (
            db.execute(
                select(func.count()).select_from(SearchLog).where(SearchLog.created_at >= since)
            ).scalar()
            or 0
        )
        if total == 0:
            print("no logs")
            return

        empty = (
            db.execute(
                select(func.count())
                .select_from(SearchLog)
                .where(SearchLog.created_at >= since, SearchLog.results_count <= 0)
            ).scalar()
            or 0
        )

        rows = db.execute(
            select(SearchLog.results_count).where(SearchLog.created_at >= since)
        ).scalars().all()
        rows_int = [int(x) for x in rows if x is not None]
        avg = (sum(rows_int) / len(rows_int)) if rows_int else 0.0
        med = median(rows_int) if rows_int else 0.0

        print(f"window_days: {args.days}")
        print(f"total_queries: {total}")
        print(f"empty_queries: {empty} ({empty/total*100:.1f}%)")
        print(f"avg_results: {avg:.1f}")
        print(f"median_results: {med:.1f}")
        print()

        print("top empty queries:")
        top = db.execute(
            select(SearchLog.query, func.count().label("n"))
            .where(SearchLog.created_at >= since, SearchLog.results_count <= 0)
            .group_by(SearchLog.query)
            .order_by(desc(func.count()))
            .limit(max(1, args.limit))
        ).all()
        for q, n in top:
            qn = (q or "").strip().replace("\n", " ")
            if len(qn) > 120:
                qn = qn[:117] + "..."
            print(f"{n:>4}  {qn}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

