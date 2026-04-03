"""Метрики витрины: заказы, визиты, конверсия."""

from __future__ import annotations

import logging
import random
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models import AnalyticsDailyVisit, ShopOrder

LOG = logging.getLogger(__name__)

# Пути, по которым считаем «визит в витрину» (GET, не API)
_VISIT_PATH_PREFIXES = ("/product/", "/search")
_VISIT_PATH_EXACT = frozenset({"/", "/cart", "/about"})


def should_count_path(path: str) -> bool:
    if path.startswith("/static") or path.startswith("/admin"):
        return False
    if path in _VISIT_PATH_EXACT:
        return True
    return any(path.startswith(p) for p in _VISIT_PATH_PREFIXES)


def record_daily_visit(request, response, db: Session) -> None:
    """Один инкремент визитов на календарный день на браузер (cookie)."""
    today_s = date.today().isoformat()
    if request.cookies.get("ml_shop_visit_day") == today_s:
        return

    today_d = date.today()
    row = db.get(AnalyticsDailyVisit, today_d)
    if row is None:
        db.add(AnalyticsDailyVisit(stat_date=today_d, visits=1))
    else:
        row.visits = int(row.visits or 0) + 1
    db.commit()

    response.set_cookie(
        key="ml_shop_visit_day",
        value=today_s,
        max_age=86400 * 2,
        path="/",
        samesite="lax",
        httponly=False,
        secure=False,
    )


def get_kpis(db: Session, days: int = 30) -> Dict[str, Any]:
    """Выручка, заказы, средний чек, визиты, конверсия за последние `days` дней."""
    days = max(1, min(int(days), 366))
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    since_d = date.today() - timedelta(days=days)

    revenue = db.scalar(
        select(func.coalesce(func.sum(ShopOrder.total_amount), 0)).where(
            ShopOrder.created_at >= since_dt
        )
    )
    orders_count = db.scalar(
        select(func.count()).select_from(ShopOrder).where(ShopOrder.created_at >= since_dt)
    )
    orders_count = int(orders_count or 0)
    rev_f = float(revenue or 0)

    visits = db.scalar(
        select(func.coalesce(func.sum(AnalyticsDailyVisit.visits), 0)).where(
            AnalyticsDailyVisit.stat_date >= since_d
        )
    )
    visits = int(visits or 0)

    avg_check: Optional[float] = None
    if orders_count > 0:
        avg_check = rev_f / orders_count

    conversion_percent: Optional[float] = None
    if visits > 0 and orders_count >= 0:
        conversion_percent = round(orders_count / visits * 100, 2)

    return {
        "period_days": days,
        "revenue_rub": rev_f,
        "orders_count": orders_count,
        "avg_check_rub": avg_check,
        "visits": visits,
        "conversion_percent": conversion_percent,
    }


def seed_demo_metrics(db: Session) -> None:
    """Добавляет тестовые заказы и визиты для демонстрации дашборда."""
    random.seed(42)
    now = datetime.now(timezone.utc)
    for _ in range(8):
        db.add(
            ShopOrder(
                total_amount=Decimal(random.randint(1200, 48000)) + Decimal("0.99"),
                created_at=now - timedelta(hours=random.randint(2, 720)),
            )
        )

    for days_ago in range(30):
        d = date.today() - timedelta(days=days_ago)
        extra = random.randint(25, 180)
        row = db.get(AnalyticsDailyVisit, d)
        if row is None:
            db.add(AnalyticsDailyVisit(stat_date=d, visits=extra))
        else:
            row.visits = int(row.visits or 0) + extra
    db.commit()
