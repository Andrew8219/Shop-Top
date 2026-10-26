from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import os
import re
import time
import threading
import uuid
import logging
from collections import defaultdict
from typing import List, Optional, Dict, Sequence, Set, Tuple
from urllib.parse import urlencode

from fastapi import (
    Depends,
    FastAPI,
    Form,
    Query,
    Request,
    status,
)
    # noqa
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request as StarletteRequest
from sqlalchemy import and_, asc, desc, func, nulls_last, select, text, update
from sqlalchemy.orm import Session, selectinload

from jose import JWTError, jwt
from passlib.context import CryptContext

from db import SessionLocal, Base, engine
from models import (
    Category,
    Company,
    ProductOffer,
    ProductProperty,
    SearchLog,
    User,
)
import analytics
from search_facets import (
    compute_search_facets,
    filter_offer_ids_by_gpu_value,
    filter_offer_ids_by_os_family,
    filter_offer_ids_by_screen_bucket,
    filter_offer_ids_by_ssd_bucket,
)
from search_index import (
    build_search_index,
    ensure_gin_index,
    fetch_offers_in_order,
    filter_offer_ids_by_query_tokens_soft,
    merge_search_rankings,
    rerank_offer_ids_cross_encoder,
    semantic_search_faiss,
    semantic_search_pgvector,
    text_search,
)
from search_smart import (
    filter_offer_ids_by_ram_gb,
    get_cached_suggest_strings,
    log_search_query,
    prepare_search_query,
    prioritize_by_battery,
    prioritize_by_display,
    prioritize_vacuum_offers_by_suction,
    suggest_completions,
)
from normalization import (
    build_mapping,
    apply_mapping,
    merge_duplicate_property_names,
    build_property_name_groups,
)
import shop_ui
import cart_cookie

# =====================
#   КОНФИГ MINI-LLM ДЛЯ ГРУПП
# =====================

# Путь к дообученной mini-LLM (HuggingFace-формат).
# Если модель не найдётся или нет torch/transformers,
# normalization.build_property_name_groups сам упадёт
# в fallback (кластеризация).
MINI_LLM_MODEL_DIR = "models/prop_groups_llm"
USE_LLM_FOR_GROUPS = True


# =====================
#   КЭШ ГРУПП СВОЙСТВ
# =====================

PROPERTY_GROUPS_CACHE: Optional[Dict[int, str]] = None


def get_property_groups() -> Dict[int, str]:
    """
    Просто отдаём уже посчитанные группы из кэша.

    Никакой тяжёлой логики здесь НЕ запускаем.
    Если кэш пуст (группы ещё не строили или они сброшены),
    возвращаем пустой словарь — все характеристики уйдут
    в «Прочие характеристики».
    """
    global PROPERTY_GROUPS_CACHE
    return PROPERTY_GROUPS_CACHE or {}


# =====================
#   ЛОГИРОВАНИЕ
# =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger(__name__)



# =====================
#   СТРУКТУРА ЗАДАЧ НОРМАЛИЗАЦИИ
# =====================

@dataclass
class NormalizationJob:
    id: str
    kind: str  # 'preview' | 'apply' | 'groups'
    status: str = "pending"  # pending|running|done|error
    progress: int = 0        # 0..100
    stage: str = ""          # collect/mapping/apply/merge/diffs/group/...
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None


JOBS: Dict[str, NormalizationJob] = {}
JOBS_LOCK = threading.Lock()


def create_job(kind: str) -> NormalizationJob:
    job_id = uuid.uuid4().hex
    job = NormalizationJob(id=job_id, kind=kind)
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def get_job(job_id: str) -> Optional[NormalizationJob]:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        for k, v in kwargs.items():
            setattr(job, k, v)


# =====================
#   RATE LIMITING (in-memory, без зависимостей)
# =====================

_RL_WINDOW = 60          # окно в секундах
_RL_MAX_ATTEMPTS = 10    # максимум попыток в окне на один IP

_rl_store: Dict[str, List[float]] = defaultdict(list)
_rl_lock = threading.Lock()


def _is_rate_limited(ip: str) -> bool:
    """Возвращает True, если IP превысил лимит запросов."""
    now = time.monotonic()
    with _rl_lock:
        hits = _rl_store[ip]
        _rl_store[ip] = [t for t in hits if now - t < _RL_WINDOW]
        if len(_rl_store[ip]) >= _RL_MAX_ATTEMPTS:
            return True
        _rl_store[ip].append(now)
        return False


_TRUST_PROXY: bool = os.getenv("TRUST_PROXY", "0").strip() == "1"


def _client_ip(request: Request) -> str:
    # X-Forwarded-For легко подделать, поэтому доверяем ему только за reverse-proxy.
    xff = request.headers.get("X-Forwarded-For") if _TRUST_PROXY else None
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# =====================
#   JWT / ПАРОЛИ
# =====================

SECRET_KEY: str = os.getenv("SECRET_KEY", "").strip()
if not SECRET_KEY:
    # Без заданного ключа генерируем случайный: приложение работает,
    # но сессии и корзина сбрасываются при каждом перезапуске.
    import secrets

    SECRET_KEY = secrets.token_hex(32)
    LOG.warning(
        "SECRET_KEY не задан — сгенерирован временный ключ. "
        "Задайте SECRET_KEY в .env: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 день

# True если приложение работает за HTTPS (задайте COOKIE_SECURE=1 в .env)
COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "0").strip() == "1"

# используем pbkdf2_sha256, чтобы избежать ограничений bcrypt
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# =====================
#   БАЗА / FASTAPI
# =====================


def _ensure_product_offers_price_rub() -> None:
    """create_all не добавляет колонки к уже существующим таблицам — только ALTER."""
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE product_offers "
                "ADD COLUMN IF NOT EXISTS price_rub NUMERIC(12, 2)"
            )
        )


def _backfill_product_offers_prices() -> None:
    """
    Старые строки после ADD COLUMN имеют price_rub = NULL — витрина показывает «по запросу».
    1) Пытаемся взять число из характеристик вроде «Цена».
    2) Если в таблице по-прежнему нет ни одной цены — заполняем детерминированно по offer_id (демо).
       Отключить: MEDIALUX_SKIP_DEMO_PRICE_BACKFILL=1
    """
    if engine.dialect.name != "postgresql":
        return
    skip_demo = os.getenv("MEDIALUX_SKIP_DEMO_PRICE_BACKFILL", "").strip() == "1"
    db = SessionLocal()
    try:
        total = db.execute(select(func.count()).select_from(ProductOffer)).scalar() or 0
        if total == 0:
            return

        stmt_null = (
            select(ProductOffer)
            .where(ProductOffer.price_rub.is_(None))
            .options(
                selectinload(ProductOffer.properties).selectinload(
                    ProductProperty.property_name
                ),
            )
        )
        for offer in db.execute(stmt_null).scalars().unique().all():
            for p in offer.properties or []:
                if not shop_ui.is_price_property(p):
                    continue
                n = shop_ui.parse_price_rub_numeric((p.value or "").strip())
                if n is not None:
                    offer.price_rub = n
                    break
        db.commit()

        priced = (
            db.execute(
                select(func.count())
                .select_from(ProductOffer)
                .where(ProductOffer.price_rub.isnot(None))
            ).scalar()
            or 0
        )
        if priced > 0 or skip_demo:
            return

        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE product_offers SET price_rub = "
                    "(500 + MOD(offer_id * 7919, 95000))::numeric "
                    "WHERE price_rub IS NULL"
                )
            )
        LOG.info(
            "Заполнены демо-цены price_rub (в таблице не было ни одной цены). "
            "Отключение: MEDIALUX_SKIP_DEMO_PRICE_BACKFILL=1"
        )
    except Exception:
        LOG.exception("Ошибка backfill price_rub")
        db.rollback()
    finally:
        db.close()


Base.metadata.create_all(bind=engine)
_ensure_product_offers_price_rub()
_backfill_product_offers_prices()


def _try_ensure_gin_index() -> None:
    """Создаёт GIN-индекс на search_index.tsv при старте (если таблица уже заполнена)."""
    db = SessionLocal()
    try:
        ensure_gin_index(db)
    except Exception:
        LOG.warning("GIN-индекс при старте не создан (возможно, таблица пуста)", exc_info=False)
        db.rollback()
    finally:
        db.close()


_try_ensure_gin_index()

app = FastAPI(title="Shop-Top")


@app.middleware("http")
async def shop_visit_tracking_middleware(request: StarletteRequest, call_next):
    response = await call_next(request)
    if request.method != "GET" or not analytics.should_count_path(request.url.path):
        return response
    db = SessionLocal()
    try:
        analytics.record_daily_visit(request, response, db)
    except Exception:
        LOG.exception("Ошибка учёта визита")
        db.rollback()
    finally:
        db.close()
    return response


@app.middleware("http")
async def cart_state_middleware(request: StarletteRequest, call_next):
    raw = request.cookies.get(cart_cookie.COOKIE_NAME)
    cart = cart_cookie.parse_cart_cookie(raw, SECRET_KEY)
    request.state.cart = cart
    request.state.cart_count = sum(cart.values())
    return await call_next(request)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals["shop_price"] = shop_ui.get_offer_price_display
templates.env.globals["shop_teaser"] = shop_ui.get_offer_teaser_lines
templates.env.globals["shop_initials"] = shop_ui.offer_card_initials
templates.env.globals["shop_is_price_prop"] = shop_ui.is_price_property
templates.env.globals["shop_price_rub"] = shop_ui.get_offer_price_rub


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



def _enrich_search_facet_hrefs(
    *,
    q: str,
    company_id: Optional[int],
    category_id: Optional[int],
    price_min: Optional[float],
    price_max: Optional[float],
    sort: str,
    ram_gb: Optional[int],
    screen_bucket: Optional[str],
    ssd_gb: Optional[int],
    os_family: Optional[str],
    gpu_value: Optional[str],
    facets: dict,
    search_sl: Optional[int] = None,
) -> None:
    """Добавляет ключ href к элементам фасетов (in-place)."""
    cat = category_id if category_id and category_id > 0 else None
    comp = company_id if company_id and company_id > 0 else None
    for row in facets.get("companies") or []:
        row["href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=int(row["id"]),
            category_id=cat,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_gb,
            screen_bucket=screen_bucket,
            ssd_gb=ssd_gb,
            os_family=os_family,
            gpu=gpu_value,
            search_sl=search_sl,
        )
    for row in facets.get("categories") or []:
        row["href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=comp,
            category_id=int(row["id"]),
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_gb,
            screen_bucket=screen_bucket,
            ssd_gb=ssd_gb,
            os_family=os_family,
            gpu=gpu_value,
            search_sl=search_sl,
        )
    for row in facets.get("price_buckets") or []:
        row["href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=comp,
            category_id=cat,
            price_min=row.get("price_min"),
            price_max=row.get("price_max"),
            sort=sort,
            ram_gb=ram_gb,
            screen_bucket=screen_bucket,
            ssd_gb=ssd_gb,
            os_family=os_family,
            gpu=gpu_value,
            search_sl=search_sl,
        )
    for row in facets.get("ram_gb") or []:
        row["href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=comp,
            category_id=cat,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=int(row["gb"]),
            screen_bucket=screen_bucket,
            ssd_gb=ssd_gb,
            os_family=os_family,
            gpu=gpu_value,
            search_sl=search_sl,
        )
    for row in facets.get("screen_buckets") or []:
        row["href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=comp,
            category_id=cat,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_gb,
            screen_bucket=str(row["key"]),
            ssd_gb=ssd_gb,
            os_family=os_family,
            gpu=gpu_value,
            search_sl=search_sl,
        )
    for row in facets.get("ssd_buckets") or []:
        row["href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=comp,
            category_id=cat,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_gb,
            screen_bucket=screen_bucket,
            ssd_gb=int(row["gb"]),
            os_family=os_family,
            gpu=gpu_value,
            search_sl=search_sl,
        )
    for row in facets.get("os_families") or []:
        row["href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=comp,
            category_id=cat,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_gb,
            screen_bucket=screen_bucket,
            ssd_gb=ssd_gb,
            os_family=str(row["key"]),
            gpu=gpu_value,
            search_sl=search_sl,
        )


def _cart_from_request(request: Request) -> Dict[int, int]:
    cart = getattr(request.state, "cart", None)
    if isinstance(cart, dict):
        return dict(cart)
    raw = request.cookies.get(cart_cookie.COOKIE_NAME)
    return cart_cookie.parse_cart_cookie(raw, SECRET_KEY)


def _safe_next_path(nxt: Optional[str], default: str = "/cart") -> str:
    if not nxt or not isinstance(nxt, str):
        return default
    p = nxt.strip()
    if not p.startswith("/") or p.startswith("//"):
        return default
    return p


# =====================
#   УТИЛИТЫ ДЛЯ ЮЗЕРОВ
# =====================

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    return db.execute(stmt).scalars().first()


async def get_optional_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Пытаемся вытащить пользователя из JWT в cookie.
    Если не получилось — возвращаем None.
    """
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None

    user = get_user_by_username(db, username)
    if not user or not user.is_active:
        return None
    return user


# =====================
#   ФОНОВЫЕ ЗАДАЧИ НОРМАЛИЗАЦИИ
# =====================

def _preview_progress_cb(job_id: str, stage: str, current: int, total: int) -> None:
    """
    Прогресс для предпросмотра:
      collect  → 0–40%
      mapping  → 40–80%
      diffs    → считается отдельно в run_preview_job (80–100)
    """
    if total <= 0:
        return

    if stage == "collect":
        base, span = 0, 40
    elif stage == "mapping":
        base, span = 40, 40
    else:
        base, span = 0, 0

    pct = base + int(current / total * span) if span > 0 else 0
    if pct > 99:
        pct = 99
    update_job(job_id, progress=pct, stage=stage)


def run_preview_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    update_job(job_id, status="running", stage="init", message="Сбор статистики по значениям")

    db = SessionLocal()
    try:
        mapping = build_mapping(
            db,
            progress_cb=lambda st, c, t: _preview_progress_cb(job_id, st, c, t),
        )

        # формируем diff-ы — 80–100%
        total_props = len(mapping) or 1
        diffs: List[dict] = []

        for idx, (prop_name, raw_map) in enumerate(mapping.items(), start=1):
            for raw, canon in raw_map.items():
                if raw != canon:
                    diffs.append(
                        {"prop_name": prop_name, "raw": raw, "canonical": canon}
                    )
            pct = 80 + int(idx / total_props * 20)
            if pct > 99:
                pct = 99
            update_job(
                job_id,
                progress=pct,
                stage="diffs",
                message="Формируем список отличий",
            )

        total_diffs = len(diffs)
        result = {
            "diffs": diffs[:500],
            "total_diffs": total_diffs,
        }
        update_job(
            job_id,
            status="done",
            progress=100,
            stage="done",
            result=result,
            message=f"Найдено отличий: {total_diffs}",
        )
    except Exception as e:
        LOG.exception("Ошибка в задаче предпросмотра %s", job_id)
        update_job(
            job_id,
            status="error",
            error=str(e),
            message="Ошибка при построении предпросмотра",
        )
    finally:
        db.close()


def _apply_progress_cb(job_id: str, stage: str, current: int, total: int) -> None:
    """
    Прогресс для полной нормализации:
      collect  → 0–20%
      mapping  → 20–40%
      apply    → 40–85%
      merge    → 85–100%
    """
    if total <= 0:
        return

    if stage == "collect":
        base, span = 0, 20
    elif stage == "mapping":
        base, span = 20, 20
    elif stage == "apply":
        base, span = 40, 45   # до ~85
    elif stage == "merge":
        base, span = 85, 15   # 85–100
    else:
        base, span = 0, 0

    pct = base + int(current / total * span) if span > 0 else 0
    if pct > 99:
        pct = 99
    update_job(job_id, progress=pct, stage=stage)


def run_apply_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    update_job(job_id, status="running", stage="init", message="Старт нормализации")

    db = SessionLocal()
    try:
        mapping = build_mapping(
            db,
            progress_cb=lambda st, c, t: _apply_progress_cb(job_id, st, c, t),
        )
        updated_values = apply_mapping(
            db,
            mapping,
            progress_cb=lambda st, c, t: _apply_progress_cb(job_id, st, c, t),
        )
        merged_pp, removed_pn = merge_duplicate_property_names(
            db,
            progress_cb=lambda st, c, t: _apply_progress_cb(job_id, st, c, t),
        )

        # После изменения имён свойств старые группы уже невалидны
        global PROPERTY_GROUPS_CACHE
        PROPERTY_GROUPS_CACHE = None

        result = {
            "updated_values": updated_values,
            "merged_properties": merged_pp,
            "removed_property_names": removed_pn,
        }
        update_job(
            job_id,
            status="done",
            progress=100,
            stage="done",
            result=result,
            message=(
                f"Обновлено значений: {updated_values}, "
                f"объединено строк свойств: {merged_pp}"
            ),
        )
    except Exception as e:
        LOG.exception("Ошибка в задаче применения нормализации %s", job_id)
        update_job(
            job_id,
            status="error",
            error=str(e),
            message="Ошибка при применении нормализации",
        )
    finally:
        db.close()


# ======== НОВОЕ: задача построения групп ========

def _groups_progress_cb(job_id: str, stage: str, current: int, total: int) -> None:
    if total <= 0:
        return
    # для групп просто считаем прогресс 0–100 по числу обработанных свойств
    pct = int(current / total * 100)
    if pct > 99:
        pct = 99
    update_job(job_id, progress=pct, stage=stage or "group")


def run_groups_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    update_job(job_id, status="running", stage="init", message="Построение групп характеристик")

    db = SessionLocal()
    try:
        groups = build_property_name_groups(
            db,
            distance_threshold=0.6,
            progress_cb=lambda st, c, t: _groups_progress_cb(job_id, st, c, t),
            llm_model_dir=MINI_LLM_MODEL_DIR if USE_LLM_FOR_GROUPS else None,
            use_llm=USE_LLM_FOR_GROUPS,
        )

        global PROPERTY_GROUPS_CACHE
        PROPERTY_GROUPS_CACHE = groups

        result = {"groups_count": len(groups)}
        update_job(
            job_id,
            status="done",
            progress=100,
            stage="done",
            result=result,
            message=f"Построено групп: {len(groups)}",
        )
    except Exception as e:
        LOG.exception("Ошибка в задаче построения групп %s", job_id)
        update_job(
            job_id,
            status="error",
            error=str(e),
            message="Ошибка при построении групп характеристик",
        )
    finally:
        db.close()


# =====================
#   SEARCH INDEX (ADMIN)
# =====================


def run_search_index_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    update_job(job_id, status="running", stage="init", message="Сборка поискового индекса")

    db = SessionLocal()
    try:
        def _search_progress(stage: str, current: int, total: int) -> None:
            if total <= 0:
                return
            stage_norm = stage if stage in ("collect", "embed", "faiss") else "collect"
            if stage_norm == "collect":
                base, span, msg = 0, 35, "Читаем товары из БД"
            elif stage_norm == "embed":
                base, span, msg = 35, 50, "Считаем эмбеддинги"
            else:
                base, span, msg = 85, 15, "Собираем FAISS индекс"
            pct = base + int(current / total * span)
            if pct > 99:
                pct = 99
            update_job(job_id, progress=pct, stage=stage_norm, message=msg)

        update_job(job_id, progress=3, stage="collect", message="Старт индексации")
        result = build_search_index(db, rebuild=True, progress_cb=_search_progress)
        update_job(
            job_id,
            status="done",
            progress=100,
            stage="done",
            result=result,
            message=f"Индекс собран. Офферов: {result.get('offers_indexed', 0)}",
        )
    except Exception as e:
        LOG.exception("Ошибка сборки поискового индекса %s", job_id)
        update_job(
            job_id,
            status="error",
            error=str(e),
            message="Ошибка при сборке поискового индекса",
        )
    finally:
        db.close()


# =====================
#   КАТАЛОГ: ФИЛЬТРЫ / СОРТИРОВКА
# =====================


def _normalize_catalog_sort(sort: str) -> str:
    if sort in ("price_asc", "price_desc"):
        return sort
    return ""


def _parse_query_optional_int_ge1(raw: Optional[str]) -> Optional[int]:
    """HTML-форма шлёт пустые поля как \"\"; FastAPI иначе падает на int."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        v = int(text)
    except ValueError:
        return None
    return v if v >= 1 else None


def _parse_query_optional_float_nonneg(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    try:
        v = float(text.replace(",", "."))
    except ValueError:
        return None
    return v if v >= 0 else None


def _normalize_price_bounds(
    price_min: Optional[float], price_max: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    if price_min is not None and price_max is not None and price_min > price_max:
        return price_max, price_min
    return price_min, price_max


def _parse_screen_bucket_q(raw: Optional[str]) -> Optional[str]:
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    return s if s in ("small", "medium", "large") else None


def _screen_bucket_from_diagonal(diagonal_inch: Optional[float]) -> Optional[str]:
    if diagonal_inch is None:
        return None
    if diagonal_inch <= 13.0:
        return "small"
    if diagonal_inch <= 15.0:
        return "medium"
    return "large"


def _parse_ssd_facet_gb(raw: Optional[str]) -> Optional[int]:
    if raw is None or not str(raw).strip():
        return None
    try:
        v = int(str(raw).strip())
    except ValueError:
        return None
    return v if v in (256, 512, 1024, 2048) else None


def _parse_os_family_q(raw: Optional[str]) -> Optional[str]:
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    return s if s in ("windows", "macos", "android", "none") else None


def _parse_gpu_q(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return s[:80]


def _catalog_filter_clauses(
    company_id: Optional[int],
    category_id: Optional[int],
    price_min: Optional[float],
    price_max: Optional[float],
) -> List:
    clauses = []
    if company_id is not None and company_id > 0:
        clauses.append(ProductOffer.company_id == company_id)
    if category_id is not None and category_id > 0:
        clauses.append(ProductOffer.category_id == category_id)
    if price_min is not None or price_max is not None:
        clauses.append(ProductOffer.price_rub.isnot(None))
    if price_min is not None:
        clauses.append(ProductOffer.price_rub >= price_min)
    if price_max is not None:
        clauses.append(ProductOffer.price_rub <= price_max)
    return clauses


def _catalog_order_by(sort: str):
    if sort == "price_asc":
        return (
            nulls_last(asc(ProductOffer.price_rub)),
            asc(ProductOffer.offer_id),
        )
    if sort == "price_desc":
        return (
            nulls_last(desc(ProductOffer.price_rub)),
            asc(ProductOffer.offer_id),
        )
    return (asc(ProductOffer.offer_id),)


def _listing_qs_no_page(
    *,
    is_search: bool,
    q: str,
    company_id: Optional[int],
    category_id: Optional[int],
    price_min: Optional[float],
    price_max: Optional[float],
    sort: str,
    ram_gb: Optional[int] = None,
    screen_bucket: Optional[str] = None,
    ssd_gb: Optional[int] = None,
    os_family: Optional[str] = None,
    gpu: Optional[str] = None,
    search_sl: Optional[int] = None,
) -> str:
    parts: Dict[str, object] = {}
    if is_search and q:
        parts["q"] = q
    if company_id is not None and company_id > 0:
        parts["company_id"] = company_id
    if category_id is not None and category_id > 0:
        parts["category_id"] = category_id
    if price_min is not None:
        parts["price_min"] = price_min
    if price_max is not None:
        parts["price_max"] = price_max
    if sort:
        parts["sort"] = sort
    if ram_gb is not None and ram_gb > 0:
        parts["ram_gb"] = int(ram_gb)
    if screen_bucket:
        parts["screen"] = str(screen_bucket)
    if ssd_gb is not None and ssd_gb > 0:
        parts["ssd_gb"] = int(ssd_gb)
    if os_family:
        parts["os"] = str(os_family)
    if gpu:
        parts["gpu"] = str(gpu)
    if search_sl is not None and search_sl > 0:
        parts["sl"] = int(search_sl)
    return urlencode(parts, doseq=True)


def _dedupe_offer_ids_preserve_order(ids: Sequence[int]) -> List[int]:
    """Один offer_id — одна позиция в выдаче (иначе len > числа карточек)."""
    seen: Set[int] = set()
    out: List[int] = []
    for raw in ids:
        x = int(raw)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _materialize_search_offer_ids(db: Session, ids: Sequence[int]) -> List[int]:
    """
    Уникальные id в исходном порядке, только существующие в product_offers.
    Иначе в списке могут быть дубликаты из ранжирования или «фантомы» из индекса —
    тогда «Показано N из M», а карточек меньше.
    """
    u = _dedupe_offer_ids_preserve_order(ids)
    if not u:
        return []
    found = set(
        db.execute(
            select(ProductOffer.offer_id).where(ProductOffer.offer_id.in_(u))
        ).scalars().all()
    )
    return [x for x in u if x in found]


def _filter_search_offer_ids(
    db: Session,
    merged_ids: List[int],
    company_id: Optional[int],
    category_id: Optional[int],
    price_min: Optional[float],
    price_max: Optional[float],
) -> List[int]:
    if not merged_ids:
        return []
    extra = _catalog_filter_clauses(company_id, category_id, price_min, price_max)
    if not extra:
        return list(merged_ids)
    stmt = select(ProductOffer.offer_id).where(
        and_(ProductOffer.offer_id.in_(list(map(int, merged_ids))), *extra)
    )
    allowed = set(db.execute(stmt).scalars().all())
    return [oid for oid in merged_ids if oid in allowed]


def _sort_offer_ids_by_price(db: Session, ids: List[int], sort: str) -> List[int]:
    if sort not in ("price_asc", "price_desc") or not ids:
        return ids
    stmt = select(ProductOffer.offer_id, ProductOffer.price_rub).where(
        ProductOffer.offer_id.in_(list(map(int, ids)))
    )
    price_map = {int(r[0]): r[1] for r in db.execute(stmt).all()}

    def key_asc(oid: int):
        p = price_map.get(int(oid))
        if p is None:
            return (True, 0.0)
        return (False, float(p))

    def key_desc(oid: int):
        p = price_map.get(int(oid))
        if p is None:
            return (True, 0.0)
        return (False, -float(p))

    return sorted(ids, key=key_asc if sort == "price_asc" else key_desc)


_catalog_cache: Optional[Tuple[List[Company], List[Category]]] = None
_catalog_cache_ts: float = 0.0
_CATALOG_CACHE_TTL = 300  # секунд


def _load_catalog_filter_lookups(db: Session) -> Tuple[List[Company], List[Category]]:
    global _catalog_cache, _catalog_cache_ts
    now = time.monotonic()
    if _catalog_cache is not None and (now - _catalog_cache_ts) < _CATALOG_CACHE_TTL:
        return _catalog_cache
    companies = list(db.execute(select(Company).order_by(Company.name)).scalars().all())
    categories = list(db.execute(select(Category).order_by(Category.name)).scalars().all())
    _catalog_cache = (companies, categories)
    _catalog_cache_ts = now
    return _catalog_cache


# =====================
#   МАГАЗИН (публичная часть)
# =====================

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
    page: int = Query(1, ge=1),
    company_id_q: Optional[str] = Query(None, alias="company_id"),
    category_id_q: Optional[str] = Query(None, alias="category_id"),
    price_min_q: Optional[str] = Query(None, alias="price_min"),
    price_max_q: Optional[str] = Query(None, alias="price_max"),
    sort: str = Query(""),
):
    page_size = 100
    sort = _normalize_catalog_sort(sort)
    company_id = _parse_query_optional_int_ge1(company_id_q)
    category_id = _parse_query_optional_int_ge1(category_id_q)
    price_min, price_max = _normalize_price_bounds(
        _parse_query_optional_float_nonneg(price_min_q),
        _parse_query_optional_float_nonneg(price_max_q),
    )

    clauses = _catalog_filter_clauses(company_id, category_id, price_min, price_max)
    where = and_(*clauses) if clauses else None

    count_stmt = select(func.count()).select_from(ProductOffer)
    if where is not None:
        count_stmt = count_stmt.where(where)
    total_offers = db.execute(count_stmt).scalar() or 0

    companies, categories = _load_catalog_filter_lookups(db)

    if total_offers == 0:
        offers = []
        total_pages = 1
        from_item = 0
        to_item = 0
    else:
        total_pages = (total_offers + page_size - 1) // page_size
        if page > total_pages:
            page = total_pages

        offset = (page - 1) * page_size

        stmt = (
            select(ProductOffer)
            .options(
                selectinload(ProductOffer.company),
                selectinload(ProductOffer.category),
                selectinload(ProductOffer.product),
                selectinload(ProductOffer.properties).selectinload(ProductProperty.property_name),
            )
        )
        if where is not None:
            stmt = stmt.where(where)
        stmt = stmt.order_by(*_catalog_order_by(sort)).offset(offset).limit(page_size)
        offers = db.execute(stmt).scalars().all()

        from_item = offset + 1
        to_item = offset + len(offers)

    listing_qs_no_page = _listing_qs_no_page(
        is_search=False,
        q="",
        company_id=company_id,
        category_id=category_id,
        price_min=price_min,
        price_max=price_max,
        sort=sort,
    )
    filters_active = bool(
        (company_id and company_id > 0)
        or (category_id and category_id > 0)
        or price_min is not None
        or price_max is not None
        or sort
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "offers": offers,
            "user": current_user,
            "page": page,
            "total_pages": total_pages,
            "total_offers": total_offers,
            "from_item": from_item,
            "to_item": to_item,
            "q": "",
            "effective_search_q": "",
            "query_refined": False,
            "show_effective_search_hint": False,
            "fuzzy_corrected": False,
            "fuzzy_corrected_query": None,
            "parsed_ram_gb": None,
            "is_search": False,
            "companies": companies,
            "categories": categories,
            "filter_company_id": company_id or 0,
            "filter_category_id": category_id or 0,
            "filter_price_min": price_min,
            "filter_price_max": price_max,
            "filter_sort": sort,
            "listing_qs_no_page": listing_qs_no_page,
            "filters_active": filters_active,
            "search_facets": None,
            "search_log_id": None,
            "search_ab_mode": None,
            "filter_ram_gb": None,
            "filter_screen_bucket": None,
            "filter_ssd_gb": None,
            "filter_os_family": None,
            "filter_gpu_value": None,
        },
    )


@app.get("/product/{offer_id}", response_class=HTMLResponse)
async def product_detail(
    offer_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
    sl: Optional[int] = Query(None, ge=1, description="ID строки search_logs для учёта клика"),
):
    stmt = (
        select(ProductOffer)
        .options(
            selectinload(ProductOffer.product),
            selectinload(ProductOffer.company),
            selectinload(ProductOffer.category),
            selectinload(ProductOffer.properties)
                .selectinload(ProductProperty.property_name),
        )
        .where(ProductOffer.offer_id == offer_id)
    )
    offer: ProductOffer | None = db.execute(stmt).scalars().first()
    if offer is None:
        return templates.TemplateResponse(
            request=request,
            name="errors/product_not_found.html",
            status_code=404,
            context={
                "user": current_user,
                "offer_id": offer_id,
            },
        )

    if sl is not None:
        try:
            db.execute(
                update(SearchLog)
                .where(SearchLog.id == int(sl))
                .where(SearchLog.clicked_offer_id.is_(None))
                .values(clicked_offer_id=int(offer_id))
            )
            db.commit()
        except Exception:
            LOG.exception("SearchLog click update failed (sl=%s)", sl)
            db.rollback()

    props = sorted(
        offer.properties,
        key=lambda p: (p.property_name.name if p.property_name else "", p.property_name_id),
    )

    groups = get_property_groups()

    group_map: Dict[str, List[ProductProperty]] = {}
    for prop in props:
        if prop.property_name:
            group_name = groups.get(prop.property_name_id, "Прочие характеристики")
        else:
            group_name = "Прочие характеристики"
        group_map.setdefault(group_name, []).append(prop)

    grouped_props: List[tuple[str, List[ProductProperty]]] = sorted(
        group_map.items(),
        key=lambda x: x[0],
    )

    key_attrs = shop_ui.get_offer_teaser_lines(offer, limit=5)

    return templates.TemplateResponse(
        request=request,
        name="product_detail.html",
        context={
            "offer": offer,
            "props": props,
            "grouped_props": grouped_props,
            "user": current_user,
            "key_attrs": key_attrs,
            "price_display": shop_ui.get_offer_price_display(offer),
            "color_display": shop_ui.get_offer_color_display(offer),
        },
    )


@app.post("/cart/add")
async def cart_add_item(
    request: Request,
    offer_id: int = Form(...),
    quantity: int = Form(1),
    redirect: Optional[str] = Form(None),
):
    cart = _cart_from_request(request)
    cart = cart_cookie.cart_add(cart, offer_id, quantity)
    dest = _safe_next_path(redirect, "/cart")
    resp = RedirectResponse(url=dest, status_code=status.HTTP_303_SEE_OTHER)
    cart_cookie.attach_cart_to_response(resp, cart, SECRET_KEY, secure=COOKIE_SECURE)
    return resp


@app.post("/cart/remove")
async def cart_remove_item(
    request: Request,
    offer_id: int = Form(...),
):
    cart = _cart_from_request(request)
    cart = cart_cookie.cart_remove(cart, offer_id)
    resp = RedirectResponse(url="/cart", status_code=status.HTTP_303_SEE_OTHER)
    cart_cookie.attach_cart_to_response(resp, cart, SECRET_KEY, secure=COOKIE_SECURE)
    return resp


@app.post("/cart/set-qty")
async def cart_set_quantity(
    request: Request,
    offer_id: int = Form(...),
    quantity: int = Form(...),
):
    cart = _cart_from_request(request)
    cart = cart_cookie.cart_set_qty(cart, offer_id, quantity)
    resp = RedirectResponse(url="/cart", status_code=status.HTTP_303_SEE_OTHER)
    cart_cookie.attach_cart_to_response(resp, cart, SECRET_KEY, secure=COOKIE_SECURE)
    return resp


@app.get("/cart", response_class=HTMLResponse)
async def cart_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    cart = _cart_from_request(request)
    cart_lines: List[dict] = []
    known_subtotal = 0.0
    missing_price = False
    missing_offers = 0

    if cart:
        ids = list(cart.keys())
        stmt = (
            select(ProductOffer)
            .options(
                selectinload(ProductOffer.company),
                selectinload(ProductOffer.category),
                selectinload(ProductOffer.properties).selectinload(
                    ProductProperty.property_name
                ),
            )
            .where(ProductOffer.offer_id.in_(ids))
        )
        offers = db.execute(stmt).scalars().all()
        by_id = {o.offer_id: o for o in offers}
        missing_offers = len(cart) - len(by_id)

        for oid in ids:
            offer = by_id.get(oid)
            if offer is None:
                continue
            qty = cart[oid]
            unit = shop_ui.get_offer_price_rub(offer)
            line_total: Optional[float] = None
            if unit is not None:
                line_total = unit * qty
                known_subtotal += line_total
            else:
                missing_price = True
            cart_lines.append(
                {
                    "offer": offer,
                    "qty": qty,
                    "unit_rub": unit,
                    "line_total_rub": line_total,
                    "price_label": shop_ui.get_offer_price_display(offer),
                }
            )

    priced_in_cart = sum(1 for row in cart_lines if row.get("unit_rub") is not None)
    cart_fully_priced = bool(cart_lines) and priced_in_cart == len(cart_lines) and not missing_price
    cart_subtotal_rub = known_subtotal if priced_in_cart else None

    return templates.TemplateResponse(
        request=request,
        name="cart.html",
        context={
            "user": current_user,
            "cart_lines": cart_lines,
            "cart_subtotal_rub": cart_subtotal_rub,
            "cart_missing_price": missing_price,
            "cart_fully_priced": cart_fully_priced,
            "cart_stale_count": missing_offers,
        },
    )


@app.get("/about", response_class=HTMLResponse)
async def about_page(
    request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={"user": current_user},
    )


# =====================
#   АДМИНКА / НОРМАЛИЗАЦИЯ (ТОЛЬКО ДЛЯ АВТОРИЗОВАННЫХ)
# =====================

def ensure_admin(user: Optional[User]) -> bool:
    return bool(user and user.is_active and user.is_superuser)


@app.get("/admin/analytics", response_class=HTMLResponse)
async def admin_analytics_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
    days: int = Query(30, ge=1, le=366),
    seeded: int = Query(0, ge=0, le=1),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    kpis = analytics.get_kpis(db, days=days)
    return templates.TemplateResponse(
        request=request,
        name="admin_analytics.html",
        context={
            "user": current_user,
            "kpis": kpis,
            "days": days,
            "show_seeded_notice": bool(seeded),
        },
    )


@app.post("/admin/analytics/seed-demo")
async def admin_analytics_seed_demo(
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    db = SessionLocal()
    try:
        analytics.seed_demo_metrics(db)
    finally:
        db.close()

    return RedirectResponse(
        url="/admin/analytics?days=30&seeded=1",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.get("/admin/normalization", response_class=HTMLResponse)
async def normalization_page(
    request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        request=request,
        name="normalization.html",
        context={
            "user": current_user,
        },
    )


@app.post("/admin/normalization/preview", response_class=HTMLResponse)
async def normalization_preview_start(
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    job = create_job("preview")
    threading.Thread(target=run_preview_job, args=(job.id,), daemon=True).start()

    return RedirectResponse(
        url=f"/admin/normalization/progress/{job.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/admin/normalization/apply")
async def normalization_apply_start(
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    job = create_job("apply")
    threading.Thread(target=run_apply_job, args=(job.id,), daemon=True).start()

    return RedirectResponse(
        url=f"/admin/normalization/progress/{job.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ======== старт задачи построения групп ========

@app.post("/admin/normalization/groups")
async def normalization_groups_start(
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    job = create_job("groups")
    threading.Thread(target=run_groups_job, args=(job.id,), daemon=True).start()

    return RedirectResponse(
        url=f"/admin/normalization/progress/{job.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/admin/search-index/build")
async def search_index_build_start(
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    job = create_job("search_index")
    threading.Thread(target=run_search_index_job, args=(job.id,), daemon=True).start()

    return RedirectResponse(
        url=f"/admin/normalization/progress/{job.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# =====================
#   SEARCH (PUBLIC)
# =====================


@app.get("/search/suggest")
async def search_suggest(
    q: str = Query("", max_length=220),
    db: Session = Depends(get_db),
):
    """JSON-подсказки для autocomplete (бренды, категории, названия товаров)."""
    choices = get_cached_suggest_strings(db)
    suggestions = suggest_completions(q, choices, limit=12)
    return JSONResponse({"suggestions": suggestions})


@app.get("/search", response_class=HTMLResponse)
async def search_page(
    request: Request,
    q: str = Query("", min_length=0),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
    page: int = Query(1, ge=1),
    company_id_q: Optional[str] = Query(None, alias="company_id"),
    category_id_q: Optional[str] = Query(None, alias="category_id"),
    price_min_q: Optional[str] = Query(None, alias="price_min"),
    price_max_q: Optional[str] = Query(None, alias="price_max"),
    sort: str = Query(""),
    ram_gb: Optional[int] = Query(None, ge=1, le=512, alias="ram_gb"),
    screen_q: Optional[str] = Query(None, alias="screen"),
    ssd_gb_q: Optional[str] = Query(None, alias="ssd_gb"),
    os_q: Optional[str] = Query(None, alias="os"),
    gpu_q: Optional[str] = Query(None, alias="gpu"),
    sl_keep: Optional[int] = Query(None, ge=1, alias="sl"),
):
    q = (q or "").strip()
    page_size = 50
    sort = _normalize_catalog_sort(sort)
    company_id = _parse_query_optional_int_ge1(company_id_q)
    category_id = _parse_query_optional_int_ge1(category_id_q)
    price_min, price_max = _normalize_price_bounds(
        _parse_query_optional_float_nonneg(price_min_q),
        _parse_query_optional_float_nonneg(price_max_q),
    )
    screen_bucket = _parse_screen_bucket_q(screen_q)
    ssd_facet_gb = _parse_ssd_facet_gb(ssd_gb_q)
    os_family = _parse_os_family_q(os_q)
    gpu_filter = _parse_gpu_q(gpu_q)

    if not q:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    companies, categories = _load_catalog_filter_lookups(db)
    company_names = [c.name for c in companies if c.name]
    category_names = [c.name for c in categories if c.name]

    (
        search_q,
        parsed_intent,
        query_refined,
        search_q_retrieval,
        fuzzy_changed,
        fuzzy_display_q,
    ) = prepare_search_query(q, company_names, category_names)
    if fuzzy_changed and fuzzy_display_q.strip():
        LOG.info(
            "Поиск: fuzzy-исправление (clean_q) %r -> %r",
            parsed_intent.clean_q,
            fuzzy_display_q,
        )
    if query_refined and search_q.casefold() != q.casefold():
        LOG.info("Поиск: уточнение запроса %r -> %r", q, search_q)

    if search_q_retrieval.casefold() != search_q.casefold():
        LOG.info("Поиск: расширение запроса (retrieval) %r -> %r", search_q, search_q_retrieval)

    if price_min is None and parsed_intent.price_min is not None:
        price_min = parsed_intent.price_min
    if price_max is None and parsed_intent.price_max is not None:
        price_max = parsed_intent.price_max
    price_min, price_max = _normalize_price_bounds(price_min, price_max)

    ram_filter_gb = ram_gb if ram_gb is not None else parsed_intent.ram_gb
    if screen_bucket is None:
        screen_bucket = _screen_bucket_from_diagonal(parsed_intent.diagonal_inch)
    if ssd_facet_gb is None and parsed_intent.ssd_gb is not None:
        ssd_facet_gb = parsed_intent.ssd_gb

    # Семантический поиск опционален: без модели/индекса работаем только на полнотексте.
    try:
        semantic_ranks = semantic_search_faiss(search_q_retrieval, top_k=200)
    except Exception as exc:
        LOG.warning("Семантический поиск недоступен (%s), используется только FTS", exc)
        semantic_ranks = []
    if not semantic_ranks:
        try:
            semantic_ranks = semantic_search_pgvector(db, search_q_retrieval, top_k=200)
        except Exception:
            semantic_ranks = []

    text_ranks = []
    try:
        text_ranks = text_search(db, search_q_retrieval, top_k=200)
    except Exception:
        text_ranks = []

    merged_ids = merge_search_rankings(
        "hybrid", semantic_ranks, text_ranks, k=60, limit=500
    )
    before_soft_filter = list(merged_ids)
    merged_ids = filter_offer_ids_by_query_tokens_soft(db, merged_ids, search_q)
    if not merged_ids and before_soft_filter:
        # Разговорный запрос может потерять все документы на soft-постфильтре.
        # В таком случае оставляем кандидатов retrieval и продолжаем стандартные фильтры.
        LOG.info(
            "Soft token filter returned empty; fallback to retrieval candidates (q=%r, candidates=%s)",
            q,
            len(before_soft_filter),
        )
        merged_ids = before_soft_filter
    ids_soft = list(merged_ids)
    merged_ids = _filter_search_offer_ids(
        db, merged_ids, company_id, category_id, price_min, price_max
    )
    if ram_filter_gb is not None:
        merged_ids = filter_offer_ids_by_ram_gb(db, merged_ids, ram_filter_gb)
    if screen_bucket:
        merged_ids = filter_offer_ids_by_screen_bucket(db, merged_ids, screen_bucket)
    if ssd_facet_gb is not None:
        merged_ids = filter_offer_ids_by_ssd_bucket(db, merged_ids, ssd_facet_gb)
    if os_family:
        merged_ids = filter_offer_ids_by_os_family(db, merged_ids, os_family)
    if gpu_filter:
        merged_ids = filter_offer_ids_by_gpu_value(db, merged_ids, gpu_filter)
    merged_ids = prioritize_vacuum_offers_by_suction(db, merged_ids, q)
    merged_ids = prioritize_by_battery(db, merged_ids, q)
    merged_ids = prioritize_by_display(db, merged_ids, q)
    if sort not in ("price_asc", "price_desc") and merged_ids:
        merged_ids = rerank_offer_ids_cross_encoder(db, search_q, merged_ids, head_size=50)
    merged_ids = _sort_offer_ids_by_price(db, merged_ids, sort)
    merged_ids = _materialize_search_offer_ids(db, merged_ids)

    total_offers = len(merged_ids)
    total_pages = max(1, (total_offers + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages

    offset = (page - 1) * page_size
    page_ids = merged_ids[offset: offset + page_size]
    offers = fetch_offers_in_order(db, page_ids)

    from_item = offset + 1 if offers else 0
    to_item = offset + len(offers)

    show_effective_search_hint = search_q.casefold().strip() != q.casefold().strip()
    filters_active = bool(
        (company_id and company_id > 0)
        or (category_id and category_id > 0)
        or price_min is not None
        or price_max is not None
        or sort
        or (ram_filter_gb is not None)
        or bool(screen_bucket)
        or (ssd_facet_gb is not None)
        or bool(os_family)
        or bool(gpu_filter)
    )

    facets = compute_search_facets(
        db,
        ids_soft,
        company_id=company_id,
        category_id=category_id,
        price_min=price_min,
        price_max=price_max,
        filter_offer_ids_fn=_filter_search_offer_ids,
        ram_gb_filter=ram_filter_gb,
        screen_bucket=screen_bucket,
        ssd_gb_filter=ssd_facet_gb,
        os_family=os_family,
        gpu_filter=gpu_filter,
    )
    _enrich_search_facet_hrefs(
        q=q,
        company_id=company_id,
        category_id=category_id,
        price_min=price_min,
        price_max=price_max,
        sort=sort,
        ram_gb=ram_filter_gb,
        screen_bucket=screen_bucket,
        ssd_gb=ssd_facet_gb,
        os_family=os_family,
        gpu_value=gpu_filter,
        facets=facets,
        search_sl=sl_keep,
    )
    if facets.get("ram_gb") or ram_filter_gb is not None:
        facets["ram_all_href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=company_id,
            category_id=category_id,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=None,
            screen_bucket=screen_bucket,
            ssd_gb=ssd_facet_gb,
            os_family=os_family,
            gpu=gpu_filter,
            search_sl=sl_keep,
        )
    if facets.get("screen_buckets") or screen_bucket:
        facets["screen_all_href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=company_id,
            category_id=category_id,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_filter_gb,
            screen_bucket=None,
            ssd_gb=ssd_facet_gb,
            os_family=os_family,
            gpu=gpu_filter,
            search_sl=sl_keep,
        )
    if facets.get("ssd_buckets") or ssd_facet_gb is not None:
        facets["ssd_all_href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=company_id,
            category_id=category_id,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_filter_gb,
            screen_bucket=screen_bucket,
            ssd_gb=None,
            os_family=os_family,
            gpu=gpu_filter,
            search_sl=sl_keep,
        )
    if facets.get("os_families") or os_family:
        facets["os_all_href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=company_id,
            category_id=category_id,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_filter_gb,
            screen_bucket=screen_bucket,
            ssd_gb=ssd_facet_gb,
            os_family=None,
            gpu=gpu_filter,
            search_sl=sl_keep,
        )
    if facets.get("gpu_values") or gpu_filter:
        facets["gpu_all_href"] = "/search?" + _listing_qs_no_page(
            is_search=True,
            q=q,
            company_id=company_id,
            category_id=category_id,
            price_min=price_min,
            price_max=price_max,
            sort=sort,
            ram_gb=ram_filter_gb,
            screen_bucket=screen_bucket,
            ssd_gb=ssd_facet_gb,
            os_family=os_family,
            gpu=None,
            search_sl=sl_keep,
        )

    search_log_id: Optional[int] = None
    if page == 1:
        search_log_id = log_search_query(
            db,
            q,
            int(total_offers),
            search_mode="hybrid",
        )
        if search_log_id is None:
            LOG.exception("Не удалось записать SearchLog")
    else:
        search_log_id = sl_keep

    listing_qs_no_page = _listing_qs_no_page(
        is_search=True,
        q=q,
        company_id=company_id,
        category_id=category_id,
        price_min=price_min,
        price_max=price_max,
        sort=sort,
        ram_gb=ram_filter_gb,
        screen_bucket=screen_bucket,
        ssd_gb=ssd_facet_gb,
        os_family=os_family,
        gpu=gpu_filter,
        search_sl=search_log_id,
    )

    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "offers": offers,
            "user": current_user,
            "page": page,
            "total_pages": total_pages,
            "total_offers": total_offers,
            "from_item": from_item,
            "to_item": to_item,
            "q": q,
            "effective_search_q": search_q,
            "query_refined": query_refined,
            "show_effective_search_hint": show_effective_search_hint,
            "fuzzy_corrected": fuzzy_changed,
            "fuzzy_corrected_query": (fuzzy_display_q.strip() if fuzzy_changed else None),
            "parsed_ram_gb": ram_filter_gb,
            "is_search": True,
            "companies": companies,
            "categories": categories,
            "filter_company_id": company_id or 0,
            "filter_category_id": category_id or 0,
            "filter_price_min": price_min,
            "filter_price_max": price_max,
            "filter_sort": sort,
            "listing_qs_no_page": listing_qs_no_page,
            "filters_active": filters_active,
            "search_facets": facets,
            "search_log_id": search_log_id,
            "search_ab_mode": "hybrid",
            "filter_ram_gb": ram_filter_gb,
            "filter_screen_bucket": screen_bucket,
            "filter_ssd_gb": ssd_facet_gb,
            "filter_os_family": os_family,
            "filter_gpu_value": gpu_filter,
        },
    )
    return response


@app.get("/admin/normalization/progress/{job_id}", response_class=HTMLResponse)
async def normalization_progress_page(
    job_id: str,
    request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    job = get_job(job_id)
    if not job:
        return HTMLResponse("Задача не найдена", status_code=404)

    return templates.TemplateResponse(
        request=request,
        name="normalization_progress.html",
        context={
            "user": current_user,
            "job_id": job.id,
            "job_kind": job.kind,
        },
    )


@app.get("/admin/normalization/job-status/{job_id}", response_class=JSONResponse)
async def normalization_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        return JSONResponse({"exists": False}, status_code=404)

    return JSONResponse(
        {
            "exists": True,
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "progress": job.progress,
            "stage": job.stage,
            "message": job.message,
            "has_result": job.result is not None,
            "error": job.error,
        }
    )


@app.get("/admin/normalization/preview-result/{job_id}", response_class=HTMLResponse)
async def normalization_preview_result(
    job_id: str,
    request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if not ensure_admin(current_user):
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    job = get_job(job_id)
    if not job or job.kind != "preview" or job.status != "done" or not job.result:
        return HTMLResponse("Результат предпросмотра недоступен", status_code=400)

    diffs = job.result.get("diffs", [])
    total_diffs = job.result.get("total_diffs", 0)

    return templates.TemplateResponse(
        request=request,
        name="normalization_preview.html",
        context={
            "user": current_user,
            "diffs": diffs,
            "total_diffs": total_diffs,
        },
    )


# =====================
#   АВТОРИЗАЦИЯ / РЕГИСТРАЦИЯ
# =====================

@app.get("/register", response_class=HTMLResponse)
async def register_page(
    request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={
            "user": current_user,
            "error": None,
        },
    )


@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    db: Session = Depends(get_db),
    username: str = Form(...),
    password: str = Form(...),
):
    if _is_rate_limited(_client_ip(request)):
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"user": None, "error": "Слишком много попыток. Подождите минуту."},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    username = username.strip()
    if not username or not password:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "user": None,
                "error": "Логин и пароль обязательны.",
            },
        )

    existing = get_user_by_username(db, username)
    if existing:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={
                "user": None,
                "error": "Пользователь с таким логином уже существует.",
            },
        )

    user = User(
        username=username,
        hashed_password=get_password_hash(password),
        is_active=True,
        is_superuser=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "user": current_user,
            "error": None,
        },
    )


@app.post("/login")
async def login(
    request: Request,
    db: Session = Depends(get_db),
    username: str = Form(...),
    password: str = Form(...),
):
    if _is_rate_limited(_client_ip(request)):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"user": None, "error": "Слишком много попыток. Подождите минуту."},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    username = username.strip()
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "user": None,
                "error": "Неверный логин или пароль.",
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    access_token = create_access_token({"sub": user.username})
    redirect = RedirectResponse(url="/admin/normalization", status_code=status.HTTP_303_SEE_OTHER)
    redirect.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=COOKIE_SECURE,
        samesite="lax",
    )
    return redirect


@app.post("/logout")
async def logout():
    redirect = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    redirect.delete_cookie("access_token")
    return redirect