from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import threading
import uuid
import logging
from typing import List, Optional, Dict

from fastapi import (
    FastAPI,
    Depends,
    Request,
    Form,
    status,
    Query,
)
    # noqa
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func
from sqlalchemy.orm import Session, selectinload

from jose import JWTError, jwt
from passlib.context import CryptContext

from db import SessionLocal, Base, engine
from models import (
    ProductOffer,
    ProductProperty,
    User,
)
from search_index import (
    build_search_index,
    fetch_offers_in_order,
    rrf_merge,
    semantic_search_faiss,
    semantic_search_pgvector,
    text_search,
)
from normalization import (
    build_mapping,
    apply_mapping,
    merge_duplicate_property_names,
    build_property_name_groups,
)

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
#   JWT / ПАРОЛИ
# =====================

SECRET_KEY = "CHANGE_ME_TO_LONG_RANDOM_SECRET"  # замените на длинную случайную строку
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 день

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

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MediaLux Shop")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
        # пока прогресс простой (в перспективе можно детализировать батчи)
        update_job(job_id, progress=5, stage="collect", message="Читаем товары из БД")
        result = build_search_index(db, rebuild=True)
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
#   МАГАЗИН (публичная часть)
# =====================

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
    page: int = Query(1, ge=1),
):
    page_size = 100

    total_offers = db.execute(select(func.count()).select_from(ProductOffer)).scalar() or 0

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
            )
            .order_by(ProductOffer.offer_id)
            .offset(offset)
            .limit(page_size)
        )
        offers = db.execute(stmt).scalars().all()

        from_item = offset + 1
        to_item = offset + len(offers)

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
        },
    )


@app.get("/product/{offer_id}", response_class=HTMLResponse)
async def product_detail(
    offer_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
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
        return HTMLResponse(
            content=f"<h1>Товар с offer_id={offer_id} не найден</h1>",
            status_code=404,
        )

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

    return templates.TemplateResponse(
        request=request,
        name="product_detail.html",
        context={
            "offer": offer,
            "props": props,
            "grouped_props": grouped_props,
            "user": current_user,
        },
    )


# =====================
#   АДМИНКА / НОРМАЛИЗАЦИЯ (ТОЛЬКО ДЛЯ АВТОРИЗОВАННЫХ)
# =====================

def ensure_admin(user: Optional[User]) -> bool:
    return bool(user and user.is_active and user.is_superuser)


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


@app.get("/search", response_class=HTMLResponse)
async def search_page(
    request: Request,
    q: str = Query("", min_length=0),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
    page: int = Query(1, ge=1),
):
    q = (q or "").strip()
    page_size = 50

    if not q:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    # кандидаты: semantic + text
    semantic_ranks = semantic_search_faiss(q, top_k=200)
    if not semantic_ranks:
        # fallback на pgvector (если есть)
        try:
            semantic_ranks = semantic_search_pgvector(db, q, top_k=200)
        except Exception:
            semantic_ranks = []

    text_ranks = []
    try:
        text_ranks = text_search(db, q, top_k=200)
    except Exception:
        text_ranks = []

    merged_ids = rrf_merge(semantic_ranks, text_ranks, k=60, limit=500)

    total_offers = len(merged_ids)
    total_pages = max(1, (total_offers + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages

    offset = (page - 1) * page_size
    page_ids = merged_ids[offset: offset + page_size]
    offers = fetch_offers_in_order(db, page_ids)

    from_item = offset + 1 if offers else 0
    to_item = offset + len(offers)

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
            "q": q,
            "is_search": True,
        },
    )


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
        is_superuser=True,
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
        secure=False,
        samesite="lax",
    )
    return redirect


@app.post("/logout")
async def logout():
    redirect = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    redirect.delete_cookie("access_token")
    return redirect