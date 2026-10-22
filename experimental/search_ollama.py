"""
search_ollama.py — Ollama-интеграция для поиска (Qwen 3.5 + Qwen3-Embedding).

Эмбеддинги: qwen3-embedding (без PyTorch, без MPS segfault на macOS).
LLM-парсинг запросов: qwen3.5 через /api/chat (think=false обязателен).

Конфигурация (.env):
  SEARCH_EMBED_PROVIDER=ollama
  SEARCH_OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
  SEARCH_LLM_PROVIDER=ollama
  SEARCH_LLM_PARSE_ENABLED=1
  SEARCH_OLLAMA_LLM_MODEL=qwen3.5:4b
  SEARCH_OLLAMA_BASE_URL=http://127.0.0.1:11434

Перед первым запуском:
  ollama pull qwen3-embedding:0.6b
  ollama pull qwen3.5:4b
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

LOG = logging.getLogger(__name__)

OLLAMA_HOST: str = (
    os.getenv("SEARCH_OLLAMA_HOST", "").strip()
    or os.getenv("SEARCH_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
).rstrip("/")
EMBED_MODEL: str = os.getenv("SEARCH_OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
LLM_MODEL: str = os.getenv("SEARCH_OLLAMA_LLM_MODEL", "qwen3.5:4b")
LLM_MODEL_FAST: str = os.getenv("SEARCH_OLLAMA_LLM_MODEL_FAST", "").strip()
EMBED_DIM: int = int(os.getenv("SEARCH_OLLAMA_EMBED_DIM", "1024"))
EMBED_BATCH_SIZE: int = int(os.getenv("SEARCH_OLLAMA_EMBED_BATCH", "32"))
EMBED_TIMEOUT_SEC: float = float(os.getenv("SEARCH_OLLAMA_TIMEOUT_SEC", "120"))
LLM_TIMEOUT_SEC: float = float(os.getenv("SEARCH_OLLAMA_LLM_TIMEOUT_SEC", "180"))
LLM_ENABLED: bool = os.getenv("SEARCH_LLM_PARSE_ENABLED", "0").strip().lower() in ("1", "true", "yes")
LLM_MIN_TOKENS: int = int(os.getenv("SEARCH_LLM_PARSE_MIN_TOKENS", "4") or "4")
LLM_CACHE_TTL_SEC: int = int(os.getenv("SEARCH_LLM_CACHE_TTL_SEC", "86400") or "86400")

# Кэш в памяти: (model, text_hash) → вектор
_EMBED_CACHE: Dict[Tuple[str, str], np.ndarray] = {}
_EMBED_CACHE_MAX = 2000


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32, copy=False)


def _ollama_embed(texts: Sequence[str]) -> List[np.ndarray]:
    if not texts:
        return []

    url = f"{OLLAMA_HOST}/api/embed"
    payload = {"model": EMBED_MODEL, "input": list(texts)}

    try:
        resp = requests.post(url, json=payload, timeout=EMBED_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Ollama недоступен по адресу {OLLAMA_HOST}. "
            f"Запустите Ollama и выполните: ollama pull {EMBED_MODEL}"
        ) from e
    except requests.HTTPError as e:
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        raise RuntimeError(
            f"Ollama /api/embed вернул ошибку для модели '{EMBED_MODEL}': {e}. {body}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ollama embed failed: {e}") from e

    raw = data.get("embeddings")
    if not raw:
        raise RuntimeError(f"Ollama embed: пустой ответ (keys={list(data.keys())})")

    out: List[np.ndarray] = []
    for vec in raw:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape[0] != EMBED_DIM:
            raise RuntimeError(
                f"Неожиданная размерность эмбеддинга: {arr.shape[0]} (ожидалось {EMBED_DIM}). "
                f"Обновите SEARCH_OLLAMA_EMBED_DIM или пересоберите индекс."
            )
        out.append(_normalize(arr))
    return out


def _cache_put(key: Tuple[str, str], vec: np.ndarray) -> None:
    if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
        _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
    _EMBED_CACHE[key] = vec


def embed_query_ollama(q: str) -> np.ndarray:
    """Эмбеддинг одного поискового запроса через Ollama."""
    key = (EMBED_MODEL, _cache_key(q))
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]

    vec = _ollama_embed([q])[0]
    _cache_put(key, vec)
    return vec


def embed_passages_ollama(texts: Sequence[str]) -> np.ndarray:
    """
    Эмбеддинги списка текстов (карточки товаров).
    Возвращает массив shape (N, EMBED_DIM).
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    all_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[i : i + EMBED_BATCH_SIZE])
        all_vecs.extend(_ollama_embed(batch))
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(0.05)

    return np.vstack(all_vecs)


def is_ollama_configured() -> bool:
    """Проверяет доступность Ollama и наличие embedding-модели."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        models = {m.get("name", "") for m in resp.json().get("models", [])}
        # ollama list может вернуть "qwen3-embedding:0.6b" или с digest
        return any(EMBED_MODEL.split(":")[0] in name for name in models)
    except Exception:
        return False


def is_ollama_llm_configured() -> bool:
    """Проверяет доступность Ollama и наличие LLM-модели (qwen3.5 и т.д.)."""
    if not LLM_ENABLED:
        return False
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        models = {m.get("name", "") for m in resp.json().get("models", [])}
        prefix = LLM_MODEL.split(":")[0]
        return any(name == LLM_MODEL or name.startswith(f"{prefix}:") for name in models)
    except Exception:
        return False


def _ollama_chat_json(system: str, user: str, *, model: str) -> dict:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0, "num_predict": 400},
    }
    try:
        resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Ollama недоступен по адресу {OLLAMA_HOST}. "
            f"Запустите Ollama и выполните: ollama pull {LLM_MODEL}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}") from e

    content = (data.get("message") or {}).get("content") or ""
    if not content.strip():
        thinking = (data.get("message") or {}).get("thinking") or ""
        if thinking.strip():
            raise RuntimeError(
                "Ollama chat: ответ ушёл в thinking (включите think:false или увеличьте num_predict)"
            )
        raise RuntimeError("Ollama chat: пустой ответ")
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        text = text.rstrip("`").strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama chat: ожидался JSON-объект")
    return parsed


def parse_query_llm_ollama(q: str):
    """
    Разбор разговорного запроса через Qwen 3.5 (Ollama /api/chat).
    Возвращает LLMQueryIntent или None.
    """
    if not q or len(q.strip()) < 3:
        return None
    qn = q.strip()
    # LLM не должен тормозить короткие/простые запросы (качество обеспечиваем rule-based пайплайном).
    if len(re.findall(r"\w+", qn, flags=re.UNICODE)) < max(1, LLM_MIN_TOKENS):
        return None
    # TTL для LRU: добавляем «эпоху» в ключ (чтобы кэш сам протухал).
    epoch = int(time.time() // max(60, LLM_CACHE_TTL_SEC))
    return _parse_query_llm_ollama_cached(f"{epoch}::{qn}")


@lru_cache(maxsize=512)
def _parse_query_llm_ollama_cached(q_norm: str):
    from search_openai import _SYSTEM_PROMPT, build_llm_intent_from_json

    if not LLM_ENABLED:
        return None
    try:
        # ключ кэша может содержать префикс epoch::
        q_raw = q_norm.split("::", 1)[1] if "::" in q_norm else q_norm
        use_fast = bool(LLM_MODEL_FAST) and len(re.findall(r"\w+", q_raw, flags=re.UNICODE)) <= 8
        model = LLM_MODEL_FAST if use_fast else LLM_MODEL
        data = _ollama_chat_json(_SYSTEM_PROMPT, q_raw, model=model)
    except Exception as e:
        LOG.warning("Ollama LLM parse failed for %r: %s", q_norm, e)
        return None
    return build_llm_intent_from_json(data, q_raw)
