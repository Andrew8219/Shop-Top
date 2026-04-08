"""Корзина в подписанной cookie (offer_id → количество)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
from typing import Dict, List, Optional

LOG = logging.getLogger(__name__)

COOKIE_NAME = "shop_cart_v1"
MAX_DISTINCT_ITEMS = 60
MAX_QTY_PER_ITEM = 99
COOKIE_MAX_AGE = 60 * 60 * 24 * 90  # 90 дней


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def parse_cart_cookie(raw: Optional[str], secret: str) -> Dict[int, int]:
    if not raw or not secret:
        return {}
    try:
        payload_b64, sig_b64 = raw.split(".", 1)
        body = _b64d(payload_b64)
        sig = _b64d(sig_b64)
        expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return {}
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, list):
            return {}
        out: Dict[int, int] = {}
        for pair in data:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            oid, qty = int(pair[0]), int(pair[1])
            if oid <= 0 or qty <= 0 or qty > MAX_QTY_PER_ITEM:
                continue
            out[oid] = qty
            if len(out) >= MAX_DISTINCT_ITEMS:
                break
        return out
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error, OSError) as e:
        LOG.debug("cart cookie parse failed: %s", e)
        return {}


def serialize_cart_cookie(cart: Dict[int, int], secret: str) -> str:
    items = sorted((int(oid), min(MAX_QTY_PER_ITEM, max(1, int(qty)))) for oid, qty in cart.items())
    items = items[:MAX_DISTINCT_ITEMS]
    body = json.dumps(items, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return f"{_b64e(body)}.{_b64e(sig)}"


def cart_add(cart: Dict[int, int], offer_id: int, quantity: int = 1) -> Dict[int, int]:
    q = int(quantity)
    if q < 1:
        q = 1
    if q > MAX_QTY_PER_ITEM:
        q = MAX_QTY_PER_ITEM
    new_cart = dict(cart)
    prev = new_cart.get(int(offer_id), 0)
    total = min(MAX_QTY_PER_ITEM, prev + q)
    new_cart[int(offer_id)] = total
    if len(new_cart) > MAX_DISTINCT_ITEMS:
        # сохраняем порядок вставки: удаляем самые «старые» ключи (минимальные id)
        for oid in sorted(new_cart.keys())[: len(new_cart) - MAX_DISTINCT_ITEMS]:
            del new_cart[oid]
    return new_cart


def cart_remove(cart: Dict[int, int], offer_id: int) -> Dict[int, int]:
    new_cart = dict(cart)
    new_cart.pop(int(offer_id), None)
    return new_cart


def cart_set_qty(cart: Dict[int, int], offer_id: int, quantity: int) -> Dict[int, int]:
    new_cart = dict(cart)
    q = int(quantity)
    if q <= 0:
        new_cart.pop(int(offer_id), None)
    else:
        new_cart[int(offer_id)] = min(MAX_QTY_PER_ITEM, q)
    return new_cart


def attach_cart_to_response(
    response,
    cart: Dict[int, int],
    secret: str,
    *,
    secure: bool = False,
    truncated_flag: Optional[bool] = None,
) -> bool:
    """
    Записывает корзину в cookie. Возвращает True если корзина была усечена.
    Вызывающий код может проверить возвращаемое значение и показать предупреждение.
    """
    if not cart:
        response.delete_cookie(COOKIE_NAME, path="/")
        return False

    token = serialize_cart_cookie(cart, secret)
    was_truncated = False
    if len(token) > 3800:
        # Оставляем последние MAX_DISTINCT_ITEMS // 2 позиций по id
        keep = MAX_DISTINCT_ITEMS // 2
        trimmed = dict(sorted(cart.items(), key=lambda x: x[0])[-keep:])
        token = serialize_cart_cookie(trimmed, secret)
        was_truncated = True
        LOG.warning(
            "cart cookie слишком большой (%d байт), усечён до %d позиций",
            len(serialize_cart_cookie(cart, secret)),
            keep,
        )

    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=COOKIE_MAX_AGE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=secure,
    )
    if was_truncated:
        # флаг для шаблона — живёт 30 секунд
        response.set_cookie(
            "cart_truncated",
            "1",
            max_age=30,
            path="/",
            httponly=False,
            samesite="lax",
            secure=secure,
        )
    return was_truncated
