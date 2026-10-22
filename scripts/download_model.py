#!/usr/bin/env python3
"""
Скачивает модель эмбеддингов для семантического поиска.

По умолчанию — Qwen3-Embedding через Ollama (без PyTorch).
Для legacy HuggingFace/e5: python -m scripts.download_model --provider local

Использование:
    python3 download_model.py
    python3 download_model.py --provider ollama
"""
from __future__ import annotations

import argparse
import sys
import time


def pull_ollama_model(model: str, host: str) -> None:
    import subprocess

    print(f"Скачиваю Ollama-модель {model}...")
    print("Размер ~640 МБ — займёт несколько минут при первом запуске.")
    t0 = time.time()
    subprocess.run(["ollama", "pull", model], check=True)
    print(f"\nГотово за {time.time() - t0:.0f}с!")

    import requests

    resp = requests.post(
        f"{host.rstrip('/')}/api/embed",
        json={"model": model, "input": ["ноутбук для игр", "видеокарта RTX 4060"]},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    embs = data.get("embeddings") or []
    print(f"  Размерность эмбеддингов: {len(embs[0]) if embs else '?'}")
    print(f"  Тестовых векторов: {len(embs)}")
    print("\nМодель готова к использованию!")


def pull_hf_model(model_name: str) -> None:
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        missing.append("sentence-transformers")
    if missing:
        print(f"Установите зависимости: pip install {' '.join(missing)}")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer

    print(f"Скачиваю HuggingFace модель {model_name}...")
    print("Размер ~1.1 ГБ — займёт несколько минут при первом запуске.")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    elapsed = time.time() - t0
    dim = model.get_sentence_embedding_dimension()
    print(f"\nГотово за {elapsed:.0f}с!")
    print(f"  Размерность эмбеддингов: {dim}")
    print(f"  Устройство: {model.device}")

    test_queries = [
        "query: видеокарта для игр",
        "query: ноутбук до 60000 рублей",
        "passage: NVIDIA GeForce RTX 4060 8GB GDDR6",
    ]
    print("\nТест эмбеддингов...")
    embeddings = model.encode(test_queries)
    print(f"  Входов: {len(test_queries)}, форма тензора: {embeddings.shape}")
    print("\nМодель готова к использованию!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download embedding model for search.")
    parser.add_argument(
        "--provider",
        choices=("ollama", "local"),
        default="ollama",
        help="ollama (Qwen3-Embedding, default) or local (sentence-transformers)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: qwen3-embedding:0.6b or intfloat/multilingual-e5-large)",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama host (only for --provider ollama)",
    )
    args = parser.parse_args()

    if args.provider == "ollama":
        model = args.model or "qwen3-embedding:0.6b"
        pull_ollama_model(model, args.host)
    else:
        model = args.model or "intfloat/multilingual-e5-large"
        pull_hf_model(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
