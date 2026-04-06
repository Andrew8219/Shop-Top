В качестве фреймворка используется FastAPI

## Интеллектуальный поиск 

Проект поддерживает гибридный поиск по товарам:

- **Семантика**: sentence-transformers (E5) → эмбеддинги → поиск кандидатов через **FAISS** и/или **pgvector**.
- **Полнотекст**: PostgreSQL `tsvector` + `websearch_to_tsquery`.
- **Гибрид**: объединение рангов через **RRF** (Reciprocal Rank Fusion).

### Подготовка PostgreSQL (опционально для pgvector)

Если хочешь хранить эмбеддинги в Postgres, установи расширение:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Если расширение недоступно — поиск всё равно работает через FAISS + tsvector.

### Сборка индекса

Залогинься в админку и запусти сборку индекса:

- `POST /admin/search-index/build`

После сборки можно искать:

- `GET /search?q=...`

### Оценка эффективности (задача 7 ВКР)

Скрипт `evaluate_vkr.py` считает Recall@K, MRR@K, nDCG@K на наборе `evaluation/task5_queries.json`, сравнивает режимы **semantic_only**, **text_only**, **hybrid_rrf_nofilter** (RRF без постфильтра) и **hybrid_rrf** (как на витрине). Методология: `evaluation/task7_methodology.md`. Отчёты: `evaluation/reports/vkr_metrics.md` и `.json`.

```bash
python evaluate_vkr.py
python evaluate_vkr.py --dry-run
python evaluate_vkr.py --task7-only
```
