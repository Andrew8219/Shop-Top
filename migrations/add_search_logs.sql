-- Журнал поисковых запросов, кликов и режима A/B (create_all создаёт таблицу на новых БД;
-- для существующей PostgreSQL выполните этот скрипт вручную при необходимости).

CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    clicked_offer_id BIGINT NULL,
    results_count INTEGER NOT NULL,
    search_mode VARCHAR(16) NOT NULL DEFAULT 'hybrid',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_search_logs_clicked_offer_id ON search_logs (clicked_offer_id);
CREATE INDEX IF NOT EXISTS ix_search_logs_created_at ON search_logs (created_at);
CREATE INDEX IF NOT EXISTS ix_search_logs_search_mode ON search_logs (search_mode);

-- Примеры метрик:
-- CTR по режиму: SELECT search_mode, COUNT(*) AS n,
--   SUM(CASE WHEN clicked_offer_id IS NOT NULL THEN 1 ELSE 0 END) AS clicks
--   FROM search_logs GROUP BY search_mode;
