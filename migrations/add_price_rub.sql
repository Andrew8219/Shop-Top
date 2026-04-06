-- Добавить колонку цены для существующей БД (create_all новые колонки не создаёт).
ALTER TABLE product_offers
    ADD COLUMN IF NOT EXISTS price_rub NUMERIC(12, 2);
