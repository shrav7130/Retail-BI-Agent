-- Extension for Day 2 (RAG embeddings)
--CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE stores (
    store_id     INT PRIMARY KEY,
    store_name   TEXT NOT NULL,
    city         TEXT NOT NULL,
    region       TEXT NOT NULL,
    opened_date  DATE NOT NULL
);

CREATE TABLE products (
    product_id    INT PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL,
    cost_price    NUMERIC(10,2) NOT NULL
);

CREATE TABLE customers (
    customer_id    INT PRIMARY KEY,
    customer_name  TEXT NOT NULL,
    city           TEXT NOT NULL,
    signup_date    DATE NOT NULL
);

CREATE TABLE orders (
    order_id      INT PRIMARY KEY,
    customer_id   INT NOT NULL REFERENCES customers(customer_id),
    store_id      INT NOT NULL REFERENCES stores(store_id),
    order_date    DATE NOT NULL,
    total_amount  NUMERIC(12,2) NOT NULL      -- after discounts
);

CREATE TABLE order_items (
    order_item_id  INT PRIMARY KEY,
    order_id       INT NOT NULL REFERENCES orders(order_id),
    product_id     INT NOT NULL REFERENCES products(product_id),
    quantity       INT NOT NULL,
    unit_price     NUMERIC(10,2) NOT NULL,
    discount_pct   NUMERIC(5,2) NOT NULL DEFAULT 0,   -- 15.00 means 15% off
    line_total     NUMERIC(12,2) NOT NULL             -- quantity * unit_price * (1 - discount)
);

CREATE INDEX idx_orders_date  ON orders(order_date);
CREATE INDEX idx_orders_store ON orders(store_id);
CREATE INDEX idx_items_order  ON order_items(order_id);
CREATE INDEX idx_items_prod   ON order_items(product_id);

-- Read-only role: this is the account the LLM's SQL tool will use.
-- (dev-only password; use secrets in real deployments)
CREATE ROLE bi_readonly LOGIN PASSWORD 'bi_readonly_pass';
GRANT CONNECT ON DATABASE retail_bi TO bi_readonly;
GRANT USAGE ON SCHEMA public TO bi_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO bi_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO bi_readonly;
-- Defence in depth: even if a write slips through, the session refuses it,
-- and runaway queries are killed after 10 seconds.
ALTER ROLE bi_readonly SET default_transaction_read_only = on;
ALTER ROLE bi_readonly SET statement_timeout = '10s';
