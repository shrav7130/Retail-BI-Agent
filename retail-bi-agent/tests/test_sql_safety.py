import pytest

from app.db import UnsafeSQLError, run_query, validate_sql


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders",
    "select count(*) from orders;",
    "WITH m AS (SELECT store_id, SUM(total_amount) t FROM orders GROUP BY 1) SELECT * FROM m",
])
def test_allows_safe_selects(sql):
    assert validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "DROP TABLE stores",
    "INSERT INTO stores VALUES (9,'x','x','x','2020-01-01')",
    "UPDATE orders SET total_amount = 0",
    "SELECT 1; DROP TABLE orders",
    "SELECT * FROM orders -- sneaky",
    "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
    "TRUNCATE orders",
    "SELECT pg_sleep(100)",
    "",
])
def test_blocks_unsafe_sql(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_database_itself_refuses_writes():
    """Even if validation were bypassed, the read-only role must refuse. Needs the DB running."""
    from app.db import get_ro_connection
    conn = get_ro_connection()
    try:
        with conn.cursor() as cur:
            with pytest.raises(Exception, match="read-only"):
                cur.execute("DELETE FROM orders")
    finally:
        conn.close()


def test_run_query_returns_rows():
    cols, rows, truncated = run_query("SELECT store_name FROM stores ORDER BY store_id")
    assert cols == ["store_name"] and len(rows) == 5 and not truncated
