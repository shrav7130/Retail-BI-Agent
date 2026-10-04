import pytest

from app.tools import calculate_change, calculator, get_database_schema, run_sql_query


def test_schema_tool_lists_tables():
    out = get_database_schema.invoke({})
    assert "TABLE orders" in out
    for fk in ("orders.customer_id -> customers.customer_id", "orders.store_id -> stores.store_id",
               "order_items.order_id -> orders.order_id", "order_items.product_id -> products.product_id"):
        assert fk in out


def test_sql_tool_returns_table_text():
    out = run_sql_query.invoke({"sql": "SELECT store_name FROM stores ORDER BY store_id LIMIT 2"})
    assert out.splitlines()[0] == "store_name" and "Mumbai Central" in out


def test_sql_tool_returns_error_string_not_exception():
    out = run_sql_query.invoke({"sql": "SELECT nope FROM stores"})
    assert out.startswith("ERROR:")


def test_sql_tool_blocks_writes_with_error_string():
    out = run_sql_query.invoke({"sql": "DELETE FROM orders"})
    assert out.startswith("ERROR: query blocked")


def test_sql_tool_empty_result():
    out = run_sql_query.invoke({"sql": "SELECT * FROM stores WHERE store_id = -1"})
    assert "0 rows" in out


def test_sql_tool_caps_large_results():
    out = run_sql_query.invoke({"sql": "SELECT order_id FROM orders"})
    assert "capped" in out and len(out.splitlines()) < 60


def test_calculate_change():
    out = calculate_change.invoke({"old_value": 7056798, "new_value": 5892903})
    assert "-16.49%" in out and "decrease" in out
    assert "undefined" in calculate_change.invoke({"old_value": 0, "new_value": 5})


@pytest.mark.parametrize("expr, expected", [
    ("2 + 3 * 4", "14"), ("(5892903 - 7056798) / 7056798 * 100", "-16.4932"), ("-5 + 2", "-3"), ("10 / 4", "2.5"),
])
def test_calculator_ok(expr, expected):
    assert calculator.invoke({"expression": expr}) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')", "open('x')", "2 ** 9999999", "a + 1", "[1,2]", "'a' * 5", "True + 1",
])
def test_calculator_rejects_anything_but_arithmetic(expr):
    assert calculator.invoke({"expression": expr}).startswith("ERROR")


def test_calculator_division_by_zero():
    assert calculator.invoke({"expression": "1/0"}) == "ERROR: division by zero"
