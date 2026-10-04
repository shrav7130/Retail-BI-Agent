"""The agent's tools. Each tool = a plain Python function + a docstring the LLM reads to decide when to use it.

Rule for tools that touch the outside world: never let an exception escape. Return an "ERROR: ..." string
instead, so the agent can read it and correct itself.
"""
import ast
import operator

from langchain_core.tools import tool

from app.db import UnsafeSQLError, get_schema_description, run_query
from app.rag import search_company_docs  # noqa: F401  (re-exported so agent.py imports all tools from here)

MAX_ROWS_SHOWN = 25   # every row the agent reads is re-sent on each later model call, so keep it small


@tool
def get_database_schema() -> str:
    """Show the database tables, their columns with types, and the foreign keys.
    Call this BEFORE writing SQL if you are unsure about table or column names."""
    return get_schema_description()


@tool
def run_sql_query(sql: str) -> str:
    """Run ONE read-only PostgreSQL SELECT query against the retail database and return the rows.
    Use this for any question about sales, revenue, products, stores, customers, orders or discounts.
    Only SELECT (or WITH ... SELECT) is allowed. If the result starts with ERROR, read the message,
    fix the query and try again.
    Input: a single SQL query string."""
    try:
        columns, rows, truncated = run_query(sql)
    except UnsafeSQLError as e:
        return f"ERROR: query blocked ({e}). Only a single read-only SELECT is allowed."
    except Exception as e:  # Postgres errors: wrong column, bad syntax, timeout...
        return f"ERROR: {str(e).strip()}"

    if not rows:
        return "Query ran successfully but returned 0 rows."
    shown = rows[:MAX_ROWS_SHOWN]
    lines = [" | ".join(columns)] + [" | ".join(str(v) for v in row) for row in shown]
    if truncated:
        lines.append(f"(result capped: more than {len(rows)} rows exist; add LIMIT/aggregation)")
    elif len(rows) > MAX_ROWS_SHOWN:
        lines.append(f"(showing first {MAX_ROWS_SHOWN} of {len(rows)} rows)")
    return "\n".join(lines)


@tool
def calculate_change(old_value: float, new_value: float) -> str:
    """Compare two numbers (for example last month's and this month's sales).
    Returns the absolute change and the percentage change relative to old_value.
    ALWAYS use this for percentage changes instead of calculating in your head."""
    diff = new_value - old_value
    if old_value == 0:
        return f"Absolute change: {diff:,.2f}. Percentage change is undefined because the old value is 0."
    pct = diff / abs(old_value) * 100
    direction = "increase" if diff > 0 else "decrease" if diff < 0 else "no change"
    return f"Absolute change: {diff:,.2f}. Percentage change: {pct:+.2f}% ({direction})."


# ---------- safe calculator ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _eval(node.operand) if isinstance(node.op, ast.UAdd) else -_eval(node.operand)
    raise ValueError("only numbers and + - * / // % ( ) are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression such as '(5892903 - 7056798) / 7056798 * 100'.
    Supports numbers and + - * / // % and parentheses. Use it for totals, shares and ratios
    instead of doing arithmetic yourself."""
    try:
        result = _eval(ast.parse(expression.strip(), mode="eval"))
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except (ValueError, SyntaxError) as e:
        return f"ERROR: could not evaluate ({e})"
    return f"{result:,.4f}".rstrip("0").rstrip(".")


ALL_TOOLS = [get_database_schema, run_sql_query, search_company_docs, calculate_change, calculator]
