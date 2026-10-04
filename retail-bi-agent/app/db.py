"""Database helpers: read-only connection, schema description, and SQL safety checks."""
import os
import re

import psycopg2
from dotenv import load_dotenv

load_dotenv()

MAX_ROWS = 100  # never pull back more than this many rows into the LLM prompt


def get_ro_connection():
    """Connect with the READ-ONLY account. The LLM's SQL never runs as bi_admin."""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "retail_bi"),
        user=os.getenv("POSTGRES_RO_USER", "bi_readonly"),
        password=os.getenv("POSTGRES_RO_PASSWORD", "bi_readonly_pass"),
    )
    conn.set_session(readonly=True, autocommit=True)  # second lock, on top of the role's own setting
    return conn


def get_schema_description() -> str:
    """Return the schema as compact text the LLM can read: tables, columns, types, foreign keys."""
    conn = get_ro_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
            tables = {}
            for table, column, dtype in cur.fetchall():
                tables.setdefault(table, []).append(f"{column} {dtype}")

            # pg_constraint is readable by any role. (information_schema hides constraints from
            # accounts that only have SELECT, so the read-only user would see no foreign keys there.)
            cur.execute("""
                SELECT c.conrelid::regclass::text, a.attname, c.confrelid::regclass::text, af.attname
                FROM pg_constraint c
                JOIN pg_attribute a  ON a.attrelid  = c.conrelid  AND a.attnum  = ANY(c.conkey)
                JOIN pg_attribute af ON af.attrelid = c.confrelid AND af.attnum = ANY(c.confkey)
                WHERE c.contype = 'f' AND c.connamespace = 'public'::regnamespace
                ORDER BY 1, 2
            """)
            fks = [f"{t}.{c} -> {rt}.{rc}" for t, c, rt, rc in cur.fetchall()]
    finally:
        conn.close()

    lines = [f"TABLE {name}({', '.join(cols)})" for name, cols in tables.items()]
    if fks:
        lines.append("FOREIGN KEYS: " + "; ".join(fks))
    return "\n".join(lines)


# ---------- SQL safety ----------
FORBIDDEN = [
    "insert", "update", "delete", "drop", "alter", "create", "truncate", "grant", "revoke",
    "copy", "call", "execute", "do", "vacuum", "pg_sleep", "pg_read_file", "lo_import", "lo_export",
]


class UnsafeSQLError(ValueError):
    pass


def validate_sql(sql: str) -> str:
    """Allow a single read-only SELECT (or WITH ... SELECT). Return cleaned SQL or raise UnsafeSQLError.

    This is the FIRST line of defence. The real guarantee is the read-only database role.
    """
    cleaned = sql.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        raise UnsafeSQLError("Empty query.")
    if "--" in cleaned or "/*" in cleaned:
        raise UnsafeSQLError("SQL comments are not allowed.")
    if ";" in cleaned:
        raise UnsafeSQLError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT queries are allowed.")
    for word in FORBIDDEN:
        if re.search(rf"\b{word}\b", cleaned, re.IGNORECASE):
            raise UnsafeSQLError(f"Forbidden keyword: {word}")
    return cleaned


def run_query(sql: str):
    """Validate and run a query. Returns (columns, rows, truncated). Raises on unsafe or failing SQL."""
    safe_sql = validate_sql(sql)
    conn = get_ro_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(safe_sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(MAX_ROWS + 1)
    finally:
        conn.close()
    truncated = len(rows) > MAX_ROWS
    return columns, rows[:MAX_ROWS], truncated


def get_data_range():
    """Earliest and latest order dates, so the agent knows the limits of its data."""
    conn = get_ro_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MIN(order_date), MAX(order_date) FROM orders")
            return cur.fetchone()
    finally:
        conn.close()
