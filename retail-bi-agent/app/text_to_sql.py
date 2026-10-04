"""Day 1 baseline:  Question -> LLM -> SQL -> PostgreSQL -> LLM -> Answer."""
import os
import re
from datetime import date

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.db import UnsafeSQLError, get_schema_description, run_query

load_dotenv()

MAX_ATTEMPTS = 2  # first try + one retry where the LLM sees the error and fixes its SQL


def get_llm() -> ChatOpenAI:
    # temperature=0 -> as deterministic as possible, which is what you want for SQL
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def build_sql_system_prompt(schema: str) -> str:
    return f"""You are a PostgreSQL expert for a retail company called UrbanCart.
Convert the user's business question into ONE PostgreSQL SELECT query.

DATABASE SCHEMA:
{schema}

BUSINESS RULES:
- Today's date is {date.today().isoformat()}. Interpret "last month", "this month", etc. relative to today.
  Example: last month = order_date >= date_trunc('month', CURRENT_DATE) - interval '1 month'
                    AND order_date <  date_trunc('month', CURRENT_DATE)
- "Sales" or "revenue" = SUM(order_items.line_total) (already after discounts). Join orders for dates and stores.
- discount_pct is a percentage on order_items (15.00 means 15% off).
- Amounts are in Indian rupees.
- Product categories: Electronics, Grocery, Apparel, Home, Beauty.

RULES:
- Output ONLY the SQL query. No explanation, no markdown fences.
- Only SELECT (or WITH ... SELECT). Never modify data.
- Use only tables and columns from the schema above.
- For "top N" or "lowest" questions, use ORDER BY and LIMIT.
- Round money to 2 decimals and give columns readable aliases."""


def extract_sql(text: str) -> str:
    """LLMs sometimes wrap SQL in ```sql fences despite instructions. Strip them."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def generate_sql(question: str, llm, schema: str, previous_error: str | None = None, previous_sql: str | None = None) -> str:
    messages = [("system", build_sql_system_prompt(schema)), ("human", question)]
    if previous_error:
        messages.append(("ai", previous_sql or ""))
        messages.append(("human", f"That query failed with this error:\n{previous_error}\nReturn a corrected SQL query only."))
    return extract_sql(llm.invoke(messages).content)


def summarise(question: str, sql: str, columns, rows, truncated: bool, llm) -> str:
    table = "\n".join([" | ".join(columns)] + [" | ".join(str(v) for v in row) for row in rows[:50]])
    note = "\n(Result truncated: more rows exist.)" if truncated else ""
    prompt = f"""Question: {question}

SQL that was run:
{sql}

Result:
{table if rows else '(no rows returned)'}{note}

Answer the question in 1-3 clear sentences using ONLY the result above. Use rupee amounts where relevant.
If the result is empty, say no matching data was found. Do not invent numbers."""
    return llm.invoke([("human", prompt)]).content.strip()


def answer_question(question: str, llm=None) -> dict:
    """Run the full pipeline. Always returns a dict; never raises for expected failures."""
    llm = llm or get_llm()
    schema = get_schema_description()

    sql, error = None, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        sql = generate_sql(question, llm, schema, previous_error=error, previous_sql=sql if error else None)
        try:
            columns, rows, truncated = run_query(sql)
            answer = summarise(question, sql, columns, rows, truncated, llm)
            return {"question": question, "sql": sql, "columns": columns, "rows": rows,
                    "answer": answer, "attempts": attempt, "error": None, "truncated":truncated}
        except UnsafeSQLError as e:   # blocked by our validator: don't retry, just report
            return {"question": question, "sql": sql, "columns": [], "rows": [],
                    "answer": "I can only run read-only SELECT queries, so I couldn't do that.",
                    "attempts": attempt, "error": f"blocked: {e}"}
        except Exception as e:        # SQL error from Postgres: let the LLM see it and try again
            error = str(e).strip()

    return {"question": question, "sql": sql, "columns": [], "rows": [],
            "answer": "Sorry, I couldn't produce a working query for that question.",
            "attempts": MAX_ATTEMPTS, "error": error}
