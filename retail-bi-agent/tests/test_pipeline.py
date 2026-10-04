"""Tests the pipeline logic with a FAKE LLM, so no API key or cost is needed."""
from types import SimpleNamespace

from app.text_to_sql import answer_question, extract_sql


class FakeLLM:
    """Returns scripted replies in order. The last reply repeats if we run out."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def invoke(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return SimpleNamespace(content=reply)


def test_extract_sql_strips_fences():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


def test_happy_path():
    llm = FakeLLM(["SELECT store_name FROM stores ORDER BY store_id LIMIT 2", "There are stores A and B."])
    r = answer_question("List two stores", llm=llm)
    assert r["error"] is None and r["attempts"] == 1 and len(r["rows"]) == 2
    assert r["answer"] == "There are stores A and B."


def test_retry_after_sql_error():
    llm = FakeLLM([
        "SELECT no_such_column FROM stores",             # fails in Postgres
        "SELECT store_name FROM stores LIMIT 1",         # LLM 'fixes' it
        "One store.",
    ])
    r = answer_question("any store", llm=llm)
    assert r["attempts"] == 2 and r["error"] is None and len(r["rows"]) == 1


def test_unsafe_sql_is_blocked_not_run():
    llm = FakeLLM(["DELETE FROM orders"])
    r = answer_question("delete everything", llm=llm)
    assert r["error"].startswith("blocked") and r["rows"] == []
