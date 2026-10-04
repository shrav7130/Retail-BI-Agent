from datetime import date

import pytest

import app.evaluation as ev
from app.db import get_data_range, run_query


def result(answer="", tools=(), error=None):
    return {"answer": answer, "error": error, "tool_calls": [{"tool": t, "input": {}, "output": ""} for t in tools]}


# ---------- number and text matching ----------
@pytest.mark.parametrize("text, expected", [
    ("Revenue was ₹918,052.00", 918052),
    ("Revenue was Rs 9,18,052", 918052),          # Indian digit grouping
    ("about 9.18 lakh rupees", 918052),
    ("roughly 3.63 million", 3632064.45),
    ("sales fell by -16.5%", 16.49),
    ("down 42.7 %", 42.7),
])
def test_has_number_accepts_common_formats(text, expected):
    assert ev.has_number(text, expected)


@pytest.mark.parametrize("text, expected", [
    ("Revenue was ₹918,052", 1516080), ("fell 20%", 16.49), ("no digits here", 5), ("about 3.2 million", 3632064.45),
])
def test_has_number_rejects_wrong_values(text, expected):
    assert not ev.has_number(text, expected)


def test_norm_unifies_lookalike_characters():
    assert ev.norm("Noise\u2011Cancelling  Headphones") == "noise-cancelling headphones"
    assert ev.norm("I can\u2019t do that") == "i can't do that"
    assert ev.norm("**Regional managers**") == "regional managers"


# ---------- grade() ----------
Q = dict(id="X", must_use=["run_sql_query"], must_not_use=["search_company_docs"], facts_all=["Pune Camp"], numbers=[918052])


def test_grade_pass():
    assert ev.grade(Q, result("Pune Camp: ₹918,052", ["run_sql_query"])) == ("PASS", [])


def test_grade_reports_every_problem():
    status, reasons = ev.grade(Q, result("Mumbai had ₹5", ["search_company_docs"]))
    assert status == "FAIL" and len(reasons) == 4
    assert any("did not call run_sql_query" in r for r in reasons)
    assert any("should not have called search_company_docs" in r for r in reasons)
    assert any("Pune Camp" in r for r in reasons) and any("918,052" in r for r in reasons)


def test_grade_facts_any_and_none_and_no_tools_and_min_sql():
    q = dict(id="Y", facts_any=["a", "b"], facts_none=["paris"], no_tools=True, min_sql=2)
    assert ev.grade(q, result("contains b", []))[0] == "FAIL"                       # min_sql unmet
    assert ev.grade(dict(id="Y", facts_any=["a", "b"]), result("contains b"))[0] == "PASS"
    assert ev.grade(dict(id="Y", facts_any=["a", "b"]), result("nothing"))[0] == "FAIL"
    assert ev.grade(dict(id="Y", facts_none=["paris"]), result("It is Paris"))[0] == "FAIL"
    assert ev.grade(dict(id="Y", no_tools=True), result("x", ["calculator"]))[0] == "FAIL"


def test_provider_errors_are_not_counted_as_quality_failures():
    assert ev.grade(Q, result(error="OpenAIRateLimitError: 429"))[0] == "ERROR"
    assert ev.grade(Q, result(error="recursion_limit"))[0] == "FAIL"
    assert ev.grade(Q, result(error="EmptyAnswer: no text"))[0] == "FAIL"


# ---------- the question set itself ----------
def test_question_set_is_well_formed():
    ids = [q["id"] for q in ev.QUESTIONS]
    assert len(ids) == len(set(ids)) == 21
    assert {q["group"] for q in ev.QUESTIONS} == {"data", "policy", "combined", "edge", "followup"}
    tools = {"run_sql_query", "search_company_docs", "get_database_schema", "calculate_change", "calculator"}
    for q in ev.QUESTIONS:
        assert q["question"].strip()
        assert set(q.get("must_use", []) + q.get("must_not_use", [])) <= tools
        assert not set(q.get("must_use", [])) & set(q.get("must_not_use", []))


def test_every_question_can_be_failed_by_an_empty_answer():
    """A grader that passes a blank answer is useless."""
    for q in ev.QUESTIONS:
        if q.get("facts_all") or q.get("facts_any") or q.get("numbers"):
            used = q.get("must_use", [])
            assert ev.grade(q, result("", used))[0] == "FAIL", q["id"]


# ---------- answer key vs. the real database ----------
@pytest.mark.skipif(get_data_range() != (date(2025, 10, 1), date(2026, 9, 30)),
                    reason="answer key is for data generated in October 2026")
def test_answer_key_matches_the_database():
    def one(sql):
        return run_query(sql)[1][0][0]
    month = "order_date >= '2026-09-01' AND order_date < '2026-10-01'"
    assert float(one(f"SELECT SUM(total_amount) FROM orders WHERE {month}")) == 5892903.00
    assert float(one("SELECT SUM(total_amount) FROM orders WHERE order_date >= '2026-08-01' AND order_date < '2026-09-01'")) == 7056798.45
    assert float(one("SELECT SUM(total_amount) FROM orders WHERE order_date >= '2026-07-01' AND order_date < '2026-08-01'")) == 6789019.40
    assert round(float(one(f"SELECT AVG(total_amount) FROM orders WHERE {month}")), 2) == 3207.89
    store = run_query(f"SELECT s.store_name, SUM(o.total_amount) r FROM orders o JOIN stores s USING(store_id) WHERE {month} GROUP BY 1 ORDER BY 2")[1]
    assert (store[0][0], float(store[0][1])) == ("Pune Camp", 918052.0)
    assert (store[-1][0], float(store[-1][1])) == ("Mumbai Central", 1516080.0)
    assert round((5892903.00 - 7056798.45) / 7056798.45 * 100, 2) == -16.49


# ---------- the runner ----------
def test_run_evaluation_with_a_fake_agent_and_report(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "_order_count", lambda: 23717)
    calls = []

    def fake_runner(question, session_id=None):
        calls.append((question, session_id))
        if "capital" in question:
            return result("I can only help with business data.", [])
        return result("", [], error="OpenAIRateLimitError: 429")

    report = tmp_path / "r.md"
    rows = ev.run_evaluation(only=["E2", "D1", "F1", "F2"], delay=0, report_path=str(report), runner=fake_runner)
    assert [r["id"] for r in rows] == ["D1", "E2", "F1", "F2"]            # runs in question-set order
    assert [r["status"] for r in rows] == ["ERROR", "PASS", "ERROR", "ERROR"]
    by_id = {r["id"]: r["status"] for r in rows}
    assert by_id["E2"] == "PASS" and by_id["D1"] == "ERROR"
    sessions = {q: s for q, s in calls}
    assert sessions["Which store had the lowest sales last month?"] == sessions["And which one had the highest?"]   # shared
    assert sessions["What's the capital of France?"] != sessions["What were our top 5 products last month?"]        # isolated
    text = report.read_text(encoding="utf-8")
    assert "# Evaluation report" in text and "unchanged" in text and "| E2 |" in text and "Manual review" in text


def test_report_flags_a_changed_database(monkeypatch, tmp_path):
    counts = iter([100, 99])
    monkeypatch.setattr(ev, "_order_count", lambda: next(counts))
    report = tmp_path / "r.md"
    ev.run_evaluation(only=["E2"], delay=0, report_path=str(report), runner=lambda q, session_id=None: result("ok"))
    assert "CHANGED!" in report.read_text(encoding="utf-8")
