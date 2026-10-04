"""API tests (FastAPI TestClient). The LLM is replaced by a scripted fake; everything else is real."""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app.main as main
from tests.test_agent import agent_with, call

client = TestClient(main.app)   # no `with`: skips the startup warm-up (tested separately below)


def fake_result(**over):
    base = {"session_id": "s1", "question": "q", "answer": "ok", "tool_calls": [], "sql_queries": [], "error": None}
    return {**base, **over}


# ---------- health ----------
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "database": "up"}


def test_health_returns_503_when_database_is_down(monkeypatch):
    def down():
        raise ConnectionError("db unreachable")
    monkeypatch.setattr(main, "get_ro_connection", down)
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["detail"]["database"] == "down"


# ---------- validation ----------
@pytest.mark.parametrize("body", [
    {}, {"question": ""}, {"question": "    "}, {"question": "x" * 1001}, {"question": 123}, {"session_id": "abc"},
])
def test_bad_requests_get_422(body):
    assert client.post("/chat", json=body).status_code == 422


# ---------- /chat with a scripted agent ----------
def test_chat_happy_path_shape(monkeypatch):
    agent, _ = agent_with(call("run_sql_query", sql="SELECT COUNT(*) AS n FROM stores"), AIMessage(content="There are 5 stores."))
    monkeypatch.setattr("app.agent.get_agent", lambda: agent)
    r = client.post("/chat", json={"question": "How many stores do we have?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "There are 5 stores."
    assert body["sql_queries"] == ["SELECT COUNT(*) AS n FROM stores"]
    assert body["tool_calls"][0]["tool"] == "run_sql_query" and "5" in body["tool_calls"][0]["output"]
    assert body["session_id"] and body["elapsed_seconds"] >= 0


def test_follow_up_with_same_session_id_remembers(monkeypatch):
    agent, model = agent_with(AIMessage(content="Noted."))
    monkeypatch.setattr("app.agent.get_agent", lambda: agent)
    first = client.post("/chat", json={"question": "Remember: my focus store is Pune Camp"}).json()
    client.post("/chat", json={"question": "Which store is my focus?", "session_id": first["session_id"]})
    assert "Pune Camp" in str(model.seen[-1])
    client.post("/chat", json={"question": "Which store is my focus?"})        # no session_id -> new conversation
    assert "Pune Camp" not in str(model.seen[-1])


def test_question_is_trimmed(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "run_agent", lambda q, session_id=None: seen.setdefault("q", q) and fake_result())
    client.post("/chat", json={"question": "   hello   "})
    assert seen["q"] == "hello"


# ---------- error mapping ----------
def test_upstream_failure_is_502(monkeypatch):
    monkeypatch.setattr(main, "run_agent", lambda q, session_id=None: fake_result(
        error="AuthenticationError: 401 invalid api key", answer="Sorry, something went wrong while answering. Please try again."))
    r = client.post("/chat", json={"question": "hi"})
    assert r.status_code == 502 and r.json()["detail"]["error"].startswith("AuthenticationError")


def test_agent_loop_is_a_friendly_200(monkeypatch):
    monkeypatch.setattr(main, "run_agent", lambda q, session_id=None: fake_result(
        error="recursion_limit", answer="I got stuck going in circles on that question."))
    r = client.post("/chat", json={"question": "hi"})
    assert r.status_code == 200 and "circles" in r.json()["answer"]


def test_huge_tool_output_is_truncated_in_response(monkeypatch):
    big = [{"tool": "run_sql_query", "input": {"sql": "SELECT 1"}, "output": "x" * 5000}]
    monkeypatch.setattr(main, "run_agent", lambda q, session_id=None: fake_result(tool_calls=big))
    r = client.post("/chat", json={"question": "hi"})
    assert len(r.json()["tool_calls"][0]["output"]) == main.MAX_TOOL_OUTPUT_CHARS


# ---------- app-level ----------
def test_startup_survives_a_failed_warmup(monkeypatch):
    def boom():
        raise RuntimeError("no embeddings model")
    monkeypatch.setattr(main, "get_vectorstore", boom)
    with TestClient(main.app) as c:                       # runs the lifespan
        assert c.get("/health").status_code == 200


def test_docs_and_openapi_are_available():
    assert client.get("/openapi.json").json()["paths"].keys() >= {"/chat", "/health"}
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"


def test_rate_limit_maps_to_429(monkeypatch):
    monkeypatch.setattr(main, "run_agent", lambda q, session_id=None: fake_result(
        error="OpenAIRateLimitError: Error code: 429", answer="Sorry, something went wrong while answering."))
    assert client.post("/chat", json={"question": "hi"}).status_code == 429


def test_swagger_example_has_no_session_id_placeholder():
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ChatRequest"]
    assert schema["examples"] == [{"question": "Why did sales decline last month?"}]
