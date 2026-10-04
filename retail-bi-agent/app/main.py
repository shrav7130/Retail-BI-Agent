"""FastAPI backend for the BI agent.

Run:   uvicorn app.main:app --reload
Docs:  http://127.0.0.1:8000/docs   (interactive Swagger UI: try the endpoints from the browser)
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent import get_agent, run_agent
from app.db import get_ro_connection
from app.rag import get_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bi-agent")

MAX_TOOL_OUTPUT_CHARS = 2000   # keep API responses small; the full output stays in the server logic


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: build the agent and the vector index now, so the first user isn't the one waiting."""
    try:
        get_vectorstore()
        get_agent()
        log.info("Agent and document index are ready.")
    except Exception:
        # Don't crash: /health will show the problem and /chat will return a clear error.
        log.exception("Warm-up failed. Check your .env (API key, database) and that Postgres is running.")
    yield


app = FastAPI(
    title="Retail BI Agent",
    description="Ask business questions in plain English. A LangChain agent decides whether to query "
                "PostgreSQL, search company policy documents, or run calculations, then explains its answer.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- request / response models ----------
class ChatRequest(BaseModel):
    # Swagger pre-fills the request body from this example. Without it, Swagger would fill session_id
    # with the literal text "string", and every test request would share one conversation.
    model_config = ConfigDict(json_schema_extra={"examples": [{"question": "Why did sales decline last month?"}]})

    question: str = Field(min_length=1, max_length=1000, examples=["Why did sales decline last month?"])
    session_id: str | None = Field(
        default=None, max_length=100,
        description="Omit it to start a new conversation. Send back the session_id you received to continue it.")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class ToolCall(BaseModel):
    tool: str
    input: dict
    output: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sql_queries: list[str]
    tool_calls: list[ToolCall]
    elapsed_seconds: float


# ---------- endpoints ----------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Liveness check that also verifies the database connection. Does NOT call the LLM (that would cost money)."""
    try:
        conn = get_ro_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        finally:
            conn.close()
    except Exception as e:
        log.warning("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "database": "down"})
    return {"status": "ok", "database": "up"}


# Plain `def`, not `async def`: the agent makes slow blocking calls (LLM, database), and FastAPI runs
# plain functions in a thread pool, so one slow question doesn't freeze the whole server.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Ask the agent a question. The response includes the answer AND which tools/SQL it used."""
    started = time.perf_counter()
    result = run_agent(req.question, session_id=req.session_id)
    elapsed = round(time.perf_counter() - started, 2)

    tools_used = [c["tool"] for c in result["tool_calls"]]
    log.info("session=%s elapsed=%.1fs tools=%s error=%s", result["session_id"], elapsed, tools_used, result["error"])

    # "Stuck in a loop" still produces a user-friendly answer, so it's a normal 200.
    # Rate limited by the LLM provider -> 429 (wait and retry). Any other upstream failure -> 502.
    if result["error"] and result["error"] != "recursion_limit":
        err = result["error"].lower().replace(" ", "")
        rate_limited = "ratelimit" in err or "429" in err
        raise HTTPException(status_code=429 if rate_limited else 502, detail={
            "message": result["answer"], "session_id": result["session_id"], "error": result["error"]})

    return ChatResponse(
        session_id=result["session_id"],
        answer=result["answer"],
        sql_queries=result["sql_queries"],
        tool_calls=[ToolCall(tool=c["tool"], input=c["input"], output=c["output"][:MAX_TOOL_OUTPUT_CHARS])
                    for c in result["tool_calls"]],
        elapsed_seconds=elapsed,
    )
