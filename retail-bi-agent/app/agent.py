"""The agent: an LLM that decides which tools to call, and in what order, until it can answer.

   question -> LLM -> (calls a tool -> reads result -> decides next step)* -> final answer

Conversation memory: every call carries a session_id. The checkpointer stores that session's messages,
so follow-ups like "what about Pune?" work.
"""
import logging
import os
import uuid
from datetime import date
from functools import lru_cache

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from app.db import get_data_range
from app.tools import ALL_TOOLS

log = logging.getLogger("bi-agent")

# Some smaller models occasionally emit a malformed tool call, which the provider rejects with this code.
# It is random, so one automatic retry usually fixes it.
TRANSIENT_MARKER = "output_parse_failed"

# Safety net against endless loops. One tool call uses ~2 steps, so 24 allows roughly 10-11 tool calls.
MAX_STEPS = 24


def build_system_prompt() -> str:
    first, last = get_data_range()
    return f"""You are the Business Intelligence assistant for UrbanCart Retail. You answer questions using
(1) the sales database (PostgreSQL) and (2) the company's policy documents.

TODAY: {date.today().isoformat()}
DATA RANGE: orders exist from {first} to {last}. Nothing outside this range exists. If a question covers a
period that is only partly inside it (for example "this year"), say which period the numbers really cover.

TOOLS - choose deliberately:
- run_sql_query: every number from the data (sales, rankings, counts, trends, discounts). If unsure about
  table or column names, call get_database_schema first.
- search_company_docs: policies (returns, refunds, discount rules), promotions and their dates, store
  operations (renovations, closures, hours). It does NOT contain sales numbers.
- calculate_change / calculator: use for ONE comparison (for example total revenue this month vs last month) and
  for arithmetic you cannot do in SQL. When comparing MANY rows (every category, store or product), compute the
  change inside the SQL query, for example ROUND((cur - prev) / NULLIF(prev, 0) * 100, 2) AS pct_change.
  Never call a calculator once per row.
You may call several tools in sequence. Do not call tools you do not need: a pure policy question needs no SQL,
and a pure data question needs no documents.

BE EFFICIENT. You have a limited step budget (aim for at most 8 tool calls per question).
- Prefer ONE query that returns both periods side by side (conditional aggregation with CASE WHEN or
  SUM(...) FILTER (WHERE ...)) over several separate queries.
- Do not repeat a query whose result you already have.
- As soon as you have what you need, stop investigating and write the answer.

SQL RULES:
- "Sales" or "revenue" = SUM(order_items.line_total) (already after discounts). Join orders for order_date and
  store_id, products for category, stores for store_name.
- order_items.discount_pct is a percentage (15.00 = 15% off). Amounts are in Indian rupees.
- "Last month" = order_date >= date_trunc('month', CURRENT_DATE) - interval '1 month'
                 AND order_date < date_trunc('month', CURRENT_DATE).
- Prefer aggregate queries with GROUP BY, and LIMIT any list. Only SELECT is allowed.
- If run_sql_query returns an ERROR, read it, fix the query and retry (at most twice).

"WHY" QUESTIONS (for example "Why did sales decline last month?"): investigate before answering.
1. One SQL query returning both periods in total, then calculate_change once for the overall change.
2. One SQL query by category and one by store, each showing both periods and the percentage change, to find
   what drove the change (add products only if needed).
3. Search the company documents once or twice with broad queries (for example "promotions and renovations in
   September 2026"), not once per category or store. Look for events that explain the biggest drivers
   (a promotion that ended, a renovation, a closure).
4. Answer with: how big the change is, the main drivers with numbers, and the documented events that
   likely contributed. Call them "likely contributing factors", not proven causes: the documents show
   that an event happened in that period, not that it produced those exact numbers.

OTHER RULES:
- Never invent numbers. Every figure must come from a tool result.
- Answer policy questions only from retrieved document text, and name the document. If the documents do not
  cover it, say so instead of guessing.
- You are read-only. If asked to change data (delete, update, insert, drop), refuse briefly and say you can only read data.
- If the question is unrelated to the business data or company policies, say that you can only help with those.
- Be concise: lead with the answer, then 3-8 short lines. Use the rupee sign and thousands separators.
- Follow-up questions may refer to earlier turns ("what about Pune?"); use the conversation so far.
"""


def get_agent_llm() -> ChatOpenAI:
    """The agent makes MANY model calls per question, so rate limits matter. max_retries makes the client
    wait and retry on 429 (rate limit) instead of failing. Raise LLM_MAX_RETRIES if you still see 429 errors."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "6")),
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
    )


def build_agent(model=None, checkpointer=None):
    return create_agent(
        model=model or get_agent_llm(),
        tools=ALL_TOOLS,
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer or InMemorySaver(),
    )


@lru_cache(maxsize=1)
def get_agent():
    """One agent (and one memory store) per process."""
    return build_agent()


# ---------- reading the result ----------
def _text(content) -> str:
    """Message content is a string, or (with some providers) a list of text parts."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def _this_turn(messages: list, question_id: str | None = None) -> list:
    """The stored history holds all earlier turns too; keep only what came after THIS question."""
    for i, m in enumerate(messages):
        if question_id and getattr(m, "id", None) == question_id:
            return messages[i + 1:]
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[last_human + 1:]


def _trace(turn: list) -> list[dict]:
    """Pair every tool call the LLM made with the output the tool returned."""
    outputs = {m.tool_call_id: _text(m.content) for m in turn if isinstance(m, ToolMessage)}
    trace = []
    for m in turn:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                trace.append({"tool": call["name"], "input": call["args"], "output": outputs.get(call["id"], "")})
    return trace


def _final_answer(turn: list) -> str:
    """The last assistant message that has real text and no pending tool calls ('' if there is none)."""
    for m in reversed(turn):
        if isinstance(m, AIMessage) and not m.tool_calls:
            text = _text(m.content).strip()
            if text:
                return text
    return ""


def _snapshot_turn(agent, config, question_id) -> list:
    """What the agent did so far, read from its saved state. Works even after a crash or a loop."""
    try:
        messages = agent.get_state(config).values.get("messages", [])
        return _this_turn(messages, question_id) if messages else []
    except Exception:
        return []


GENERIC_ERROR = "Sorry, something went wrong while answering. Please try again."
LOOP_ERROR = "I ran out of steps before finishing that investigation. Try asking something narrower."
WRAP_UP_NUDGE = ("You have used your step budget. Do NOT call any more tools. Using only the results you already "
                 "have, write the final answer to my question now, and say briefly if something could not be checked.")
PARTIAL_NOTE = "\n\n(Note: I reached my step limit, so this answer may not cover every angle.)"
EMPTY_ERROR = "The model finished without writing an answer. Please try again."
FINALIZE_NUDGE = ("You stopped without writing an answer. Using the tool results above, "
                  "write the final answer to my question now.")


def run_agent(question: str, session_id: str | None = None, agent=None) -> dict:
    """Ask the agent one question. Never raises: failures come back as a dict with an 'error' field.
    Even on failure, 'tool_calls' shows what the agent managed to do, so you can see where it went wrong."""
    session_id = session_id or str(uuid.uuid4())
    question_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": MAX_STEPS}
    base = {"session_id": session_id, "question": question, "tool_calls": [], "sql_queries": []}

    def sql_of(trace):
        return [t["input"].get("sql", "") for t in trace if t["tool"] == "run_sql_query"]

    def failure(error, answer):
        trace = _trace(_snapshot_turn(agent, config, question_id)) if agent is not None else []
        log.warning("Agent failed (%s). Tool calls before failing: %s", error[:120],
                    [(t["tool"], str(t["input"])[:60]) for t in trace])
        return {**base, "tool_calls": trace, "sql_queries": sql_of(trace), "error": error, "answer": answer}

    def invoke(message):
        return agent.invoke({"messages": [message]}, config=config)

    def wrap_up():
        """Out of steps: ask ONCE for a best-effort answer from what was already gathered. The saved history is
        safe to continue from: it always ends with a finished tool result, never a half-done tool call."""
        log.warning("Hit the step limit; asking the model to wrap up with what it has.")
        try:
            res = agent.invoke({"messages": [HumanMessage(WRAP_UP_NUDGE)]}, config={**config, "recursion_limit": 6})
        except Exception:
            return failure("recursion_limit", LOOP_ERROR)
        wrapped_turn = _this_turn(res["messages"], question_id)
        text = _final_answer(wrapped_turn)
        if not text:
            return failure("recursion_limit", LOOP_ERROR)
        wrapped_trace = _trace(wrapped_turn)
        return {**base, "answer": text + PARTIAL_NOTE, "tool_calls": wrapped_trace,
                "sql_queries": sql_of(wrapped_trace), "error": None}

    try:
        agent = agent or get_agent()   # a missing API key or DB error must not escape
    except Exception as e:
        return failure(f"{type(e).__name__}: {e}", GENERIC_ERROR)

    result = None
    for attempt in (1, 2):
        try:
            # Same message id on the retry, so the question is replaced, not duplicated, in the history.
            result = invoke(HumanMessage(question, id=question_id))
            break
        except GraphRecursionError:
            return wrap_up()
        except Exception as e:  # rate limits, network errors, provider problems
            if attempt == 1 and TRANSIENT_MARKER in str(e):
                log.warning("Model produced an unparseable tool call; retrying once.")
                continue
            return failure(f"{type(e).__name__}: {e}", GENERIC_ERROR)

    turn = _this_turn(result["messages"], question_id)
    answer = _final_answer(turn)

    if not answer:  # some models occasionally end their turn with an empty message; ask once for the answer
        log.warning("Model ended with an empty answer; asking it to write one.")
        try:
            result = invoke(HumanMessage(FINALIZE_NUDGE))
        except GraphRecursionError:
            return failure("recursion_limit", LOOP_ERROR)
        except Exception as e:
            return failure(f"{type(e).__name__}: {e}", GENERIC_ERROR)
        turn = _this_turn(result["messages"], question_id)
        answer = _final_answer(turn)
        if not answer:
            return failure("EmptyAnswer: the model returned no text", EMPTY_ERROR)

    trace = _trace(turn)
    return {**base, "answer": answer, "tool_calls": trace, "sql_queries": sql_of(trace), "error": None}
