"""Tests the agent LOOP with a scripted fake LLM: tool calls, error handling, memory, limits.

These check the plumbing around the model. Whether the REAL model chooses sensible tools is what
you check by hand with `python -m app.agent_cli` (and, on Day 3, with the evaluation questions).
"""
import itertools

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import app.rag as rag
from app.agent import build_agent, build_system_prompt, run_agent
from app.db import get_data_range
from tests.test_rag import WordHashEmbeddings

_ids = itertools.count(1)


def call(tool, **args):
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": f"call_{next(_ids)}"}])


class ScriptedChat(BaseChatModel):
    """Plays back prepared replies in order (the last one repeats) and records what it was shown."""
    replies: list
    seen: list = Field(default_factory=list)
    i: int = 0
    boom: bool = False
    fail_first: int = 0          # raise this many times before behaving normally
    fail_message: str = "provider exploded"
    calls: int = 0
    fail_on_call: int = 0        # raise on exactly this call number (1-based); 0 = never

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise RuntimeError(self.fail_message)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError(self.fail_message)
        if self.boom:
            raise RuntimeError("provider exploded")
        self.seen.append(list(messages))
        reply = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        # A real LLM creates a brand-new message (with fresh tool-call ids) every turn; mimic that.
        fresh_calls = [{**c, "id": f"call_{next(_ids)}"} for c in reply.tool_calls]
        reply = reply.model_copy(update={"tool_calls": fresh_calls, "id": None})
        return ChatResult(generations=[ChatGeneration(message=reply)])


def agent_with(*replies, **kw):
    model = ScriptedChat(replies=list(replies), **kw)
    return build_agent(model=model), model


def test_system_prompt_contains_data_range_and_rules():
    first, last = get_data_range()
    prompt = build_system_prompt()
    assert str(first) in prompt and str(last) in prompt
    assert "read-only" in prompt and "Never invent numbers" in prompt
    assert "Never call a calculator once per row" in prompt and "BE EFFICIENT" in prompt


def test_single_tool_flow_and_trace():
    sql = "SELECT store_name FROM stores ORDER BY store_id LIMIT 1"
    agent, _ = agent_with(call("run_sql_query", sql=sql), AIMessage(content="The first store is Mumbai Central."))
    r = run_agent("Name a store", agent=agent)
    assert r["error"] is None and r["answer"] == "The first store is Mumbai Central."
    assert r["sql_queries"] == [sql]
    assert [t["tool"] for t in r["tool_calls"]] == ["run_sql_query"]
    assert "Mumbai Central" in r["tool_calls"][0]["output"]


def test_multi_tool_why_flow(monkeypatch):
    monkeypatch.setattr(rag, "get_vectorstore", lambda: rag.build_vectorstore(WordHashEmbeddings()))
    agent, _ = agent_with(
        call("run_sql_query", sql="SELECT 1 AS x"),
        call("calculate_change", old_value=7056798, new_value=5892903),
        call("search_company_docs", query="Electronics Clearance Promotion end date"),
        AIMessage(content="Sales fell 16.49%; the Electronics promo ended on 31 August 2026."),
    )
    r = run_agent("Why did sales decline?", agent=agent)
    assert [t["tool"] for t in r["tool_calls"]] == ["run_sql_query", "calculate_change", "search_company_docs"]
    assert "-16.49%" in r["tool_calls"][1]["output"]
    assert "discount_policy.md" in r["tool_calls"][2]["output"]
    assert "16.49%" in r["answer"]


def test_agent_survives_a_bad_query_and_continues():
    agent, _ = agent_with(
        call("run_sql_query", sql="SELECT nope FROM stores"),
        call("run_sql_query", sql="SELECT COUNT(*) AS n FROM stores"),
        AIMessage(content="There are 5 stores."),
    )
    r = run_agent("How many stores?", agent=agent)
    assert r["tool_calls"][0]["output"].startswith("ERROR:")
    assert not r["tool_calls"][1]["output"].startswith("ERROR")
    assert r["answer"] == "There are 5 stores." and r["error"] is None


def test_write_attempt_is_blocked_inside_the_agent():
    agent, _ = agent_with(call("run_sql_query", sql="DELETE FROM orders"), AIMessage(content="I can only read data."))
    r = run_agent("Delete everything", agent=agent)
    assert r["tool_calls"][0]["output"].startswith("ERROR: query blocked")


def test_conversation_memory_is_per_session():
    agent, model = agent_with(AIMessage(content="Noted."))
    run_agent("My favourite store is Pune Camp", session_id="A", agent=agent)
    run_agent("Which store did I mention?", session_id="A", agent=agent)
    assert "Pune Camp" in str(model.seen[-1])          # session A remembers turn 1
    run_agent("Which store did I mention?", session_id="B", agent=agent)
    assert "Pune Camp" not in str(model.seen[-1])      # session B starts empty


def test_endless_tool_loop_is_stopped():
    agent, _ = agent_with(call("calculator", expression="1+1"))   # the "LLM" never stops calling tools
    r = run_agent("loop forever", agent=agent)
    assert r["error"] == "recursion_limit" and "ran out of steps" in r["answer"]
    # the response must still show WHAT the agent kept doing, otherwise the failure can't be diagnosed
    assert len(r["tool_calls"]) >= 5 and {t["tool"] for t in r["tool_calls"]} == {"calculator"}


def test_provider_failure_returns_error_not_exception():
    agent, _ = agent_with(AIMessage(content="x"), boom=True)
    r = run_agent("hello", agent=agent)
    assert r["error"].startswith("RuntimeError") and "went wrong" in r["answer"]


def test_agent_construction_failure_returns_error_not_exception(monkeypatch):
    import app.agent as agent_module

    def broken():
        raise RuntimeError("OPENAI_API_KEY is missing")

    monkeypatch.setattr(agent_module, "get_agent", broken)
    r = run_agent("hello")
    assert r["error"].startswith("RuntimeError") and "went wrong" in r["answer"]


PARSE_FAIL = "Error code: 400 - output_parse_failed: model generated output that could not be parsed"


def test_malformed_tool_call_is_retried_once_and_recovers():
    agent, model = agent_with(AIMessage(content="Recovered."), fail_first=1, fail_message=PARSE_FAIL)
    r = run_agent("hello", agent=agent)
    assert r["error"] is None and r["answer"] == "Recovered." and model.calls == 2


def test_persistent_parse_failure_is_retried_only_once():
    agent, model = agent_with(AIMessage(content="x"), fail_first=99, fail_message=PARSE_FAIL)
    r = run_agent("hello", agent=agent)
    assert model.calls == 2 and r["error"] is not None


def test_other_errors_are_not_retried():
    agent, model = agent_with(AIMessage(content="x"), fail_first=99, fail_message="401 invalid api key")
    r = run_agent("hello", agent=agent)
    assert model.calls == 1 and "401" in r["error"]


def test_agent_llm_uses_env_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_MAX_RETRIES", "9")
    from app.agent import get_agent_llm
    llm = get_agent_llm()
    assert llm.max_retries == 9 and "groq" in llm.openai_api_base


def test_blank_final_answer_is_recovered_with_one_nudge():
    agent, model = agent_with(AIMessage(content=""), AIMessage(content="Here is the answer."))
    r = run_agent("hello", agent=agent)
    assert r["error"] is None and r["answer"] == "Here is the answer." and model.calls == 2


def test_blank_answer_after_tool_use_keeps_the_trace():
    agent, _ = agent_with(call("calculator", expression="2+2"), AIMessage(content=""), AIMessage(content="It is 4."))
    r = run_agent("what is 2+2", agent=agent)
    assert r["answer"] == "It is 4." and [t["tool"] for t in r["tool_calls"]] == ["calculator"]


def test_answer_that_stays_blank_is_reported_as_an_error():
    agent, model = agent_with(AIMessage(content=""))
    r = run_agent("hello", agent=agent)
    assert r["error"].startswith("EmptyAnswer") and model.calls == 2


def test_provider_failure_mid_run_still_returns_the_partial_trace():
    agent, _ = agent_with(call("calculator", expression="2+2"), AIMessage(content="4"),
                          fail_on_call=2, fail_message="502 upstream exploded")
    r = run_agent("what is 2+2", agent=agent)
    assert "502" in r["error"] and [t["tool"] for t in r["tool_calls"]] == ["calculator"]


def test_step_limit_triggers_one_wrap_up_that_returns_a_best_effort_answer():
    # 12 model calls use the whole 24-step budget; the 13th call is the wrap-up and returns text.
    agent, model = agent_with(*[call("calculator", expression="1+1") for _ in range(12)],
                              AIMessage(content="Best effort answer."))
    r = run_agent("big investigation", agent=agent)
    assert r["error"] is None and r["answer"].startswith("Best effort answer.")
    assert "step limit" in r["answer"] and len(r["tool_calls"]) >= 12
    assert model.calls == 13


def test_wrap_up_that_keeps_calling_tools_gives_up_cleanly():
    agent, _ = agent_with(call("calculator", expression="1+1"))        # never stops, even when asked to
    r = run_agent("big investigation", agent=agent)
    assert r["error"] == "recursion_limit" and "ran out of steps" in r["answer"]
