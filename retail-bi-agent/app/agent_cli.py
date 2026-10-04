"""Chat with the agent in the terminal. Shows every tool it calls, so you can see HOW it reasons.

  python -m app.agent_cli
Type 'new' to start a fresh conversation, 'quit' to leave.
"""
import json
import uuid

from app.agent import run_agent


def show(result: dict):
    print()
    if not result["tool_calls"]:
        print("  (no tools used)")
    for i, call in enumerate(result["tool_calls"], 1):
        args = call["input"]
        detail = args.get("sql") if call["tool"] == "run_sql_query" else json.dumps(args, ensure_ascii=False)
        print(f"  [{i}] {call['tool']}")
        if detail:
            print("      " + str(detail).strip().replace("\n", "\n      "))
        out = call["output"].strip().splitlines()
        preview = out[:4] + (["..."] if len(out) > 4 else [])
        print("      -> " + "\n         ".join(preview))
    print(f"\nAnswer:\n{result['answer']}")
    if result["error"]:
        print(f"\n[debug] {result['error']}")
    print()


if __name__ == "__main__":
    session = str(uuid.uuid4())
    print("Ask a question ('new' = new conversation, 'quit' = exit).")
    while True:
        q = input("> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "new":
            session = str(uuid.uuid4())
            print("Started a new conversation.")
            continue
        if q:
            show(run_agent(q, session_id=session))
