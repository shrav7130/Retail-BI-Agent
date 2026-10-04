"""Try the baseline from the terminal.

  python -m app.cli                      # interactive
  python -m app.cli "Top 5 products last month?"
"""
import sys

from app.text_to_sql import answer_question


def show(result: dict):
    print("\nSQL:\n" + (result["sql"] or "(none)"))
    if result["rows"]:
        print("\nRows:")
        print(" | ".join(result["columns"]))
        for row in result["rows"][:10]:
            print(" | ".join(str(v) for v in row))
        if len(result["rows"]) > 10:
            print(f"... ({len(result['rows'])} rows total)")
    print(f"\nAnswer: {result['answer']}")
    if result["error"]:
        print(f"[debug] {result['error']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(answer_question(" ".join(sys.argv[1:])))
    else:
        print("Ask a business question (or 'quit').")
        while True:
            q = input("> ").strip()
            if q.lower() in ("quit", "exit", ""):
                break
            show(answer_question(q))
