"""Multi-turn chat loop demonstrating OAS history threading.

Usage:
    python examples/chat-agent/chat.py

How it works
------------
OAS is stateless — it never stores conversation history.
This script maintains the `history` list and passes it back on every
invocation as a plain JSON input field.  The runner injects it between
the system prompt and the current user message before calling the model.

The pattern is:
1. User types a message.
2. Build input_data = {"message": ..., "history": [...prior turns...]}
3. Call oa.run_task(spec_path, "chat", input_data)
4. Append the user turn and the assistant reply to history.
5. Repeat.

Memory across sessions is out of scope for OAS.  For long-term memory,
delegate a retrieval step to oa://prime-vector/memory-retriever — see
examples/memory-chat/ for that pattern.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from oas_cli.runner import run_task  # noqa: E402


SPEC_PATH = Path(__file__).parent / "spec.yaml"


def main() -> None:
    history: list[dict] = []

    print("OAS chat-agent  (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        input_data: dict = {"message": user_input}
        if history:
            input_data["history"] = history

        result = run_task(SPEC_PATH, "chat", input_data)
        reply: str = result.get("reply", "")

        print(f"Agent: {reply}\n")

        # Append completed turn to history for the next call.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
