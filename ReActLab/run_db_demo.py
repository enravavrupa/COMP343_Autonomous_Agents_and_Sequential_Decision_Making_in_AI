"""
run_db_demo.py
====================
A minimal ReAct-style loop wired to sriroute_tools_db.py -- so tool
calls read and write sriroute_fake.db (a real SQLite file) instead of
an in-memory dict.

Self-contained: it plays back a fixed script of JSON tool calls (a
stand-in for what an LLM would emit), the same shape LAB_HANDOUT.md
describes -- {"thought": ..., "action": ..., "action_input": {...}} --
parses each with json.loads, validates it against the Pydantic models
in sriroute_tools.py, and only then calls the matching SQLite-backed
tool. This shows the JSON -> Pydantic -> SQL chain end to end without
needing a live LLM.

Run it directly:

    python3 run_db_demo.py

It assigns VAN-04 to REQ-901, then prints the `assignments` table so
you can see that assignment actually persisted to disk.
"""

import json
from pydantic import ValidationError

from sriroute_tools import TOOL_ARG_MODELS
from sriroute_tools_db import make_tool_registry
from fake_db import build_database, DB_PATH

MAX_ITERATIONS = 8

# A fixed sequence of steps, playing the role an LLM's responses would --
# see LAB_HANDOUT.md for the full reasoning behind this shape.
SCRIPT = [
    {"thought": "Look up the request first.", "action": "get_delivery_request",
     "action_input": {"request_id": "REQ-901"}},
    {"thought": "Check what's available in South Chennai.", "action": "check_fleet_availability",
     "action_input": {"zone": "South Chennai"}},
    {"thought": "VAN-04 has plenty of capacity for 4kg.", "action": "assign_courier",
     "action_input": {"request_id": "REQ-901", "vehicle_id": "VAN-04"}},
    {"thought": "Done.", "action": "finish",
     "action_input": {"final_answer": "REQ-901 assigned to VAN-04."}},
]


def run_loop(script, request_text: str, max_iterations=MAX_ITERATIONS, verbose=True):
    tools = make_tool_registry()
    messages = [{"role": "user", "content": request_text}]

    for iteration in range(max_iterations):
        if iteration >= len(script):
            raise RuntimeError("Script ran out of steps before calling finish.")
        raw_response = json.dumps(script[iteration])
        messages.append({"role": "assistant", "content": raw_response})

        try:
            step = json.loads(raw_response)
        except json.JSONDecodeError:
            obs = ('ERROR: your response was not valid JSON. Respond ONLY with a JSON '
                   'object: {"thought": ..., "action": ..., "action_input": {...}}.')
            messages.append({"role": "user", "content": obs})
            if verbose:
                print(f"[iter {iteration}] JSON parse error")
            continue

        action = step.get("action")
        action_input = step.get("action_input", {})

        if action not in TOOL_ARG_MODELS:
            obs = f"ERROR: unknown action '{action}'. Valid actions are: {list(TOOL_ARG_MODELS.keys())}."
            messages.append({"role": "user", "content": obs})
            if verbose:
                print(f"[iter {iteration}] unknown action: {action}")
            continue

        model_cls = TOOL_ARG_MODELS[action]
        try:
            validated = model_cls(**action_input)
        except ValidationError as e:
            obs = f"ERROR: invalid arguments for '{action}': {e}"
            messages.append({"role": "user", "content": obs})
            if verbose:
                print(f"[iter {iteration}] validation error for {action}")
            continue

        if action == "finish":
            if verbose:
                print(f"[iter {iteration}] finish")
            return validated.final_answer, messages

        tool_fn = tools[action]
        result = tool_fn(**validated.model_dump())
        obs = f"Observation: {json.dumps(result)}"
        messages.append({"role": "user", "content": obs})
        if verbose:
            print(f"[iter {iteration}] {action}({validated.model_dump()}) -> {result}")

    raise RuntimeError(f"Exceeded max_iterations={max_iterations} without calling finish.")


if __name__ == "__main__":
    build_database()  # fresh sriroute_fake.db every run, same as fake_db.py's own demo
    print(f"Built {DB_PATH}\n")

    final_answer, transcript = run_loop(SCRIPT, "Fulfill pending order REQ-901.")
    print("\nFinal answer:", final_answer)

    print("\n--- assignments table after the run ---")
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    for row in conn.execute("SELECT * FROM assignments"):
        print(dict(row))
    conn.close()
