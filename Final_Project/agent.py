"""
agent.py - ReAct loop for the library bot. Takes a natural-language
availability/borrow request, looks up the catalog, physically checks
the shelf to reconcile against catalog/shelf disagreements, and verifies
fetch feasibility via Z3 before confirming availability to the user.

Usage (once implemented):
    cp .env.example .env       
    python agent.py available        
    python agent.py unavailable      
    python agent.py impossible      
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from verifier import verify_dispatch_smt

load_dotenv()  # reads OPENAI_API_KEY (etc.) from .env into os.environ -- see env-setup.md

from openai import OpenAI

import digital_database # digital database of books
import shelves # actual physical shelves database
from integrate import CheckSpec, check_shelf, compare_db

client = OpenAI()
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TURNS = 8            # Task 2: hard turn budget -- the loop must not run forever
DEFAULT_MODEL = "gpt-5-mini"  # or whatever model your SDK/key supports


# ---------------------------------------------------------------------------
# Given: the step schema every LLM turn must validate against (Task 2)
# ---------------------------------------------------------------------------

class AgentStep(BaseModel):
    thought: str
    action: str
    action_input: Dict[str, Any]


# ---------------------------------------------------------------------------
# Given: the three graded evaluation prompts (README.md, Section 6)
# ---------------------------------------------------------------------------

TEST_CASES = {
    "available": (
        "Is 'book-1' available? I want to borrow it"
    ),
    "unavailable": (
        "Is 'book-2' available? I want to borrow it"
    ),
    "impossible": (
        "Is 'book-3' available? I want to borrow it"
    ),
}


# ---------------------------------------------------------------------------
# TODO (Task 1): system prompt
#
# Write the system prompt here. It must, at minimum:
#   - mandate exactly one JSON object per turn, matching AgentStep, and
#     nothing else (no prose outside the JSON, no markdown fences)
#   - define exactly three actions and their action_input shape:
#       1. verify_dispatch_smt  -> {"packages": [...], "agvs": [...]}
#          each AGV may also carry a "drain_rate" (percentage points of
#          battery per kg carried, default 0) -- only the "drain" test
#          case (Task 4) states one in the request; parse it into the
#          payload the same way you already parse "minimum battery
#          threshold" into min_battery.
#       2. ask_user             -> {"question": "<str>"}
#       3. finish                -> {"answer": "<str>", "assignments": {...}}
#   - state Key Rule 1: after an UNSATISFIABLE observation, the agent is
#     forbidden from resubmitting the exact same action_input -- it must
#     reflect on the diagnostics and either propose a genuinely different
#     (relaxed) payload, or call ask_user if the right relaxation depends
#     on a user preference
#   - state Key Rule 2: never call finish claiming feasibility unless the
#     most recent verify_dispatch_smt observation for that exact plan was
#     SATISFIABLE
#
# See README.md Section 5 (Task 1) for the full requirements, and
# cheatsheet-react-smt-z3.md Section 1 for the loop shape this prompt
# needs to fit into.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an agent on a warehouse floor

You may respond with only one JSOn object per turn. It must be of the shape:
{"thought": "<your reasoning>", "action": "<action name>", "action_input": {<args>}}

The actions available to you are:

1. "action": lookup_digital_database -> "action_input": {"book_id": "<str>"}

2. "action": check_shelf -> "action_input": "action_input": {"book_id": "<str>", "shelf_id": "<str>"}
    The robot checks the shelf and compares that database with the digital database
    check_status is NULL when the check disagreed with the digital database but with such a low
    probability that case you do not know for sure even now

3. "action": get_robot_state -> "action_input": {"robot_id": "<str>"}
    
4. "action": verify_dispatch_smt  -> "action_input": {"books": [{"id": "<str>", "weight": <int>}, ...], "robots": [{"id": "<str>", "capacity": <int>, "battery_pct": <int>,
        "min_battery": <int>, "drain_rate": <float>}], ...}
        each robot may also carry a "drain_rate" (percentage points of
        battery per kg carried, default 0) -- only the "drain" test
        case (Task 4) states one in the request; parse it into the
        payload the same way you already parse "minimum battery
        threshold" into min_battery.

5. "action": ask_user -> "action_input": {"question": "<str>"}

6. "action": finish -> "action_input": {"answer": "<str>", "available": <bool>}

When you get available, call finish and do not loop further.

Please note after an unavailable observation, the agent is not allowed to submit the same action_input. It must either 
reflect on the diagnostic results or propose a different (relaxed) payload [remove overly large books] or call ask_user if the relxation depends
on the user's preference.

Please also note that you must never call finish while claiming feasibility unless the latest verify_dispatch_smt
observation on this plan was available

Please also never call finish with "available": True unless the most recent check_shelf's check_status was not None
"""


# ---------------------------------------------------------------------------
# TODO (Task 2): JSON / plumbing helpers
# ---------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Strip ```json ... ``` fences and any prose wrapped around the JSON
    object the model returned, so json.loads() gets a clean object.
    See cheatsheet-react-smt-z3.md Section 2 for the pattern."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1]


def freeze(action_input: Dict[str, Any]) -> str:
    """Canonical string key for a JSON-able dict -- used by SeenPayloads
    below to detect a repeated verify_dispatch_smt payload. Dicts aren't
    hashable, but json.dumps(d, sort_keys=True) is."""
    return json.dumps(action_input, sort_keys=True)


class SeenPayloads:
    """Provided: tracks every action_input you've already sent to
    verify_dispatch_smt, so your run() loop (Task 2) can short-circuit a
    resubmitted UNSAT payload in code, instead of relying on the prompt
    alone (README Section 5, Task 2's "Repeated-payload detection" bullet).
    You don't need to write this class -- just call `.seen(...)` before
    calling verify_dispatch_smt, and `.add(...)` right after, inside run()."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def seen(self, action_input: Dict[str, Any]) -> bool:
        return freeze(action_input) in self._seen

    def add(self, action_input: Dict[str, Any]) -> None:
        self._seen.add(freeze(action_input))


def parse_step(raw: str) -> AgentStep:
    """Parse + validate one model turn against AgentStep. Let
    pydantic.ValidationError propagate -- run() below handles it as the
    syntactic loop."""
    return AgentStep.model_validate_json(strip_fences(raw))


# ---------------------------------------------------------------------------
# TODO (Task 2): the LLM client
#
# A thin wrapper around a single raw API call. Keep it minimal -- this is
# NOT the place to add retries, agent frameworks, or extra abstraction.
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = client

    def complete(self, messages: List[Dict[str, Any]]) -> str:
        """Send the full message history, return the raw text of the
        model's reply for this turn (before any parsing/stripping)."""
        response = self.client.responses.create(
            model=self.model,
            input=messages,  # type: ignore[arg-type] # debugging done with google
        )
        return response.output_text


# ---------------------------------------------------------------------------
# TODO: ask_user action handler
# ---------------------------------------------------------------------------

def ask_user(question: str) -> str:
    """Prompt the human for a decision the agent can't make on its own.
    Non-interactive runs (e.g. automated grading) should fall back to a
    sensible default instead of blocking forever on stdin -- check
    sys.stdin.isatty()."""
    print(f"Question: {question}")
    if sys.stdin.isatty() is True:
        return input("Answer: ")
    else:
        return "The answer is not feasible, either relax contraints or wait"


# ---------------------------------------------------------------------------
# TODO (Task 2 + Task 3): the ReAct loop
#
# Must implement:
#   - message history that grows every turn (system, user, assistant,
#     user, assistant, ... -- observations go back in as "user" turns)
#   - the SYNTACTIC loop: catch pydantic.ValidationError from parse_step,
#     feed the validation error back to the model, retry -- never crash
#     the program on bad model JSON
#   - the LOGICAL loop: on an UNSATISFIABLE observation, feed the
#     `diagnostics` and `feedback` fields back verbatim; if the model
#     resubmits an action_input you've already seen for
#     verify_dispatch_smt, intercept it in code -- use the provided
#     SeenPayloads (call `.seen(...)` before verifying, `.add(...)` after)
#     rather than calling Z3 again or trusting the prompt alone
#   - a MAX_TURNS hard stop that raises/returns cleanly instead of
#     looping silently forever
#   - routing for all three actions: verify_dispatch_smt, ask_user, finish
#
# Return value: (result, messages) where `result` is the action_input
# passed to `finish`, and `messages` is the full trace for save_trace().
# ---------------------------------------------------------------------------

def run(llm, user_request: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request}]
    seen_payloads = set()

    for turn in range(MAX_TURNS):
        raw = llm.complete(messages)          # → AgentStep, or ValidationError

        try:
            step = parse_step(raw)
        except ValidationError as e:
            messages.append({"role": "assistant", "content" : raw})
            messages.append({"role": "user", "content" : json.dumps(e.errors())})
            continue  

        messages.append({"role": "assistant", "content" : raw})                           # <- Syntactic Loop

        if step.action == "finish":
            return step.action_input, messages
        

        elif step.action == "verify_dispatch_smt":
            key = freeze(step.action_input)
            if key in seen_payloads:
                observation = {"status": "rejected",
                "message": ("This was already rejected earlier. Either relax the contraints or call ask_user")}
                   # don't even call Z3 again
            else:
                seen_payloads.add(key)
                observation = verify_dispatch_smt(step.action_input)
                  # <- Logical Loop

            messages.append({"role": "user", "content": json.dumps(observation)})

        elif step.action == "ask_user":
            observation = ask_user(step.action_input["question"])
            messages.append({"role": "user", "content": json.dumps(observation)})

        elif step.action == "check_shelf":
            book_id = step.action_input["book_id"]
            shelf_id = step.action_input["shelf_id"]
            history = digital_database.get_book_record(book_id)
            if history is None:
                observation = {"error": "not_found"}
            else:
                check = check_shelf(book_id, shelf_id)
                verified = CheckSpec(book_id=book_id, available=check.available, correct_prob=check.correct_prob)
                observation = compare_db(history["status"], verified)
            messages.append({"role": "user", "content": json.dumps(observation)})

        elif step.action == "get_robot_state":
            robot_state = digital_database.get_robot_state(step.action_input["robot_id"])
            if robot_state is None:
                observation = {"error": "not_found"}
            else:
                observation = {"id": robot_state["id"],
                "capacity": robot_state["capacity_kg"],
                "min_battery": 5,
                "drain_rate": 0.01}
            messages.append({"role": "user", "content": json.dumps(observation)})

        elif step.action == "lookup_digital_database":
            observation = digital_database.get_book_record(step.action_input["book_id"])
            if observation is None:
                observation = {"error": "not_found"}
            messages.append({"role": "user", "content": json.dumps(observation)})

        else:
            messages.append({"role": "user", "content": "Invalid Action"})
    raise TimeoutError("MAX_TURNS exceeded without finish")



def save_trace(path: str, messages: List[Dict[str, str]]) -> None:
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)
    print(f"Saved trace to {path}")


def main() -> None:
    args = sys.argv[1:]
    case = args[0] if args else "available"
    if case not in TEST_CASES:
        raise SystemExit(f"Unknown test case '{case}', expected one of {list(TEST_CASES)}")

    user_request = TEST_CASES[case]

    llm = LLMClient()
    trace_path = f"trace_{case}.json"

    result, messages = run(llm, user_request)
    save_trace(trace_path, messages)

    print("\nFinal result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
