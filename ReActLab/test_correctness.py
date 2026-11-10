"""
test_correctness.py
=====================
Run this against YOUR OWN react_loop_lab.py.

RECREATED and adapted to the SQLite backend: react_loop_lab.py now calls
sriroute_tools_db.make_tool_registry() under the hood, so these same
five checks exercise the SQLite-backed tools. build_database() is called
once up front so every run starts from the same known data, regardless
of what a previous run of fake_db.py or run_db_demo.py left in
sriroute_fake.db.

Five checks, matching the lab handout's checkpoints:
  1. Happy path
  2. Constraint-gated branching (infeasible request)
  3. Malformed tool-call recovery (the break-it case)
  4/5. HITL surge approval, both the approved and denied branches

If any of these fail, add verbose=True (it's on by default) and read the
printed transcript -- it shows exactly which observation your loop
produced at each step, which is almost always more informative than the
assertion message alone.
"""

from fake_db import build_database
from react_loop_lab import run_react_loop
from scripted_llm import (happy_path_script, constraint_gated_script,
                           malformed_then_recover_script, surge_script)


def test_happy_path():
    final_answer, transcript = run_react_loop(happy_path_script(), "Fulfill pending order REQ-901.")
    assert "VAN-04" in final_answer, f"Expected VAN-04 in final answer, got: {final_answer}"
    print("happy_path: PASS")


def test_constraint_gated():
    final_answer, transcript = run_react_loop(constraint_gated_script(), "Fulfill pending order REQ-902.")
    assert "cannot" in final_answer.lower(), (
        f"Expected the agent to report infeasibility, got: {final_answer}")
    print("constraint_gated: PASS")


def test_malformed_recovery():
    final_answer, transcript = run_react_loop(malformed_then_recover_script(), "Fulfill pending order REQ-901.")
    assert "VAN-04" in final_answer, f"Expected recovery to VAN-04, got: {final_answer}"
    # Check the loop actually surfaced a validation error somewhere in the
    # transcript, rather than crashing or silently skipping the bad call.
    joined = " ".join(m["content"] for m in transcript if m["role"] == "user")
    assert "error" in joined.lower(), (
        "Expected a validation-error observation to appear in the transcript -- "
        "did your loop catch pydantic.ValidationError and feed it back as an observation?"
    )
    print("malformed_recovery: PASS")


def test_surge_approved():
    final_answer, transcript = run_react_loop(
        surge_script(expect_approval=True), "Fulfill pending order REQ-904.",
        approval_fn=lambda request_id, extra_cost, reason: True,
    )
    assert "PRIORITY-VAN-01" in final_answer, f"Expected PRIORITY-VAN-01, got: {final_answer}"
    print("surge_approved: PASS")


def test_surge_denied():
    final_answer, transcript = run_react_loop(
        surge_script(expect_approval=False), "Fulfill pending order REQ-904.",
        approval_fn=lambda request_id, extra_cost, reason: False,
    )
    assert "denied" in final_answer.lower(), f"Expected a denial outcome, got: {final_answer}"
    print("surge_denied: PASS")


if __name__ == "__main__":
    build_database()
    test_happy_path()
    test_constraint_gated()
    test_malformed_recovery()
    test_surge_approved()
    test_surge_denied()
    print("\nAll correctness checks passed.")
