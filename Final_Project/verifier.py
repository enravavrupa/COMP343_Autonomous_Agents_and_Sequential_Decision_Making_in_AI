"""
verifier.py -- Z3 SMT feasibility check for the library fetch problem.
Given a set of books and robots, determines whether every book can be
assigned to some robot such that: (1) each book maps to exactly one
robot, (2) no book goes to a robot below its battery threshold, (3) no
robot's assigned books exceed its weight capacity, (4) carrying that
weight doesn't drain the robot's battery below its minimum.
"""

from __future__ import annotations

from typing import Any, Dict, List

import z3
from pydantic import BaseModel, Field


class BookSpec(BaseModel):
    id: str
    weight: int = Field(gt=0)


class RobotSpec(BaseModel):
    id: str
    capacity: int = Field(gt=0)
    battery_pct: int = Field(ge=0, le=100)
    min_battery: int = Field(default=20)
    # How many percentage points of battery it costs this robot to carry one
    # kg of assigned load. 0 (the default) means "no drain modeled" -- the
    # robot only needs to clear `min_battery` once, regardless of load. See
    # README Section 1 ("Extending the model") and Task 4 below.
    drain_rate: float = Field(default=0.0, ge=0)


class VerificationPayload(BaseModel):
    books: List[BookSpec]
    robots: List[RobotSpec]


def battery_drain_constraint(robot: RobotSpec, assigned_weight: "z3.ArithRef") -> "z3.BoolRef":
    """
    TODO (Task 4): return the z3 constraint expression for "carrying weight
    also costs battery" -- see README Section 1 ("Framing this as a word
    problem" / "Extending the model") for the relationship in words, and
    Section 5, Task 4 for exactly what's required.

    `robot` is the RobotSpec being checked; `assigned_weight` is a z3 arithmetic
    expression (already built for you, same one used in the capacity check
    below) representing the total weight assigned to this robot in the model
    being checked.

    Until you implement this, it returns an always-true constraint (a
    no-op) -- Test Cases 1 and 2 don't set `drain_rate`, so they pass
    either way; Test Case 3 (Section 6) is the one that depends on this.
    """
    return robot.battery_pct - robot.drain_rate * assigned_weight >= robot.min_battery  # TODO: replace z3.BoolVal(True) with the real inequality (Task 4)
    # negating the drain cost # since the drain rate is per one kg, for total weight



def verify_dispatch_smt(raw_payload: dict) -> Dict[str, Any]:
    """
    Evaluates payload against First-Order Logic constraints using Z3.
    Returns structured mathematical feedback for the agent context.
    """
    try:
        payload = VerificationPayload.model_validate(raw_payload)
    except Exception as e:
        return {"status": "SYNTACTIC_ERROR", "message": str(e)}

    solver = z3.Solver()
    num_robots = len(payload.robots)
    pkg_vars = {book.id: z3.Int(f"assign_{book.id}") for book in payload.books}

    # 1. Bounds: Every book mapped to a valid robot index
    for book_id, var in pkg_vars.items():
        solver.add(var >= 0, var < num_robots)

    # 2. Battery Theory Check: Low battery robots cannot receive assignments
    for idx, robot in enumerate(payload.robots):
        if robot.battery_pct < robot.min_battery:
            for book_id, var in pkg_vars.items():
                solver.add(var != idx)

    # 3. Weight Theory Check: Sum of book weights <= robot capacity
    for idx, robot in enumerate(payload.robots):
        assigned_weight = z3.Sum([
            z3.If(pkg_vars[book.id] == idx, book.weight, 0)
            for book in payload.books
        ])
        solver.add(assigned_weight <= robot.capacity) # was flipped

    # 4. Battery-Drain Theory Check (Task 4 -- fill in battery_drain_constraint above)
    for idx, robot in enumerate(payload.robots):
        assigned_weight = z3.Sum([
            z3.If(pkg_vars[book.id] == idx, book.weight, 0)
            for book in payload.books
        ])
        solver.add(battery_drain_constraint(robot, assigned_weight))

    # Solve
    if solver.check() == z3.sat:
        model = solver.model()
        assignments = {
            book.id: payload.robots[model[pkg_vars[book.id]].as_long()].id
            for book in payload.books
        }
        return {
            "status": "SATISFIABLE",
            "assignments": assignments,
            "feedback": "Allocation verified mathematically.",
        }

    # ---- UNSAT: figure out WHICH of three distinct failures applies ----
    active_robots = [a for a in payload.robots if a.battery_pct >= a.min_battery]
    disabled_robots = [a for a in payload.robots if a.battery_pct < a.min_battery]
    total_weight = sum(book.weight for book in payload.books)
    total_active_capacity = sum(a.capacity for a in active_robots)
    max_active_capacity = max((a.capacity for a in active_robots), default=0)
    oversized_books = [
        book.id for book in payload.books if book.weight > max_active_capacity
    ]

    diagnostics = {
        "total_requested_weight": total_weight,
        "available_active_capacity": total_active_capacity,
        "active_robots": [a.id for a in active_robots],
        "disabled_robots_low_battery": [a.id for a in disabled_robots],
        "oversized_books": oversized_books,
    }

    if not active_robots:
        feedback = (
            "UNSAT: every robot is below its minimum battery threshold. "
            "No assignment is possible until at least one robot is charged."
        )
    elif oversized_books:
        feedback = (
            f"UNSAT: book(s) {oversized_books} individually exceed the "
            f"capacity of every active robot (largest active capacity: "
            f"{max_active_capacity}kg). This cannot be fixed by reassigning "
            f"books alone -- split the book, free up a bigger robot, or "
            f"drop it."
        )
    elif total_weight > total_active_capacity:
        feedback = (
            f"UNSAT: total requested weight ({total_weight}kg) exceeds "
            f"usable active capacity ({total_active_capacity}kg). Check "
            f"disabled low-battery robots or reduce the book set."
        )
    else:
        feedback = (
            f"UNSAT: no valid assignment exists even though total weight "
            f"({total_weight}kg) fits within total active capacity "
            f"({total_active_capacity}kg). This is a packing/arrangement "
            f"conflict, not a raw capacity shortfall -- try a different "
            f"split of which books travel together."
        )
        if any(a.drain_rate > 0 for a in active_robots):
            feedback += (
                " Note: one or more active robots have a nonzero drain_rate, "
                "meaning the weight they're given also costs them battery "
                "headroom above their own min_battery -- that interaction "
                "may be part of why no split of these books works, even "
                "though the raw capacity numbers look sufficient."
            )

    return {
        "status": "UNSATISFIABLE",
        "diagnostics": diagnostics,
        "feedback": feedback,
    }


if __name__ == "__main__":
    # Quick manual sanity check -- python verifier.py
    import json

    sat_case = {
        "books": [{"id": "Item-1", "weight": 10}, {"id": "Item-2", "weight": 15}],
        "robots": [
            {"id": "Robot-A", "capacity": 30, "battery_pct": 80, "min_battery": 25},
            {"id": "Robot-B", "capacity": 20, "battery_pct": 50, "min_battery": 25},
        ],
    }
    unsat_case = {
        "books": [
            {"id": "Heavy-Book-1", "weight": 40},
            {"id": "Heavy-Book-2", "weight": 25},
        ],
        "robots": [
            {"id": "Robot-1", "capacity": 30, "battery_pct": 90, "min_battery": 25},
            {"id": "Robot-2", "capacity": 20, "battery_pct": 10, "min_battery": 25},
        ],
    }

    print("SAT case:")
    print(json.dumps(verify_dispatch_smt(sat_case), indent=2))
    print("\nUNSAT case:")
    print(json.dumps(verify_dispatch_smt(unsat_case), indent=2))
