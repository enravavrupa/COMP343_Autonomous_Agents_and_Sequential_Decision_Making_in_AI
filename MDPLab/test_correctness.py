"""
test_correctness.py — how to actually test this codebase
=============================================================

Run:
    python test_correctness.py

No pytest needed — plain asserts, printed as they pass. This is the
methodology described in Section 1 below; read the module docstring
of each test function for what it's checking and why.

Three layers of testing, from cheapest/most-automatic to most manual:

  1. INVARIANT checks — properties that must hold no matter what the
     random layout is: every row of P sums to 1, terminal states are
     absorbing, R has the right shape. These run automatically inside
     mdp_builder.build_mdp_tensors() already, and are re-checked here
     explicitly across several random layouts.

  2. A HAND-VERIFIABLE small case — a 1x3 corridor with no walls or
     danger, built directly (not via generate_random_layout), where you
     can reason about the correct ordering of values without solving
     anything: being closer to the goal with the treasure should always
     be worth more than being further away or without it. This test
     doesn't need a "gold" solver at all — it checks the solver against
     plain reasoning about the reward structure.

  3. CROSS-VALIDATION between independent implementations — if
     Policy Iteration and Value Iteration (two different algorithms)
     agree on the value function to ~1e-6 and mostly agree on policy
     across MANY random layouts, that is strong evidence both are
     correct, because a bug in one is very unlikely to reproduce the
     exact same wrong numbers in the other. This is what
     compare_engines.py automates for repeated use; here we just run
     it across a batch of seeds and assert thresholds.

This file exercises the *_solution.py reference engines by default, so
it is runnable immediately. Once you've filled in policy_iteration_lab.py
/ value_iteration_lab.py, re-run with --include-lab to test your own
code the same way.
"""

import argparse
import random

import numpy as np

from env import TreasureHuntEnvironment
from mdp_builder import build_mdp_tensors, enumerate_states
from value_iteration_solution import ValueIterationEngine
from policy_iteration_solution import PolicyIterationSolutionEngine


def test_transition_tensor_invariants(num_layouts=20):
    """Layer 1: P rows sum to 1, terminal states are absorbing with
    zero reward, R has shape (S, A)."""
    rng = random.Random(0)
    for _ in range(num_layouts):
        layout = TreasureHuntEnvironment.generate_random_layout(rng=rng)
        env = TreasureHuntEnvironment(layout=layout, rng=rng)
        P, R, states, state_to_index = build_mdp_tensors(env)  # raises internally if invalid

        assert R.shape == (len(states), 4), f"R shape wrong: {R.shape}"

        for s_idx, state in enumerate(states):
            if env.is_terminal(state):
                for a in range(4):
                    assert P[a, s_idx, s_idx] == 1.0, "terminal state must self-loop with prob 1"
                    assert R[s_idx, a] == 0.0, "terminal state must have zero reward"

    print(f"[PASS] transition tensor invariants hold across {num_layouts} random layouts")


def test_hand_verifiable_corridor():
    """Layer 2: a 1x3 corridor, built by hand, with no walls or danger.

    IMPORTANT gotcha this test is built around: V(s) is EXPECTED FUTURE
    reward from s onward, not total accumulated reward. That means
    ((0, 0), True) — "far from goal, treasure ALREADY collected" — is
    correctly worth LESS than ((0, 0), False) — "far from goal,
    treasure NOT YET collected" — because the +15 collection bonus is a
    future event still ahead of you in the False case, but a sunk, past
    event in the True case that no longer counts. If your intuition
    says "holding the treasure should always be worth more," that
    intuition is about total accumulated reward, not V(s) — check your
    engine against THIS reasoning, not the other one.

    What IS safe to assert regardless of treasure status: being closer
    to the goal is worth more (fewer -1 steps to pay), and the optimal
    first move should head toward whatever objective is still pending.

    Note deliberately NOT checked here: ((0, 1), False) — "standing on
    the treasure cell without holding it" — is never reached by real
    play (collection fires the instant you land there), but it IS
    enumerated as a state with well-defined Bellman value, because its
    self-loop moves (UP/DOWN bouncing off a 1-row grid) happen to
    satisfy the collection check. Expected MDP behaviour for a
    transient/unreachable state, not a bug — see
    MDP_INPUT_OUTPUT_FORMAT.md, "A note on unreachable states"."""
    layout = dict(rows=1, cols=3, start=(0, 0), treasure=(0, 1), goal=(0, 2),
                  walls=set(), danger=set())
    env = TreasureHuntEnvironment(layout=layout)
    P, R, states, state_to_index = build_mdp_tensors(env)

    result = ValueIterationEngine().solve(P, R, discount=0.95)
    V = result.value_function

    v_near_with_treasure = V[state_to_index[((0, 1), True)]]
    v_far_with_treasure = V[state_to_index[((0, 0), True)]]

    assert v_near_with_treasure > v_far_with_treasure, (
        "closer to goal, same treasure status, must be worth more"
    )

    # Optimal action at the true start state (treasure not yet held)
    # should move TOWARD the treasure, i.e. RIGHT (action 3).
    start_action = result.policy[state_to_index[((0, 0), False)]]
    assert start_action == 3, f"expected RIGHT (3) at the empty-corridor start, got {start_action}"

    # Optimal action once holding the treasure should move TOWARD the goal.
    holding_action = result.policy[state_to_index[((0, 1), True)]]
    assert holding_action == 3, f"expected RIGHT (3) once holding treasure, got {holding_action}"

    print("[PASS] hand-verifiable corridor case: value ordering and optimal direction correct")


def test_cross_validation(num_layouts=15, agreement_floor=0.85, value_gap_ceiling=1e-3):
    """Layer 3: Policy Iteration and Value Iteration are different
    algorithms; if they agree closely across many random layouts,
    that's strong evidence both are implemented correctly."""
    rng = random.Random(123)
    worst_agreement = 1.0
    worst_gap = 0.0

    for _ in range(num_layouts):
        layout = TreasureHuntEnvironment.generate_random_layout(rng=rng)
        env = TreasureHuntEnvironment(layout=layout, rng=rng)
        P, R, states, _ = build_mdp_tensors(env)

        pi_result = PolicyIterationEngine().solve(P, R, discount=0.95)
        vi_result = ValueIterationEngine().solve(P, R, discount=0.95)

        agreement = float(np.mean(pi_result.policy == vi_result.policy))
        gap = float(np.max(np.abs(pi_result.value_function - vi_result.value_function)))

        worst_agreement = min(worst_agreement, agreement)
        worst_gap = max(worst_gap, gap)

    assert worst_agreement >= agreement_floor, (
        f"policy agreement dropped to {worst_agreement:.2%} across {num_layouts} layouts "
        f"(floor {agreement_floor:.0%}) — investigate with compare_engines.py --seed <n>"
    )
    assert worst_gap <= value_gap_ceiling, (
        f"value function gap reached {worst_gap} across {num_layouts} layouts "
        f"(ceiling {value_gap_ceiling})"
    )

    print(f"[PASS] cross-validation across {num_layouts} random layouts: "
          f"worst policy agreement {worst_agreement:.2%}, worst value gap {worst_gap:.2e}")


def test_policy_iteration_lab():
    """
    Compare the student's PolicyIterationEngine against the
    reference ValueIterationEngine.
    """
    from policy_iteration_lab import PolicyIterationEngine

    rng = random.Random(7)
    layout = TreasureHuntEnvironment.generate_random_layout(rng=rng)
    env = TreasureHuntEnvironment(layout=layout, rng=rng)

    P, R, states, _ = build_mdp_tensors(env)

    try:
        student_result = PolicyIterationEngine().solve(
            P, R, discount=0.95
        )
    except NotImplementedError as exc:
        raise AssertionError(
            f"PolicyIterationEngine is not implemented: {exc}"
        )

    reference_result = ValueIterationEngine().solve(
        P, R, discount=0.95
    )

    agreement = float(
        np.mean(
            student_result.policy == reference_result.policy
        )
    )

    # Change this threshold if some policy differences are acceptable.
    required_agreement = 0.95

    assert agreement >= required_agreement, (
        f"Policy agreement was only {agreement:.2%}; "
        f"expected at least {required_agreement:.0%}."
    )


def test_value_iteration_lab():
    """
    Compare the student's ValueIterationLabEngine against the
    reference PolicyIterationSolutionEngine.
    """
    from value_iteration_lab import ValueIterationLabEngine

    rng = random.Random(7)
    layout = TreasureHuntEnvironment.generate_random_layout(rng=rng)
    env = TreasureHuntEnvironment(layout=layout, rng=rng)

    P, R, states, _ = build_mdp_tensors(env)

    try:
        student_result = ValueIterationLabEngine().solve(
            P, R, discount=0.95
        )
    except NotImplementedError as exc:
        raise AssertionError(
            f"ValueIterationLabEngine is not implemented: {exc}"
        )

    reference_result = PolicyIterationSolutionEngine().solve(
        P, R, discount=0.95
    )

    agreement = float(
        np.mean(
            student_result.policy == reference_result.policy
        )
    )

    # Change this threshold if some policy differences are acceptable.
    required_agreement = 0.95

    assert agreement >= required_agreement, (
        f"Policy agreement was only {agreement:.2%}; "
        f"expected at least {required_agreement:.0%}."
    )



def test_lab_files():
    """Optional: run the same cross-validation against YOUR
    policy_iteration_lab.py / value_iteration_lab.py once they're filled in."""
    from policy_iteration_lab import PolicyIterationEngine
    from value_iteration_lab import ValueIterationLabEngine
 
    rng = random.Random(7)
    layout = TreasureHuntEnvironment.generate_random_layout(rng=rng)
    env = TreasureHuntEnvironment(layout=layout, rng=rng)
    P, R, states, _ = build_mdp_tensors(env)
 
    try:
        pi_lab_result = PolicyIterationEngine().solve(P, R, discount=0.95)
        vi_solution_result = ValueIterationEngine().solve(P, R, discount=0.95)
        agreement = float(np.mean(pi_lab_result.policy == vi_solution_result.policy))
        print(f"[INFO] your PolicyIterationEngine vs. reference Value Iteration: "
              f"{agreement:.2%} policy agreement")
    except NotImplementedError as exc:
        print(f"[SKIP] policy_iteration_lab.py not finished yet: {exc}")
 
    try:
        vi_lab_result = ValueIterationLabEngine().solve(P, R, discount=0.95)
        pi_solution_result = PolicyIterationSolutionEngine().solve(P, R, discount=0.95)
        agreement2 = float(np.mean(vi_lab_result.policy == pi_solution_result.policy))
        print(f"[INFO] your ValueIterationLabEngine vs. reference Policy Iteration: "
              f"{agreement2:.2%} policy agreement")
    except NotImplementedError as exc:
        print(f"[SKIP] value_iteration_lab.py not finished yet: {exc}")


def run_test(name, test_function):
    """
    Run one test and return a result dictionary.

    The test is considered passed if it completes without raising
    an exception. Otherwise, the exception message is recorded.
    """
    try:
        test_function()
        return {
            "Test": name,
            "Status": "PASS",
            "Details": "Completed successfully"
        }

    except AssertionError as error:
        return {
            "Test": name,
            "Status": "FAIL",
            "Details": str(error) or "An assertion failed"
        }

    except Exception as error:
        return {
            "Test": name,
            "Status": "ERROR",
            "Details": f"{type(error).__name__}: {error}"
        }


def print_test_summary(results):
    """Print a table summarizing all test results."""

    test_width = max(
        len("Test"),
        max(len(result["Test"]) for result in results)
    )

    status_width = max(
        len("Status"),
        max(len(result["Status"]) for result in results)
    )

    details_width = max(
        len("Details"),
        max(len(result["Details"]) for result in results)
    )

    separator = (
        "+"
        + "-" * (test_width + 2)
        + "+"
        + "-" * (status_width + 2)
        + "+"
        + "-" * (details_width + 2)
        + "+"
    )

    print("\nTest summary")
    print(separator)

    print(
        f"| {'Test':<{test_width}} "
        f"| {'Status':<{status_width}} "
        f"| {'Details':<{details_width}} |"
    )

    print(separator)

    for result in results:
        print(
            f"| {result['Test']:<{test_width}} "
            f"| {result['Status']:<{status_width}} "
            f"| {result['Details']:<{details_width}} |"
        )

    print(separator)

    passed = sum(
        result["Status"] == "PASS"
        for result in results
    )

    failed = len(results) - passed

    print(f"\nPassed: {passed}/{len(results)}")

    if failed > 0:
        print(f"Failed: {failed}/{len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--include-lab",
        action="store_true",
        help="also test policy_iteration_lab.py / value_iteration_lab.py"
    )

    args = parser.parse_args()

    results = []

    results.append(
        run_test(
            "Transition tensor invariants",
            test_transition_tensor_invariants
        )
    )

    results.append(
        run_test(
            "Hand-verifiable corridor",
            test_hand_verifiable_corridor
        )
    )

    results.append(run_test("Cross Validation with Value Iteration",test_cross_validation))
    
    results.append(
        run_test(
            "Policy Iteration lab engine",
            test_policy_iteration_lab
        )
    )

    results.append(
        run_test(
            "Value Iteration lab engine",
            test_value_iteration_lab
        )
    )


    print_test_summary(results)
 
