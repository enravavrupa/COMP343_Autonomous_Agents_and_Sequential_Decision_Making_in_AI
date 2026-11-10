"""
test_correctness.py
=====================
Run this against YOUR OWN sarsa_lab.py / qlearning_lab.py before doing
anything else. It checks your update rule against a hand-computed example
on a single, fixed transition -- if this fails, nothing downstream in the
lab will make sense.

Usage:
    python3 test_correctness.py
"""

import numpy as np
from packbot_env import state_index, N_STATES, N_ACTIONS

ALPHA = 0.1
GAMMA = 0.9


def make_fixed_Q():
    """A hand-chosen (state,action) -> value table. Deterministic, no env needed."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[state_index(0, 5), 0] = 2.0   # Q(wear=5, FAST)    = 2.0
    Q[state_index(0, 5), 1] = 0.5   # Q(wear=5, CAREFUL)  = 0.5
    Q[state_index(0, 7), 0] = -1.0  # Q(wear=7, FAST)    = -1.0
    Q[state_index(0, 7), 1] = 0.8   # Q(wear=7, CAREFUL)  = 0.8
    return Q


def test_sarsa_update():
    from sarsa_lab import sarsa_update  # <-- the function you must implement

    Q = make_fixed_Q()
    s = state_index(0, 5)
    a = 0  # FAST
    r = 3.0
    s_next = state_index(0, 7)
    a_next = 1  # CAREFUL (this is the SAMPLED next action -- SARSA uses it directly)

    new_value = sarsa_update(Q, s, a, r, s_next, a_next, alpha=ALPHA, gamma=GAMMA)

    # By hand: target = r + gamma * Q[s_next, a_next] = 3.0 + 0.9*0.8 = 3.72
    #          new_value = Q[s,a] + alpha*(target - Q[s,a]) = 2.0 + 0.1*(3.72-2.0) = 2.172
    expected = 2.172
    assert abs(new_value - expected) < 1e-6, (
        f"SARSA update wrong. Got {new_value}, expected {expected}.\n"
        f"Remember: SARSA bootstraps off Q[s_next, a_next] using the ACTION "
        f"THAT WAS ACTUALLY SAMPLED (a_next) -- not the max over actions."
    )
    print("sarsa_update: PASS")


def test_qlearning_update():
    from qlearning_lab import qlearning_update  # <-- the function you must implement

    Q = make_fixed_Q()
    s = state_index(0, 5)
    a = 0  # FAST
    r = 3.0
    s_next = state_index(0, 7)
    # NOTE: no a_next argument -- Q-learning does not use the sampled next
    # action in its target at all.

    new_value = qlearning_update(Q, s, a, r, s_next, alpha=ALPHA, gamma=GAMMA)

    # By hand: target = r + gamma * max_a' Q[s_next, a'] = 3.0 + 0.9*max(-1.0, 0.8)
    #                  = 3.0 + 0.9*0.8 = 3.72
    #          new_value = 2.0 + 0.1*(3.72-2.0) = 2.172
    expected = 2.172
    assert abs(new_value - expected) < 1e-6, (
        f"Q-learning update wrong. Got {new_value}, expected {expected}.\n"
        f"Remember: Q-learning bootstraps off max_a' Q[s_next, a'] -- the "
        f"BEST available action at s_next, regardless of what your "
        f"behavior policy would actually pick there."
    )
    print("qlearning_update: PASS")


def test_terminal_case():
    """A second check: when done=True, neither algorithm should bootstrap at all."""
    from sarsa_lab import sarsa_update
    from qlearning_lab import qlearning_update

    Q = make_fixed_Q()
    s = state_index(0, 9)
    a = 0
    r = -30.0  # e.g. a breakdown

    # Convention used throughout this lab: on a terminal transition, call
    # the update function with s_next=None (and a_next=None for SARSA).
    # Your implementation must special-case this and NOT index into Q.
    new_value_sarsa = sarsa_update(Q, s, a, r, None, None, alpha=ALPHA, gamma=GAMMA)
    new_value_q = qlearning_update(Q, s, a, r, None, alpha=ALPHA, gamma=GAMMA)

    expected = Q[s, a] + ALPHA * (r - Q[s, a])  # target == r, no bootstrap
    assert abs(new_value_sarsa - expected) < 1e-6, "SARSA terminal-state handling is wrong."
    assert abs(new_value_q - expected) < 1e-6, "Q-learning terminal-state handling is wrong."
    print("terminal-state handling: PASS")


if __name__ == "__main__":
    test_sarsa_update()
    test_qlearning_update()
    test_terminal_case()
    print("\nAll correctness checks passed.")
