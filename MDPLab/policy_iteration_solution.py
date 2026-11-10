"""
policy_iteration_solution.py — reference solver for Policy Iteration
=======================================================================

This is the completed version of policy_iteration_lab.py. Treat it as
ground truth for grading / as an answer key.

Two deliberate differences from the most "obvious" implementation,
both discovered by running check_solutions.py against it:

1. POLICY EVALUATION IS EXACT, not iterative. Solving
   V_pi = (I - discount * P_pi)^-1 R_pi directly (a linear system)
   instead of sweeping to a tolerance removes a whole class of
   floating-point noise that an iterative evaluation introduces.

2. POLICY IMPROVEMENT IS TIE-STABLE: a state only switches to a new
   action if that action is MEANINGFULLY better (by more than
   `tie_epsilon`), not just numerically better by some tiny amount.

Why both matter: with iterative (approximate) evaluation and a naive
"switch to whatever's best" improvement step, two actions that are
genuinely very close in true value can cause a state's chosen action to
flip back and forth between them forever — the approximation noise in V
tips the comparison a different way each outer iteration, so
`new_policy == policy` never becomes true and the algorithm never
terminates. This isn't a hypothetical: it happened on a real randomly
generated layout during testing (see check_solutions.py's docstring and
MDP_INPUT_OUTPUT_FORMAT.md, "A note on policy-iteration cycling"). Exact
evaluation removes the noise that causes the flip-flopping in the first
place; tie-stability is a second line of defense in case two actions are
ever exactly, analytically tied.
"""

import numpy as np

from policy_engine import PolicyEngine, PolicyEngineResult


class PolicyIterationSolutionEngine(PolicyEngine):

    name = "Policy Iteration (reference)"

    def __init__(self, tie_epsilon=1e-9, max_outer_iterations=200):
        self.tie_epsilon = tie_epsilon
        self.max_outer_iterations = max_outer_iterations

    def solve(self, P, R, discount=0.95, **kwargs):
        A, S, _ = P.shape
        policy = np.zeros(S, dtype=int)
        value = np.zeros(S)

        for iteration in range(1, self.max_outer_iterations + 1):
            value = self._policy_evaluation(policy, P, R, discount)
            new_policy = self._policy_improvement(policy, value, P, R, discount)

            if np.array_equal(new_policy, policy):
                return PolicyEngineResult(policy, value, iteration, self.name)

            policy = new_policy

        # Safety net, not expected to trigger with exact evaluation +
        # tie-stable improvement: return the last policy found rather
        # than hang forever, so a live lab session never freezes.
        return PolicyEngineResult(policy, value, self.max_outer_iterations,
                                   self.name + " [hit max_outer_iterations]")

    def _policy_evaluation(self, policy, P, R, discount):
        """Exact policy evaluation: solve (I - discount * P_pi) V = R_pi
        directly instead of iterating to a tolerance."""
        S = P.shape[1]
        P_pi = P[policy, np.arange(S), :]
        R_pi = R[np.arange(S), policy]
        return np.linalg.solve(np.eye(S) - discount * P_pi, R_pi)

    def _policy_improvement(self, policy, value, P, R, discount):
        """Tie-stable improvement: only switch a state's action away from
        its current choice if the best alternative is better by more than
        tie_epsilon — prevents infinite flip-flopping between two actions
        of near-identical true value."""
        A, S, _ = P.shape
        Q = np.stack([R[:, a] + discount * P[a].dot(value) for a in range(A)], axis=1)

        best_action = Q.argmax(axis=1)
        best_value = Q.max(axis=1)
        current_value = Q[np.arange(S), policy]

        meaningfully_better = (best_value - current_value) > self.tie_epsilon
        return np.where(meaningfully_better, best_action, policy)
