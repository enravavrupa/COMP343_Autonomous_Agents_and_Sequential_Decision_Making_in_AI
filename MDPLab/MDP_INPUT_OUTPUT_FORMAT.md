# MDP Input/Output Format — Treasure Hunt

This is the contract every file in this codebase agrees on. If you're
writing a new engine (or hooking this up to `pymdptoolbox` directly),
this is the page to check.

## 1. State space

A state is `(position, has_treasure)`:
- `position` is `(row, col)`, `0 <= row < rows`, `0 <= col < cols`, and
  never a wall cell.
- `has_treasure` is `True`/`False`.

`env.all_states()` returns every such state on the CURRENT layout, in a
fixed order. `mdp_builder.enumerate_states(env)` turns that list into
`state_to_index`, a `dict` mapping `(position, has_treasure) -> int`.
This index is what every tensor and every policy array is indexed by —
nothing downstream ever looks at `(row, col)` again.

Because layouts are randomized, **the number of states S and the
meaning of index 7 (say) are different every run.** Never hardcode a
state index. Always go through `state_to_index`.

## 2. Actions

Fixed and layout-independent — always 4:

| index | meaning |
|---|---|
| 0 | UP |
| 1 | DOWN |
| 2 | LEFT |
| 3 | RIGHT |

## 3. The transition tensor `P`

```
P.shape == (A, S, S)
P[a, s, s'] = probability of ending in state s' given you took action a in state s
```

Built by `mdp_builder.build_mdp_tensors(env)`. For each non-terminal
state, `env.action_outcomes(a)` gives the three `(actual_action,
probability)` pairs the 0.80/0.10/0.10 slip model produces; each is run
through `env.simulate_transition(state, actual_action)` to get the
resulting state, and probabilities are accumulated into `P[a, s, :]`.

**Every row `P[a, s, :]` sums to 1.** `build_mdp_tensors` checks this
before returning — if you see a `ValueError` about row sums, the bug is
in how a transition was computed, not in your policy engine.

**Terminal states are absorbing.** Once `position == env.goal`, every
action loops back to the same state with probability 1 and reward 0.
This is what lets value/policy iteration converge instead of trying to
keep extracting reward from a state that no longer exists in play.

## 4. The reward tensor `R`

```
R.shape == (S, A)
R[s, a] = expected immediate reward for taking action a in state s
```

Because the actual action is stochastic, the reward you'd get from a
given `(s, a)` isn't a single number — it's an expectation over the
three possible outcomes: `R[s, a] = sum over outcomes of prob * reward`.
This is the standard convention `pymdptoolbox` expects for its `R`
argument, so if you want to cross-check with that library:

```python
import mdptoolbox
solver = mdptoolbox.mdp.PolicyIteration(P, R, discount=0.95)
solver.run()
# solver.policy is a tuple of length S, same indexing as our policy arrays
```

## 5. Discount factor

A plain float, passed explicitly to `engine.solve(P, R, discount=...)`.
Not stored on `P` or `R`. Default used across the lab is `0.95`.

## 6. What a solver returns

Every engine (`policy_engine.PolicyEngine` subclass) returns a
`PolicyEngineResult`:

```python
result.policy            # np.ndarray, shape (S,), dtype int  — action per state
result.value_function    # np.ndarray, shape (S,), dtype float — V(s)
result.iterations        # int — outer loop count, for comparing convergence speed
result.engine_name        # str
```

`result.policy[state_to_index[state]]` gives you the action for any
state — this is exactly what `game_gui.get_policy_action()` does at
play time.

## 7. Where each piece lives

| File | Knows about grids/pygame? | Knows about P/R tensors? |
|---|---|---|
| `env.py` | Yes (it defines the grid) | No |
| `mdp_builder.py` | Reads env, but is pure math | Builds them |
| `policy_engine.py` | No | Defines the interface |
| `value_iteration_solution.py` | No | Consumes them |
| `policy_iteration_lab.py` | No | Consumes them — **you fill this in** |
| `compare_engines.py` | No (drives env only to simulate returns) | Orchestrates |
| `game_gui.py` | Yes | Never touches them directly — only calls `engine.solve(P, R, discount)` and reads back `result.policy` |

The rule of thumb: if you're tempted to import `pygame` in a file that
also imports `numpy` for `P`/`R` math, stop — that coupling is exactly
what this structure is designed to prevent.

## 8. Transition probabilities (the slip model)

When you choose an action, the environment doesn't always execute it.
It executes your chosen action **0.80** of the time, and "slips" into
one of the two actions perpendicular to it, **0.10** each. UP/DOWN can
never slip into each other, and LEFT/RIGHT can never slip into each
other — only into the perpendicular pair:

| you chose | 0.80 (intended) | 0.10 (slip) | 0.10 (slip) |
|---|---|---|---|
| UP (0)    | UP    | LEFT  | RIGHT |
| DOWN (1)  | DOWN  | LEFT  | RIGHT |
| LEFT (2)  | LEFT  | UP    | DOWN  |
| RIGHT (3) | RIGHT | UP    | DOWN  |

This is `PERPENDICULAR_ACTIONS` in `env.py`, consumed by
`env.action_outcomes(chosen_action)`. If a slipped-into move would hit
a wall or the grid edge, the player simply stays in place for that
outcome (see `deterministic_move`) — the slip still "happens", it just
has no effect that step.

## 9. A note on unreachable states

`env.all_states()` enumerates every `(position, has_treasure)`
combination on the grid, including ones that can never actually occur
during real play — e.g. `(treasure_position, False)` after the
treasure has already been collected in a real episode never recurs,
but the *combination* still exists in the state space and gets a
well-defined value under the Bellman equations. This is normal MDP
behaviour (transient/unreachable states are still valid to solve for),
but it can produce surprising numbers if you don't expect it — e.g. a
1-row corridor where UP/DOWN bounce back in place can trigger the
"you've arrived at the treasure" check even from a state that
shouldn't logically be "at the treasure without holding it" yet. If a
value looks wrong, check first whether the state is actually reachable
from the start state before assuming it's a bug.

Related, and worth internalizing before you start debugging: **V(s) is
expected reward from s *onward*, not total accumulated reward.** A
state where the treasure is already collected can have a LOWER value
than the otherwise-identical state where it isn't collected yet,
because the +15 collection bonus is a future event in one case and a
sunk, already-counted event in the other. `test_correctness.py` has a
worked example of exactly this.

## 10. Suggested lab timing (100 minutes)

Matches the four-phase lab structure used elsewhere in this course:

- **0:00–0:20 — Opening.** Walk through this document and `env.py` /
  `mdp_builder.py` together. Run `compare_engines.py` once as-is (it
  will fail on `NotImplementedError` from `policy_iteration_lab.py`) so
  students see the target error and the harness that will grade them.
- **0:20–1:10 — Single-question experiment.** The question: *"Does your
  policy iteration converge to the same policy as value iteration, on a
  layout you've never seen?"* Students implement `_policy_evaluation`
  and `_policy_improvement` in `policy_iteration_lab.py`. Checkpoint at
  the 30-minute mark: `_policy_evaluation` alone should produce a value
  function close to zero-initialized-then-converged for the fixed
  all-UP policy — a good sanity print before improvement is attempted.
- **1:10–1:20 — Break-it.** Run `compare_engines.py --seed <various>`
  across several random layouts. Find a layout where policy agreement
  is well below 100% — is it a tie between equally good actions (fine),
  or a real bug (check the value gap)?
- **1:20–1:30 — Exit question.** Why does value iteration take roughly
  10x more outer-loop iterations than policy iteration to converge on
  these layouts, and what's the cost tradeoff being made in each inner
  policy-evaluation loop?

## 11. Testing tools in this codebase

| File | What it checks |
|---|---|
| `test_correctness.py` | Run this first. Automated: tensor invariants, a hand-verifiable small case, cross-validation between the two reference solvers across many random layouts. Pass `--include-lab` once your own files are filled in. |
| `compare_engines.py --engine-a X --engine-b Y` | Pick any two of `pi_lab`, `pi_solution`, `vi_lab`, `vi_solution` and compare them on one random layout: iterations, policy agreement, value gap, simulated return. |
| `evaluate_policy.py` | Given ANY policy (not just from these four engines — a hand-written heuristic works too), reports its exact discounted value, a matching discounted Monte-Carlo simulation as a cross-check, and its regret against the optimal policy. |

There are two full engine implementations in this codebase
(`pi_solution`, `vi_solution`) and two skeletons you fill in
(`pi_lab`, `vi_lab`). Checking your skeleton against the *other
algorithm's* solution file (e.g. `pi_lab` vs `vi_solution`) is a
stronger test than checking it against its own algorithm's solution
file, since a shared bug in "your policy iteration" and "the reference
policy iteration" wouldn't necessarily show up — two independent
algorithms agreeing is the more convincing signal.

## 12. A note on policy-iteration cycling

If you implement policy evaluation *iteratively* (sweeping to a
tolerance, as `policy_iteration_lab.py` asks you to) rather than
*exactly* (solving the linear system directly), you can hit a real
failure mode: when two actions in some state are very close in true
value, the small residual error left by an approximate evaluation can
flip which one looks better between successive outer iterations. The
result is a state whose chosen action oscillates forever between two
candidates, so `new_policy == policy` never becomes true and the
algorithm never terminates.

This isn't hypothetical — it happened on a real randomly generated
layout while building this lab (`check_solutions.py`, seed 15: state 3
cycled between actions 1 and 3 indefinitely). `policy_iteration_solution.py`
avoids it two ways: **exact** evaluation via `np.linalg.solve(I -
discount*P_pi, R_pi)` instead of iterating to a tolerance, and a
**tie-stable** improvement step that only switches a state's action if
the alternative is better by more than a small `tie_epsilon` — not
just numerically better by noise.

If your own implementation hangs on some seed, this is almost
certainly why. Two ways to fix it, either is fine for the lab:
1. Switch `_policy_evaluation` to the exact linear-solve form above.
2. Keep the iterative version, but make `_policy_improvement`
   tie-stable (only switch away from the current action if the
   improvement exceeds a small epsilon).
