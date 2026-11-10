# Treasure Hunt MDP Lab — Cheat Sheet

One page you can have open while you work. For the full tensor
convention reference, see `MDP_INPUT_OUTPUT_FORMAT.md`.

## 1. The two algorithms

Both solve the same Bellman optimality equation; they differ in HOW.

```
Q(s,a) = R(s,a) + discount * sum_s' P(a,s,s') * V(s')
```

### Policy Iteration
```
policy = arbitrary (e.g. always action 0)
repeat:
    V = exact value of the CURRENT policy        # policy evaluation
        (solve (I - discount*P_pi) V = R_pi)
    new_policy(s) = argmax_a Q(s,a) using V       # policy improvement
until new_policy == policy
```
Converges in few outer iterations (~4-6 typically here), but each
iteration does more work (a full policy evaluation).

### Value Iteration
```
V = 0
repeat:
    V_new(s) = max_a Q(s,a) using CURRENT V       # Bellman backup
until V barely changes
policy(s) = argmax_a Q(s,a) using final V         # extract ONCE, at the end
```
Converges in more sweeps (~40-50 typically here), but each sweep is
cheap — no separate evaluation loop.

**The one-sentence difference**: policy iteration fixes a policy,
solves for its exact value, THEN improves it; value iteration never
fixes a policy along the way — it improves the value estimate directly,
every single sweep, and only reads off a policy at the very end.

## 2. Code map

```
  env.py  --------->  mdp_builder.py  --------->  policy engines  --------->  game_gui.py
 (grid, moves,       (builds P, R           (pi_lab / pi_solution /       compare_engines.py
  rewards, random     tensors from an        vi_lab / vi_solution —        evaluate_policy.py
  layouts — NO        env instance)          all implement                test_correctness.py
  pygame here)                               policy_engine.PolicyEngine)  check_solutions.py
```
See the diagram in chat for the same picture. Data flows strictly
left to right — nothing on the right ever reaches back and modifies
`env.py` or `mdp_builder.py`.

| File | Role |
|---|---|
| `env.py` | The game engine. Grid, moves, stochastic slip, rewards, random solvable layouts. No pygame, no numpy tensor math. |
| `mdp_builder.py` | The ONLY place that builds `P` (A,S,S) and `R` (S,A) from an `env` instance. |
| `policy_engine.py` | The interface every solver implements: `solve(P, R, discount) -> PolicyEngineResult`. |
| `policy_iteration_lab.py` | **You edit this.** Two TODOs. |
| `value_iteration_lab.py` | **You edit this.** Two TODOs. |
| `policy_iteration_solution.py` | Complete instructor reference for PI. Don't edit — ground truth. |
| `value_iteration_solution.py` | Complete instructor reference for VI. Don't edit — ground truth. |
| `game_gui.py` | Pygame view. Calls `engine.solve(...)`, reads back `.policy`. No MDP math. |
| `main.py` | Entry point — picks which engine the GUI uses. |
| `compare_engines.py` | Compares any two engines on one random layout. |
| `evaluate_policy.py` | Generic "how good is this policy" checker for ANY policy array. |
| `test_correctness.py` / `check_solutions.py` | Automated correctness tests. |

## 3. What to actually do

- [ ] Read `env.py` and `mdp_builder.py` (don't edit either).
- [ ] Read `MDP_INPUT_OUTPUT_FORMAT.md` §1-4 for the state/action/tensor conventions.
- [ ] Open **your** lab file (`policy_iteration_lab.py` or `value_iteration_lab.py`).
- [ ] Fill in the two `TODO` methods. Docstrings tell you the exact math and give a numpy hint for each.
- [ ] Run `python compare_engines.py --engine-a <yours>_lab --engine-b <other>_solution --seed 0` — should show ~100% policy agreement and a value gap under 1e-3.
- [ ] Run `python test_correctness.py --include-lab` — should print `[INFO]` lines with high agreement, no crashes.
- [ ] Try `python main.py --engine <yours>_lab` and play in policy mode (press `P`, then `A` for auto-play) to watch your solver navigate a random layout.
- [ ] If something hangs or disagrees a lot on a specific seed, re-run `compare_engines.py --seed <that seed>` alone and inspect — see §5.

## 4. The "library" you're given (don't reimplement)

From `mdp_builder.build_mdp_tensors(env)`:
```python
P, R, states, state_to_index = build_mdp_tensors(env)
# P.shape == (4, S, S)   P[a, s, s'] = transition probability
# R.shape == (S, 4)      R[s, a]    = expected immediate reward
# states                  list of every (position, has_treasure) state
# state_to_index          dict: state -> int index into P/R/policy arrays
```
From `policy_engine.py`:
```python
class PolicyEngineResult:
    policy          # np.ndarray shape (S,) int — action per state
    value_function  # np.ndarray shape (S,) float — V(s)
    iterations      # int
    engine_name     # str
```
Your `solve(self, P, R, discount=0.95, **kwargs)` must return one of these.

## 5. Function quick-reference

| Function | File | Signature |
|---|---|---|
| `generate_random_layout` | `env.py` | `(rows=4, cols=4, num_walls=2, num_danger=1, rng=None) -> layout dict` |
| `all_states` | `env.py` | `(self) -> list of (position, has_treasure)` |
| `simulate_transition` | `env.py` | `(self, state, actual_action) -> (next_state, reward)` — pure, no side effects |
| `action_outcomes` | `env.py` | `(self, chosen_action) -> [(actual_action, prob), ...]` (0.8, 0.1, 0.1) |
| `build_mdp_tensors` | `mdp_builder.py` | `(env) -> (P, R, states, state_to_index)` |
| `solve` | every engine | `(self, P, R, discount=0.95, **kwargs) -> PolicyEngineResult` |
| `evaluate_policy` | `evaluate_policy.py` | `(env, policy, state_to_index, discount=0.95, num_episodes=1000, rng=None) -> report dict` |

## 6. Known gotchas (read before you panic)

- **V(s) is FUTURE reward only.** A state where you've already collected
  the treasure can have LOWER value than the same position before
  collecting it — the +15 is a sunk, already-counted event in one case
  and a pending event in the other. Not a bug.
- **Ties can cause infinite policy cycling** if policy evaluation is
  only approximate (iterative to a tolerance) — two near-tied actions
  can flip back and forth forever. The shipped `policy_iteration_solution.py`
  avoids this with EXACT evaluation (linear solve) and a tie-stability
  check. If your own policy-iteration implementation hangs on some seed,
  this is almost certainly why — see `MDP_INPUT_OUTPUT_FORMAT.md`.
- **Some enumerated states are unreachable in real play** (e.g. "standing
  on the treasure without holding it") but still get a well-defined
  Bellman value. Expected MDP behavior, not a bug.
