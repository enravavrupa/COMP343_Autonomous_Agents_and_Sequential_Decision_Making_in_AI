# Setup

## Install

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Environment convention (read before starting)

These are fixed for this lab. Do not change them unless a module explicitly
tells you to sweep over them:

- `is_slippery=True` for Modules 1, 2, and 4. Module 3 explicitly compares
  `is_slippery=False` vs `is_slippery=True` — that's the only place it varies.
- `gamma = 0.99` everywhere, including Modules 1–2 (the manual only stated
  this for Module 3 originally; it applies globally).
- Seeds: use `seed=1` through `seed=5` for every multi-seed run. Don't
  substitute your own seeds — results won't be comparable to the reference
  solutions or to your lab partner's.
- Terminal states: by convention `V(s) = 0` for both the goal and hole
  states. The reward is consumed on the *transition into* the terminal
  state, not stored at the terminal state itself. Your heatmaps in Module 4
  should show V(goal) = 0, not V(goal) ≈ 1 — this trips people up.
- `terminated` vs `truncated`: Gymnasium's `FrozenLake-v1` wraps a 100-step
  `TimeLimit`. If an episode is cut off by truncation (not a real
  terminal state), you must still bootstrap using `V(S_T)` — do NOT treat
  a truncated final state as having value 0. Collapsing
  `terminated or truncated` into one "done" flag is the most common bug in
  this lab. Keep them separate.

## Ground truth

Run `python env_utils.py` once before anything else. This solves the Bellman
equations for the uniform random policy exactly (a linear solve, not
iterative — so there's no tolerance/convergence ambiguity) and caches the
result to `data/ground_truth_values.json`. Every MSE computation in this lab
reads from that file. If your algorithm's MSE looks wrong, check this file
was actually regenerated before you assume your update rule is broken.
