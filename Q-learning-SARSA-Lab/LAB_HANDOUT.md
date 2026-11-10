# Lab: SARSA vs. Q-Learning on PackBot (Take-Home)

**Estimated time:** 3-4 hours.
**Submission:** your completed `sarsa_lab.py` and `qlearning_lab.py`, the two
plots produced by `compare_policies.py`, your predict-then-verify writeup
(below), and a W&B run link for both training runs (same requirement as
the previous lab this term).

## The setup

You're controlling **PackBot**, a robotic gripper at a warehouse pick
station. For each item in a shift's queue, PackBot chooses:

- **FastGrasp** — quick, good reward, but a real (and, at high wear,
  *severe*) chance of a mishap.
- **CarefulGrasp** — slow, small reward, always safe, and relieves wear.

The gripper accumulates **wear** the more it FastGrasps. At maximum wear,
FastGrasp risks an immediate breakdown that ends the shift on the spot
with a heavy penalty — CarefulGrasp is always safe from this. State is
`(item_type, wear)`; there is no position, map, or route here — this is a
pure sequential risk-management problem, not a pathfinding task.

Read `packbot_env.py` before doing anything else. It's fully implemented
and you shouldn't need to modify it — but you do need to understand the
breakdown mechanic to reason about the results later.

## Milestone 1 — Correctness

Implement `sarsa_update()` in `sarsa_lab.py` and `qlearning_update()` in
`qlearning_lab.py`. Then run:

```
python3 test_correctness.py
```

Both functions are checked against a hand-computed example on a single
fixed transition. **Do not move on until both pass.** If you're stuck,
re-read the docstring in each function carefully — the entire difference
between the two update rules is a single term, and the test error message
tells you which one.

## Milestone 2 — Full training loops

Implement the rest of `train_sarsa()` and `train_qlearning()` (structure
is sketched in each file's docstring). When done, each file should run
standalone:

```
python3 sarsa_lab.py
python3 qlearning_lab.py
```

and print a final mean return and a set of learned wear-thresholds. Don't
worry if the printed thresholds look noisy or inconsistent between runs —
that's expected, and it's part of why Milestone 3 uses a more robust
metric than "the threshold."

## Milestone 3 — Comparison

Run:

```
python3 compare_policies.py
```

This trains both algorithms across 10 seeds and produces two plots:

- `learning_curves.png` — sanity check that both algorithms are learning.
- `fast_advantage.png` — **the main result of this lab.** For each wear
  level, it plots `Q(wear, FastGrasp) - Q(wear, CarefulGrasp)`, averaged
  over seeds. A strongly negative value near max wear means the algorithm
  has learned to treat FastGrasp there as clearly worse — i.e., it's
  built in a safety margin. A value near zero means the algorithm is
  staying indifferent right up to the edge.

If your implementations are correct, you should see the two curves
diverge as wear approaches the maximum, with SARSA dropping lower (more
conservative) than Q-learning before the point where both correctly
recognize FastGrasp-at-max-wear as catastrophic.

## Milestone 4 — Predict, then verify (written, required)

**Before running anything further**, write down your prediction:

> At `wear = 8` on a **Fragile** item (`item_type = 1`), which action do
> you expect SARSA's learned policy to prefer? Which does Q-learning's?
> Justify your answer from the bootstrap-target difference between the
> two algorithms (i.e., in terms of *what each algorithm assumes about
> the agent's own future behavior* when computing its update target) —
> not just "SARSA is more cautious."

Then check your prediction against your own trained agents:

```python
from compare_policies import query_policy_action
# Q_sarsa, Q_qlearning are Q tables you already have from train_sarsa() / train_qlearning()
query_policy_action(Q_sarsa, item_type=1, wear=8)
query_policy_action(Q_qlearning, item_type=1, wear=8)
```

Submit a short writeup (a paragraph or two) stating your prediction,
what you actually observed, and — if they didn't match — why. ("They
didn't match and I don't know why" is an acceptable answer only if you've
first checked that both your implementations pass `test_correctness.py`
and you're comparing against a policy trained for the full 4000 episodes.)

## Optional (no extra credit, just for the curious)

Your instructor may distribute a `mystery_plot.png` — the same
FastGrasp-advantage curve as above, but from an unlabeled agent. See if
you can tell from the shape of the curve alone whether it was trained
with SARSA or Q-learning, and write one sentence on what feature of the
curve gave it away.

## Optional extension: `experiments.py`

If you want to go further than the required milestones, `experiments.py`
runs two sweeps on top of your own `sarsa_lab.py` / `qlearning_lab.py`:

```
python3 experiments.py --sweep alpha       # vary the learning rate
python3 experiments.py --sweep episodes    # vary the training budget (# episodes)
python3 experiments.py --sweep both        # both (takes ~1-2 minutes)
```

`sweep_alpha.png` shows whether the SARSA/Q-learning divergence survives
across learning rates (too small and it may not show up within your
episode budget; too large and both curves get noisy). `sweep_episodes.png`
shows how much experience each algorithm needs before the gap is visible
above seed-to-seed noise — this is really a question about how many
`(item_type, wear)` transitions get visited, not about wall-clock time.
Neither sweep is required, but if you're curious why your own single-seed
run in Milestone 2 looked noisier or cleaner than a classmate's, this is
usually why.

## Grading

- Correctness test passes for both algorithms (required to receive credit
  for anything else in this lab)
- Both training loops run and produce sensible learning curves
- `fast_advantage.png` shows the expected qualitative divergence
- Predict-then-verify writeup: prediction, observed result, and (if
  applicable) a substantive explanation of any mismatch
- W&B run logged for both algorithms
