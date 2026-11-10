# Model-Free RL Prediction Lab

**Environment:** `FrozenLake-v1` (4x4), target policy = uniform random
**Session length:** 110 minutes (see `docs/LAB_STRUCTURE.md`)
**Modules:** 5, weighted 20% each (see `docs/MODULE_SPEC.md` in your course materials)

## What you're building

Three prediction algorithms for estimating $V^\pi(s)$ under a fixed uniform-random
policy: First-Visit Monte Carlo, TD(0), and n-step TD. You'll then run controlled
experiments comparing their sample efficiency, tune their hyperparameters, and
connect what you observe back to theory.

## Repo layout

```
algorithms/        Your implementations (mc_lab.py, td0_lab.py, nstep_td_lab.py)
                    + instructor reference solutions (*_solution.py)
                    + shared step-size schedules (lr_schedules.py)
experiments/        Module 2 (budget discovery) and Module 3 (hyperparameter sweeps)
diagnostics/        Module 4 (multi-seed curves, value heatmaps)
tests/              Correctness suite — run this before trusting any experiment result
data/               Cached exact ground-truth V^pi(s), generated once
results/            Your output lands here (plots, filled tables) — not tracked in git
docs/               Corrected lab manual, timing, reflection questions
```

## Quick start

```bash
pip install -r requirements.txt
python env_utils.py                    # generates data/ground_truth_values.json
python tests/test_correctness.py       # sanity-check before you touch experiments/
```

Then fill in the TODOs in `algorithms/mc_lab.py`, `td0_lab.py`, `nstep_td_lab.py`
before running anything in `experiments/`.

## Submission requirements

- All experiment runs logged to Weights & Biases — see `wandb_config.py` for the
  required run-naming convention and metrics. **This is a hard requirement,
  not optional.**
- Filled Module 2 budget table + Module 3 sweep plots in `results/`
- `docs/REFLECTION_QUESTIONS.md` answered and submitted alongside your W&B run link
