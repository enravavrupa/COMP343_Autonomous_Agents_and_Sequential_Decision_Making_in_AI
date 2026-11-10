"""
experiments.py
================
OPTIONAL EXTENSION — not required for the core lab, but a natural next
step if you want to dig further into how SARSA and Q-learning behave
under different settings.

Two sweeps are provided:

  1. sweep_learning_rate() -- how does alpha affect (a) final performance
     and (b) how cleanly the SARSA/Q-learning divergence in
     fast_advantage.png shows up? Too-small alpha learns slowly (you may
     not see the divergence within your episode budget); too-large alpha
     is noisy and can wash the divergence back out.

  2. sweep_training_budget() -- how many episodes does each algorithm
     need before the divergence in Milestone 3 is even visible above
     seed noise? This is the "number of learning tasks" question --
     more episodes = more (item, wear) transitions experienced, which is
     the thing that actually drives convergence, not wall-clock time.

Both sweeps train across multiple seeds at each setting (fewer seeds per
setting than compare_policies.py, since you're now training many more
configurations) and plot the results.

Usage:
    python3 experiments.py --sweep alpha
    python3 experiments.py --sweep episodes
    python3 experiments.py --sweep both
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sarsa_lab import train_sarsa
from qlearning_lab import train_qlearning
from rl_utils import fast_advantage_curve

SEEDS_PER_SETTING = 6
EDGE_WEAR = 9  # the wear level used as the headline "how conservative is this policy" readout


def edge_advantage(train_fn, num_episodes, alpha, seeds):
    vals = []
    for seed in seeds:
        Q, _, _ = train_fn(num_episodes=num_episodes, alpha=alpha, seed=seed)
        vals.append(fast_advantage_curve(Q, item_type=0)[EDGE_WEAR])
    return np.mean(vals), np.std(vals)


def sweep_learning_rate(alphas=(0.02, 0.05, 0.1, 0.2, 0.4, 0.6), num_episodes=4000):
    seeds = range(SEEDS_PER_SETTING)
    sarsa_mean, sarsa_std, q_mean, q_std = [], [], [], []

    for alpha in alphas:
        print(f"  alpha={alpha} ...")
        m, s = edge_advantage(train_sarsa, num_episodes, alpha, seeds)
        sarsa_mean.append(m); sarsa_std.append(s)
        m, s = edge_advantage(train_qlearning, num_episodes, alpha, seeds)
        q_mean.append(m); q_std.append(s)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(alphas, sarsa_mean, yerr=sarsa_std, marker="o", label="SARSA", capsize=3)
    ax.errorbar(alphas, q_mean, yerr=q_std, marker="s", label="Q-learning", capsize=3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Learning rate (alpha)")
    ax.set_ylabel(f"Q(wear={EDGE_WEAR}, FastGrasp) - Q(wear={EDGE_WEAR}, CarefulGrasp)")
    ax.set_title(f"Effect of learning rate on the SARSA/Q-learning gap\n"
                 f"({num_episodes} episodes, mean +/- std over {SEEDS_PER_SETTING} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("sweep_alpha.png", dpi=150)
    print("Saved sweep_alpha.png")


def sweep_training_budget(episode_counts=(500, 1000, 2000, 4000, 8000), alpha=0.1):
    seeds = range(SEEDS_PER_SETTING)
    sarsa_mean, sarsa_std, q_mean, q_std = [], [], [], []

    for n in episode_counts:
        print(f"  num_episodes={n} ...")
        m, s = edge_advantage(train_sarsa, n, alpha, seeds)
        sarsa_mean.append(m); sarsa_std.append(s)
        m, s = edge_advantage(train_qlearning, n, alpha, seeds)
        q_mean.append(m); q_std.append(s)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(episode_counts, sarsa_mean, yerr=sarsa_std, marker="o", label="SARSA", capsize=3)
    ax.errorbar(episode_counts, q_mean, yerr=q_std, marker="s", label="Q-learning", capsize=3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel(f"Q(wear={EDGE_WEAR}, FastGrasp) - Q(wear={EDGE_WEAR}, CarefulGrasp)")
    ax.set_title(f"How much experience is needed before the gap appears?\n"
                 f"(alpha={alpha}, mean +/- std over {SEEDS_PER_SETTING} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("sweep_episodes.png", dpi=150)
    print("Saved sweep_episodes.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["alpha", "episodes", "both"], default="both")
    args = parser.parse_args()

    if args.sweep in ("alpha", "both"):
        print("Running learning-rate sweep...")
        sweep_learning_rate()
    if args.sweep in ("episodes", "both"):
        print("Running training-budget sweep...")
        sweep_training_budget()
