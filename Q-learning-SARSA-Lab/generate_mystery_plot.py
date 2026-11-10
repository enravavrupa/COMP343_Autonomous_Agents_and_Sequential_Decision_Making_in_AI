"""
generate_mystery_plot.py
==========================
INSTRUCTOR-ONLY TOOL. Not distributed to students.

Generates the optional "guess which algorithm" plot referenced at the end
of LAB_HANDOUT.md: trains ONE of {SARSA, Q-learning} (chosen at random,
or pass --algo explicitly for a fixed answer key across sections), saves
an unlabeled version of the FastGrasp-advantage plot as mystery_plot.png,
and writes the answer to mystery_answer.txt (keep this yourself).

Usage:
    python3 generate_mystery_plot.py               # random choice
    python3 generate_mystery_plot.py --algo sarsa   # fixed choice
    python3 generate_mystery_plot.py --algo qlearning
"""

import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sarsa_solution import train_sarsa
from qlearning_solution import train_qlearning
from rl_utils import fast_advantage_curve

NUM_SEEDS = 10
NUM_EPISODES = 4000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sarsa", "qlearning"], default=None)
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for the coin flip, if --algo not given.")
    args = parser.parse_args()

    if args.algo is None:
        rng = random.Random(args.seed)
        algo = rng.choice(["sarsa", "qlearning"])
    else:
        algo = args.algo

    train_fn = train_sarsa if algo == "sarsa" else train_qlearning

    curves = []
    for seed in range(NUM_SEEDS):
        Q, _, _ = train_fn(num_episodes=NUM_EPISODES, seed=seed)
        curves.append(fast_advantage_curve(Q, item_type=0))
    curves = np.array(curves)
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    wear_levels = np.arange(curves.shape[1])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(wear_levels, mean, marker="o", color="tab:purple")
    ax.fill_between(wear_levels, mean - std, mean + std, alpha=0.2, color="tab:purple")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Wear level")
    ax.set_ylabel("Q(wear, FastGrasp) - Q(wear, CarefulGrasp)")
    ax.set_title("Mystery agent: SARSA or Q-learning?")
    fig.tight_layout()
    fig.savefig("mystery_plot.png", dpi=150)

    with open("mystery_answer.txt", "w") as f:
        f.write(f"Algorithm used: {algo}\n")

    print(f"Saved mystery_plot.png (answer written to mystery_answer.txt, algo={algo})")


if __name__ == "__main__":
    main()
