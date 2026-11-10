"""
compare_policies.py
=====================
Run this AFTER both sarsa_lab.py and qlearning_lab.py pass test_correctness.py.

It trains both algorithms across several seeds, then produces:
  1. A plot of average episode return vs. training episode, for both algorithms.
  2. A plot of the "FastGrasp advantage" (Q(wear,FAST) - Q(wear,CAREFUL)) as a
     function of wear, for both algorithms -- the main result of this lab.

If `wandb` is installed and you're logged in, this also logs both runs to
a project called "packbot-sarsa-qlearning" (the hard submission requirement
for this lab, same as the other labs this term). If wandb isn't available,
everything still runs and saves plots locally.

Usage:
    python3 compare_policies.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sarsa_lab import train_sarsa
from qlearning_lab import train_qlearning
from rl_utils import fast_advantage_curve
from packbot_env import N_ITEM_TYPES

try:
    import wandb
    HAVE_WANDB = True
except ImportError:
    HAVE_WANDB = False

NUM_SEEDS = 10
NUM_EPISODES = 4000


def smooth(x, window=50):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def run_all_seeds(train_fn, num_seeds=NUM_SEEDS, num_episodes=NUM_EPISODES):
    all_returns = []
    all_advantage_curves = []
    for seed in range(num_seeds):
        Q, returns, _ = train_fn(num_episodes=num_episodes, seed=seed)
        all_returns.append(returns)
        all_advantage_curves.append(fast_advantage_curve(Q, item_type=0))
    return np.array(all_returns), np.array(all_advantage_curves)


def query_policy_action(Q, item_type, wear):
    """
    Helper for the predict-then-verify writeup: what does this trained
    Q table's greedy policy do at a specific (item_type, wear)?
    """
    from packbot_env import state_index, ACTION_NAMES
    s = state_index(item_type, wear)
    return ACTION_NAMES[int(np.argmax(Q[s]))]


def main():
    if HAVE_WANDB:
        run = wandb.init(project="packbot-sarsa-qlearning",
                          config={"num_seeds": NUM_SEEDS, "num_episodes": NUM_EPISODES})

    print(f"Training SARSA across {NUM_SEEDS} seeds...")
    sarsa_returns, sarsa_adv = run_all_seeds(train_sarsa)
    print(f"Training Q-learning across {NUM_SEEDS} seeds...")
    q_returns, q_adv = run_all_seeds(train_qlearning)

    # --- Plot 1: learning curves ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(smooth(sarsa_returns.mean(axis=0)), label="SARSA")
    ax.plot(smooth(q_returns.mean(axis=0)), label="Q-learning")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (smoothed, mean over seeds)")
    ax.set_title("PackBot: learning curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=150)
    print("Saved learning_curves.png")

    # --- Plot 2: the main result ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    wear_levels = np.arange(sarsa_adv.shape[1])
    sarsa_mean = sarsa_adv.mean(axis=0)
    sarsa_std = sarsa_adv.std(axis=0)
    q_mean = q_adv.mean(axis=0)
    q_std = q_adv.std(axis=0)

    ax2.plot(wear_levels, sarsa_mean, label="SARSA", marker="o")
    ax2.fill_between(wear_levels, sarsa_mean - sarsa_std, sarsa_mean + sarsa_std, alpha=0.2)
    ax2.plot(wear_levels, q_mean, label="Q-learning", marker="s")
    ax2.fill_between(wear_levels, q_mean - q_std, q_mean + q_std, alpha=0.2)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_xlabel("Wear level")
    ax2.set_ylabel("Q(wear, FastGrasp) - Q(wear, CarefulGrasp)")
    ax2.set_title("FastGrasp advantage near the breakdown edge\n(Standard items, mean +/- std over seeds)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("fast_advantage.png", dpi=150)
    print("Saved fast_advantage.png")

    if HAVE_WANDB:
        wandb.log({
            "learning_curves": wandb.Image("learning_curves.png"),
            "fast_advantage": wandb.Image("fast_advantage.png"),
        })
        run.finish()

    print("\nDone. Use fast_advantage.png as evidence in your written predict-then-verify section.")


if __name__ == "__main__":
    main()
