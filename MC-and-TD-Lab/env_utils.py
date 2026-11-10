"""
Environment utilities: FrozenLake-v1 wrapper conventions + exact ground-truth
V^pi(s) computation for the uniform random policy.

This file is COMPLETE — do not modify. It is infrastructure, not an exercise.
The point of this lab is prediction algorithms, not re-deriving policy
evaluation; you already did that in the Treasure Hunt MDP lab.

Run this file directly to (re)generate data/ground_truth_values.json:
    python env_utils.py
"""
import json
import os

import gymnasium as gym
import numpy as np

GAMMA = 0.99
IS_SLIPPERY = True  # fixed for Modules 1, 2, 4 — Module 3 explicitly varies this
N_STATES = 16
N_ACTIONS = 4
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ground_truth_values.json")


def make_env(is_slippery: bool = IS_SLIPPERY, seed: int | None = None):
    """Create a FrozenLake-v1 env with the lab's fixed conventions."""
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _transition_model(is_slippery: bool = IS_SLIPPERY):
    """
    Extract P[s][a] = list of (prob, next_state, reward, terminated) from the
    env's internal desc, using gymnasium's own unwrapped .P attribute so we
    never hand-derive slip dynamics ourselves (error-prone, unnecessary).
    """
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
    P = env.unwrapped.P  # dict: state -> action -> list of (prob, next_state, reward, done)
    env.close()
    return P


def compute_ground_truth_v(is_slippery: bool = IS_SLIPPERY, gamma: float = GAMMA) -> np.ndarray:
    """
    Exact V^pi(s) for the uniform random policy via direct linear solve of
    the Bellman expectation equations:

        V = R_pi + gamma * P_pi @ V   =>   (I - gamma * P_pi) V = R_pi

    Terminal states (goal, holes) are pinned to V(s) = 0 by convention — the
    reward is earned on the transition INTO the terminal state, not stored
    at it. We enforce this explicitly rather than relying on the linear
    solve to discover it, since a subtly wrong P_pi could otherwise produce
    a small nonzero "leak" at terminal states that's easy to miss.
    """
    P = _transition_model(is_slippery)
    terminal = np.zeros(N_STATES, dtype=bool)

    P_pi = np.zeros((N_STATES, N_STATES))
    R_pi = np.zeros(N_STATES)

    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            for prob, s_next, reward, done in P[s][a]:
                P_pi[s, s_next] += 0.25 * prob
                R_pi[s] += 0.25 * prob * reward
                if done:
                    terminal[s_next] = True

    A = np.eye(N_STATES) - gamma * P_pi
    V = np.linalg.solve(A, R_pi)
    V[terminal] = 0.0
    return V


def load_or_generate_ground_truth(is_slippery: bool = IS_SLIPPERY, gamma: float = GAMMA) -> np.ndarray:
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            cached = json.load(f)
        if cached.get("is_slippery") == is_slippery and cached.get("gamma") == gamma:
            return np.array(cached["V"])
    V = compute_ground_truth_v(is_slippery, gamma)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump({"is_slippery": is_slippery, "gamma": gamma, "V": V.tolist()}, f, indent=2)
    return V


if __name__ == "__main__":
    V = load_or_generate_ground_truth()
    grid = V.reshape(4, 4)
    print("Ground truth V^pi(s), uniform random policy, is_slippery=True, gamma=0.99")
    print(np.round(grid, 4))
    print(f"\nValue range: [{V.min():.4f}, {V.max():.4f}]  (use this to sanity-check your MSE threshold)")
    print(f"Cached to {DATA_PATH}")
