"""
evaluate_policy.py — generic "how good is this policy?" checker
====================================================================

This does NOT care which engine produced a policy — hand it any
np.ndarray of shape (S,) (action per state index) for a given
environment, and it reports:

  1. EXACT value function under that policy (via policy evaluation,
     the same math as _policy_evaluation in the PI lab — but this
     copy is independent, so it also serves as a second check on that
     code if you want one).
  2. SIMULATED average return over N episodes (Monte-Carlo rollouts),
     which should closely match #1 — the two computations are
     completely different code paths (linear algebra vs. sampling),
     so this cross-check function is a useful correctness test on
     its own: if they disagree by a lot, something is wrong somewhere.
  3. REGRET against a known-good optimal policy (from
     value_iteration_solution.py): V*(s) - V_policy(s) for every
     state, reported as max and mean regret. A correct/optimal policy
     has ~zero regret everywhere; a weak or buggy policy shows up as
     large positive regret in specific states.

Usage as a script (evaluates a fresh random policy for demonstration):
    python evaluate_policy.py --engine pi_lab --seed 3

Usage as a library:
    from evaluate_policy import evaluate_policy
    report = evaluate_policy(env, policy, state_to_index, discount=0.95)
"""

import argparse
import random

import numpy as np

from env import TreasureHuntEnvironment
from mdp_builder import build_mdp_tensors
from value_iteration_solution import ValueIterationEngine


def exact_value_of_policy(policy, P, R, discount, tolerance=1e-8, max_sweeps=10000):
    """Same math as policy evaluation in the PI lab, written
    independently — used here purely as a checking tool."""
    S = P.shape[1]
    value = np.zeros(S)
    P_pi = P[policy, np.arange(S), :]
    R_pi = R[np.arange(S), policy]
    for _ in range(max_sweeps):
        new_value = R_pi + discount * P_pi.dot(value)
        if np.max(np.abs(new_value - value)) < tolerance:
            value = new_value
            break
        value = new_value
    return value


def simulate_average_return(env_layout, policy, state_to_index, rng, discount,
                             num_episodes=1000, max_steps=200):
    """
    Returns DISCOUNTED per-episode returns: sum_t (discount**t * reward_t).

    This has to use the same discount as the exact value function or the
    two numbers are answering different questions and will never agree
    even for a perfect policy — undiscounted total reward and discounted
    expected return are simply not the same quantity. (First version of
    this function summed raw rewards; caught by comparing against the
    exact value function on a test run — a good example of exactly the
    cross-check this module exists to do.)
    """
    returns = []
    for _ in range(num_episodes):
        env = TreasureHuntEnvironment(layout=env_layout, rng=rng)
        env.reset()
        total = 0.0
        discount_factor = 1.0
        for _ in range(max_steps):
            state = env.get_state()
            action = int(policy[state_to_index[state]])
            _, reward, done, _ = env.step(action)
            total += discount_factor * reward
            discount_factor *= discount
            if done:
                break
        returns.append(total)
    return np.array(returns)


def evaluate_policy(env, policy, state_to_index, discount=0.95,
                     num_episodes=1000, rng=None):
    """
    Returns a dict report:
        value_function   : (S,) exact V under this policy
        simulated_returns : (num_episodes,) array of simulated returns
        simulated_mean / simulated_std
        optimal_value_function : (S,) V* from a trusted reference solver
        max_regret / mean_regret
    """
    rng = rng or random.Random()

    P, R, states, _ = build_mdp_tensors(env)
    value_function = exact_value_of_policy(policy, P, R, discount)

    layout = dict(rows=env.rows, cols=env.cols, start=env.start,
                  treasure=env.treasure, goal=env.goal,
                  walls=env.walls, danger=env.danger)
    simulated_returns = simulate_average_return(
        layout, policy, state_to_index, rng, discount, num_episodes=num_episodes,
    )

    optimal_result = ValueIterationEngine().solve(P, R, discount=discount)
    regret = optimal_result.value_function - value_function

    return {
        "value_function": value_function,
        "simulated_returns": simulated_returns,
        "simulated_mean": float(simulated_returns.mean()),
        "simulated_std": float(simulated_returns.std()),
        "optimal_value_function": optimal_result.value_function,
        "max_regret": float(np.max(regret)),
        "mean_regret": float(np.mean(regret)),
    }


def _demo():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["pi_lab", "pi_solution", "vi_lab", "vi_solution"],
                         default="vi_solution")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    from compare_engines import ENGINES  # reuse the same registry

    rng = random.Random(args.seed)
    layout = TreasureHuntEnvironment.generate_random_layout(rng=rng)
    env = TreasureHuntEnvironment(layout=layout, rng=rng)
    P, R, states, state_to_index = build_mdp_tensors(env)

    engine = ENGINES[args.engine]()
    result = engine.solve(P, R, discount=0.95)

    report = evaluate_policy(env, result.policy, state_to_index,
                              num_episodes=args.episodes, rng=rng)

    print(f"Engine: {result.engine_name}")
    print(f"Simulated DISCOUNTED return: {report['simulated_mean']:.2f} "
          f"(std {report['simulated_std']:.2f}) over {args.episodes} episodes")
    print(f"Exact discounted value at start state: "
          f"{report['value_function'][state_to_index[(env.start, False)]]:.2f}")
    print("(these two numbers should be close — that's the cross-check)")
    print(f"Max regret vs. optimal (any state): {report['max_regret']:.4f}")
    print(f"Mean regret vs. optimal (all states): {report['mean_regret']:.4f}")


if __name__ == "__main__":
    _demo()
