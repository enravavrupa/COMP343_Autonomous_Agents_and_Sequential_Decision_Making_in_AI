"""
compare_engines.py — check any two engines against each other
=================================================================

Run:
    python compare_engines.py                                  # pi_lab vs vi_solution (default)
    python compare_engines.py --engine-a vi_lab --engine-b vi_solution
    python compare_engines.py --engine-a pi_lab --engine-b pi_solution
    python compare_engines.py --engine-a pi_solution --engine-b vi_solution

Available engine names:
    pi_lab        -> policy_iteration_lab.PolicyIterationEngine        (yours, has TODOs)
    pi_solution   -> policy_iteration_solution.PolicyIterationSolutionEngine (instructor reference)
    vi_lab        -> value_iteration_lab.ValueIterationLabEngine       (yours, has TODOs)
    vi_solution   -> value_iteration_solution.ValueIterationEngine     (instructor reference)

What it does:
  1. Generates ONE random layout (grid changes every run unless you
     pass --seed).
  2. Builds P, R for that layout with mdp_builder.build_mdp_tensors().
  3. Solves it with engine A and engine B.
  4. Reports iterations, policy agreement, value-function gap, and
     average simulated return for each.

Two correct solvers can output DIFFERENT policies when several actions
tie for optimal in a given state — that's expected. What should NOT
differ (beyond floating-point tolerance) is the value function and the
average simulated return. If agreement is low AND average return is
also very different, that's a sign of a bug, not a tie-breaking
difference.
"""

import argparse
import random

import numpy as np

from env import TreasureHuntEnvironment
from mdp_builder import build_mdp_tensors

from policy_iteration_lab import PolicyIterationEngine as PILabEngine
from policy_iteration_solution import PolicyIterationSolutionEngine
from value_iteration_lab import ValueIterationLabEngine
from value_iteration_solution import ValueIterationEngine


ENGINES = {
    "pi_lab": PILabEngine,
    "pi_solution": PolicyIterationSolutionEngine,
    "vi_lab": ValueIterationLabEngine,
    "vi_solution": ValueIterationEngine,
}


def run_policy_episode(layout, policy, state_to_index, rng, max_steps=200):
    env = TreasureHuntEnvironment(layout=layout, rng=rng)
    env.reset()
    total_reward = 0
    for _ in range(max_steps):
        state = env.get_state()
        action = int(policy[state_to_index[state]])
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def solve_or_report_incomplete(engine, engine_name, P, R, discount):
    try:
        return engine.solve(P, R, discount=discount)
    except NotImplementedError as exc:
        print(f"[{engine_name}] is not finished yet: {exc}")
        raise SystemExit(1)


def main(engine_a_name, engine_b_name, seed, num_episodes, discount,
         rows, cols, num_walls, num_danger):
    rng = random.Random(seed)

    layout = TreasureHuntEnvironment.generate_random_layout(
        rows=rows, cols=cols, num_walls=num_walls, num_danger=num_danger, rng=rng,
    )
    env = TreasureHuntEnvironment(layout=layout, rng=rng)

    P, R, states, state_to_index = build_mdp_tensors(env)
    print(f"Layout: {rows}x{cols} grid, start={layout['start']}, "
          f"treasure={layout['treasure']}, goal={layout['goal']}, "
          f"walls={layout['walls']}, danger={layout['danger']}")
    print(f"State space size: {len(states)} states, 4 actions\n")

    engine_a = ENGINES[engine_a_name]()
    engine_b = ENGINES[engine_b_name]()

    result_a = solve_or_report_incomplete(engine_a, engine_a_name, P, R, discount)
    result_b = solve_or_report_incomplete(engine_b, engine_b_name, P, R, discount)

    agreement = float(np.mean(result_a.policy == result_b.policy))
    value_gap = float(np.max(np.abs(result_a.value_function - result_b.value_function)))

    print(f"{result_a.engine_name} converged in {result_a.iterations} outer iterations")
    print(f"{result_b.engine_name} converged in {result_b.iterations} outer iterations")
    print(f"Policy agreement (fraction of states): {agreement:.2%}")
    print(f"Max |V_A(s) - V_B(s)| over all states: {value_gap:.6f}\n")

    returns_a = [run_policy_episode(layout, result_a.policy, state_to_index, rng)
                 for _ in range(num_episodes)]
    returns_b = [run_policy_episode(layout, result_b.policy, state_to_index, rng)
                 for _ in range(num_episodes)]

    print(f"Avg return over {num_episodes} episodes — {result_a.engine_name}: "
          f"{np.mean(returns_a):.2f} (std {np.std(returns_a):.2f})")
    print(f"Avg return over {num_episodes} episodes — {result_b.engine_name}: "
          f"{np.mean(returns_b):.2f} (std {np.std(returns_b):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine-a", choices=ENGINES.keys(), default="pi_lab")
    parser.add_argument("--engine-b", choices=ENGINES.keys(), default="vi_solution")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--walls", type=int, default=2)
    parser.add_argument("--danger", type=int, default=1)
    args = parser.parse_args()

    main(args.engine_a, args.engine_b, args.seed, args.episodes, args.discount,
         args.rows, args.cols, args.walls, args.danger)
