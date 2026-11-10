"""
main.py — entry point

Usage:
    python main.py                       # human play only, no engine attached
    python main.py --engine vi           # policy mode uses Value Iteration (reference)
    python main.py --engine pi           # policy mode uses Policy Iteration (reference)
    python main.py --engine vi_lab       # policy mode uses YOUR value iteration
    python main.py --engine pi_lab       # policy mode uses YOUR policy iteration
    python main.py --engine pi_lab --rows 5 --cols 5 --walls 3 --danger 2

Press P in-game to solve the current random layout and enter policy mode.
"""

import argparse

from game_gui import TreasureHuntGUI
from value_iteration_solution import ValueIterationEngine
from value_iteration_lab import ValueIterationLabEngine
from policy_iteration_lab import PolicyIterationEngine
from policy_iteration_solution import PolicyIterationSolutionEngine


ENGINES = {
    "vi": ValueIterationEngine,                  # instructor reference, always works
    "vi_lab": ValueIterationLabEngine,            # your value-iteration exercise
    "pi": PolicyIterationSolutionEngine,          # instructor reference, always works
    "pi_lab": PolicyIterationEngine,              # your policy-iteration exercise
    "none": lambda: None,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES.keys(), default="vi",
                         help="Which PolicyEngine to attach for policy mode (default: vi)")
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--walls", type=int, default=2)
    parser.add_argument("--danger", type=int, default=1)
    args = parser.parse_args()

    engine = ENGINES[args.engine]()

    gui = TreasureHuntGUI(
        engine=engine,
        discount=args.discount,
        rows=args.rows,
        cols=args.cols,
        num_walls=args.walls,
        num_danger=args.danger,
    )
    gui.run()


if __name__ == "__main__":
    main()
