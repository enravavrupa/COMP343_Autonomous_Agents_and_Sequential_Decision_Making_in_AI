"""
W&B logging convention for this lab. Logging is a HARD submission
requirement, not optional — ungraded if missing.

Run naming: {your_roll_number}_{algorithm}_{module}
  e.g. "2024CS1234_td0_module2"

Required logged config per run:
  algorithm, seed, budget (or checkpoint step), gamma, is_slippery, alpha_schedule

Required logged metrics per run:
  mse_vs_ground_truth, steps_used, episodes_used
"""
import wandb


def init_run(algorithm: str, module: str, roll_number: str, config: dict):
    return wandb.init(
        project="rl-prediction-lab",
        name=f"{roll_number}_{algorithm}_{module}",
        config=config,
    )


def log_result(run, mse: float, steps_used: int, episodes_used: int, step: int | None = None):
    run.log(
        {"mse_vs_ground_truth": mse, "steps_used": steps_used, "episodes_used": episodes_used},
        step=step,
    )
