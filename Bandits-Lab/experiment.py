import numpy as np
import matplotlib.pyplot as plt

from environment import WarehouseEnvironment
from agents import EpsilonGreedyAgent, UCBAgent
from visuals import plot_hyperparameter_sweep, plot_regret_curves

class RandomAgent:
    """Baseline Random Agent for comparison in regret plots."""
    def __init__(self, n_arms):
        self.n_arms = n_arms
    def select_action(self):
        return np.random.randint(self.n_arms)
    def update(self, action, reward):
        pass

def track_cumulative_regret(agent_class, agent_kwargs, steps=1000, n_runs=50):
    """
    Calculates expected cumulative regret over T steps averaged across multiple runs.
    Formula per step t: Regret_t = v* - E[R_t | a_t]
    """
    sample_env = WarehouseEnvironment()
    v_star = np.max(sample_env.zone_rates)  # Best zone expected reward = 0.8
    all_runs_regret = np.zeros((n_runs, steps))

    for run in range(n_runs):
        env = WarehouseEnvironment()
        agent = agent_class(n_arms=env.n_arms, **agent_kwargs)
        
        cumulative_regret = 0.0
        for t in range(steps):
            action = agent.select_action()
            _, reward = env.step(action)
            agent.update(action, reward)
            
            # Expected instantaneous regret using true zone probability
            instant_regret = v_star - env.zone_rates[action]
            cumulative_regret += instant_regret
            all_runs_regret[run, t] = cumulative_regret

    return np.mean(all_runs_regret, axis=0)

def evaluate_agent(agent_class, param_name, param_value, n_runs=50, steps=1000):
    """Evaluates agent cumulative reward for hyperparameter tuning."""
    total_rewards = []
    for _ in range(n_runs):
        env = WarehouseEnvironment()
        kwargs = {'n_arms': env.n_arms, param_name: param_value}
        agent = agent_class(**kwargs)
        
        run_reward = 0
        for _ in range(steps):
            action = agent.select_action()
            _, reward = env.step(action)
            agent.update(action, reward)
            run_reward += reward
            
        total_rewards.append(run_reward)
        
    return np.mean(total_rewards)

if __name__ == "__main__":
    # 1. Hyperparameter Tuning Sweep # e = 0.1 # c = 5 # these give the highest cumulative rewards
    epsilons = [0.1]
    c_values = [5.0]

    print("--- Running Epsilon-Greedy Parameter Sweep ---")
    eps_rewards = [evaluate_agent(EpsilonGreedyAgent, 'epsilon', eps) for eps in epsilons]

    print("--- Running UCB Parameter Sweep ---")
    ucb_rewards = [evaluate_agent(UCBAgent, 'c', c) for c in c_values]

    plot_hyperparameter_sweep(epsilons, eps_rewards, c_values, ucb_rewards)

    # 2. Cumulative Regret Comparison Analysis
    print("\n--- Running Regret Comparison Experiment (1000 steps, 50 runs) ---")
    
    random_regret = track_cumulative_regret(RandomAgent, {})
    eps_regret = track_cumulative_regret(EpsilonGreedyAgent, {'epsilon': 0.5})
    ucb_regret = track_cumulative_regret(UCBAgent, {'c': 5.0})

    plot_regret_curves(eps_regret, ucb_regret, random_regret)
