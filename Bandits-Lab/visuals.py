import matplotlib.pyplot as plt
import numpy as np
def render_warehouse(environment, save_filename="warehouse_bandit_status.png"):
    """Renders the environment's true zone probabilities and single-step active state."""
    zones = [f"Zone {i+1}" for i in range(environment.n_arms)]
    rates = environment.zone_rates
    current_state = environment.state

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Zone Probabilities
    bars = ax1.bar(zones, rates, color='#2b5c8f')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Probability of Event")
    ax1.set_title("Warehouse Zone Configuration")
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')

    # Plot 2: Current Active State
    colors = ['#2ecc71' if s == 1 else '#e74c3c' for s in current_state]
    ax2.bar(zones, current_state, color=colors)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Inactive', 'Active'])
    ax2.set_title("Current Zone Activity State")

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Visual saved as {save_filename}")


def plot_agent_comparison(agent_results, env, save_filename="algorithm_comparison.png"):
    """
    Plots labeled comparison charts across all tested algorithms:
    1. Learned Q-Values vs. True Zone Probabilities
    2. Zone Selection Counts (N)
    3. Total Accumulated Rewards
    """
    zones = [f"Zone {i+1}" for i in range(env.n_arms)]
    x = np.arange(len(zones))
    width = 0.2

    colors = ['#e67e22', '#27ae60', '#2980b9']  # Colors for Fixed, Decaying, and UCB
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Learned Q-Values vs True Rates
    ax1.bar(x - width, env.zone_rates, width, label='True Rate', color='#7f8c8d', alpha=0.5)
    for idx, (name, (agent, _)) in enumerate(agent_results.items()):
        offset = idx * width
        ax1.bar(x + offset, agent.Q, width, label=name, color=colors[idx % len(colors)])

    ax1.set_ylabel("Expected Value (Q)")
    ax1.set_title("Learned Q-Values vs True Rates")
    ax1.set_xticks(x)
    ax1.set_xticklabels(zones)
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Zone Selection Counts (N)
    for idx, (name, (agent, _)) in enumerate(agent_results.items()):
        offset = (idx - 1) * width
        ax2.bar(x + offset, agent.N, width, label=name, color=colors[idx % len(colors)])

    ax2.set_ylabel("Times Zone Selected (N)")
    ax2.set_title("Zone Selection Frequency")
    ax2.set_xticks(x)
    ax2.set_xticklabels(zones)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # 3. Total Reward Comparison
    names = list(agent_results.keys())
    rewards = [reward for _, reward in agent_results.values()]
    bars = ax3.bar(names, rewards, color=colors[:len(names)])
    
    ax3.set_ylabel("Total Reward Accumulated")
    ax3.set_title("Overall Reward Comparison")
    ax3.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Comparison plot saved as {save_filename}")
    
def plot_regret_curves(eps_regret, ucb_regret, random_regret=None, save_filename="regret_comparison.png"):
    """
    Plots cumulative regret over time steps (T) comparing agent algorithms.
    """
    plt.figure(figsize=(8, 5))
    steps = np.arange(1, len(eps_regret) + 1)
    
    if random_regret is not None:
        plt.plot(steps, random_regret, label="Random Agent", color="#7f8c8d", linestyle="--", linewidth=1.5)
    
    plt.plot(steps, eps_regret, label="Epsilon-Greedy (ε=0.1)", color="#e67e22", linewidth=2)
    plt.plot(steps, ucb_regret, label="UCB (c=1.0)", color="#2980b9", linewidth=2)
    
    plt.xlabel("Time Step (T)")
    plt.ylabel("Cumulative Expected Regret")
    plt.title("Cumulative Regret Growth: Linear vs. Logarithmic")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Regret curve plot saved as {save_filename}")
    
def plot_hyperparameter_sweep(epsilons, eps_rewards, c_values, ucb_rewards, save_filename="hyperparameter_sweep.png"):
    """
    Plots cumulative rewards across different values of epsilon and c.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Epsilon Sweep Plot
    ax1.plot(epsilons, eps_rewards, marker='o', color='#e67e22', linewidth=2)
    ax1.set_xlabel("Epsilon (ε)")
    ax1.set_ylabel("Average Cumulative Reward")
    ax1.set_title("Epsilon-Greedy Sensitivity")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # UCB Sweep Plot
    ax2.plot(c_values, ucb_rewards, marker='s', color='#2980b9', linewidth=2)
    ax2.set_xlabel("Exploration Constant (c)")
    ax2.set_ylabel("Average Cumulative Reward")
    ax2.set_title("UCB Constant Sensitivity")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Hyperparameter sweep plot saved as {save_filename}")
