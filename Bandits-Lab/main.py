from environment import WarehouseEnvironment
from agents import EpsilonGreedyAgent, UCBAgent
from visuals import render_warehouse, plot_agent_comparison

def run_agent_demo(agent_name, agent, env, steps=1000):
    print("\n==========================================")
    print(f" Running Simulation: {agent_name}")
    print("==========================================")
    env.reset()
    total_reward = 0
    
    for t in range(steps):
        action = agent.select_action()
        state, reward = env.step(action)
        agent.update(action, reward)
        total_reward += reward

    print(f"Total Steps:          {steps}")
    print(f"Total Reward Earned:  {total_reward}")
    print(f"True Zone Rates:      {env.zone_rates}")
    print(f"Estimated Q-Values:   {agent.Q.round(3)}")
    print(f"Action Counts (N):    {agent.N}")
    
    return total_reward, agent

if __name__ == "__main__":
    env = WarehouseEnvironment()
    
    # 1. Fixed Epsilon-Greedy Agent
    eps_agent = EpsilonGreedyAgent(n_arms=env.n_arms, epsilon=0.1)
    eps_reward, eps_agent = run_agent_demo("Fixed Epsilon-Greedy (eps=0.1)", eps_agent, env)
    
    # 2. Decaying Epsilon-Greedy Agent
    try:
        decay_agent = EpsilonGreedyAgent(
            n_arms=env.n_arms, 
            epsilon=1.0, 
            decay_type='exponential', 
            decay_rate=0.995, 
            min_epsilon=0.01
        )
    except TypeError:
        decay_agent = EpsilonGreedyAgent(n_arms=env.n_arms, epsilon=1.0)
        decay_agent.decay_type = 'exponential'
        decay_agent.decay_rate = 0.995
        decay_agent.min_epsilon = 0.01

    decay_reward, decay_agent = run_agent_demo("Decaying Epsilon-Greedy", decay_agent, env)

    # 3. UCB Agent
    ucb_agent = UCBAgent(n_arms=env.n_arms, c=1.0)
    ucb_reward, ucb_agent = run_agent_demo("UCB Agent (c=1.0)", ucb_agent, env)

    # Dictionary storing results with clear labels
    agent_results = {
        "Fixed ε-Greedy": (eps_agent, eps_reward),
        "Decaying ε-Greedy": (decay_agent, decay_reward),
        "UCB Agent": (ucb_agent, ucb_reward)
    }

    # Summary Table
    print("\n" + "="*45)
    print("           ALGORITHM COMPARISON")
    print("="*45)
    for name, (_, reward) in agent_results.items():
        print(f"{name:<25} Total Reward: {reward}")
    print("="*45)

    # Generate labeled comparison plots for all algorithms
    print("\nGenerating visual comparison plot...")
    plot_agent_comparison(agent_results, env, save_filename="algorithm_comparison.png")
