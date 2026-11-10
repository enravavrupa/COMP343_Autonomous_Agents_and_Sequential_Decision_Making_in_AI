import numpy as np
from hmm_environment import WarehouseHMMEnvironment
from hmm_filter import HMMStateEstimator

MAP_TINY = """
#####
#...#
#####
"""

def print_belief_grid(env: WarehouseHMMEnvironment, agent: HMMStateEstimator, step_name: str):
    """Renders the 2D grid in the terminal with probabilities formatted to two decimal places."""
    print(f"\n--- {step_name} ---")
    
    for r in range(env.height):
        row_string = ""
        for c in range(env.width):
            if env.grid[r, c] == 1:  # It is a wall
                row_string += "[ #### ] "
            else:
                state_idx = env.coord_to_state(r, c)
                prob = agent.belief_state[state_idx]
                
                # Highlight the most likely state with an asterisk
                if state_idx == agent.get_most_likely_state() and prob > 0:
                    row_string += f"[*{(prob * 100):02.0f}%*] "
                else:
                    row_string += f"[ {(prob * 100):02.0f}% ] "
        print(row_string)
    print("-" * 45)

def run_debug_sequence():
    # 1. Initialize with standard 80% accurate sensors
    env = WarehouseHMMEnvironment(MAP_TINY, sensor_accuracy=0.8)
    agent = HMMStateEstimator(env.num_states, env.T, env.E)
    
    print_belief_grid(env, agent, "INITIAL STATE (Uniform Prior)")
    
    # 2. Sequence: Move East (2), Observe 2 walls (hallway)
    print("\nAction: Move EAST. Sensor Reads: 2 Walls.")
    agent.bayesian_filter_step(action=2, observation=2)
    print_belief_grid(env, agent, "STEP 1 POSTERIOR")
    
    # 3. Sequence: Move East (2), Observe 3 walls (dead-end)
    print("\nAction: Move EAST. Sensor Reads: 3 Walls.")
    agent.bayesian_filter_step(action=2, observation=3)
    print_belief_grid(env, agent, "STEP 2 POSTERIOR")

if __name__ == "__main__":
    run_debug_sequence()
