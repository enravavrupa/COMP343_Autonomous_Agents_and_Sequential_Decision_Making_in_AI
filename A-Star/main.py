import sys
from environment import SurgicalLabEnvironment
from planner import ForkliftPlanner
from visualizer import visualize_paths

# =====================================================================
# CONFIGURATION PROFILES
# =====================================================================
# 1. The Original Inbuilt Defaults (Preserving your exact legacy setup)
INBUILT_WIDTH = 20
INBUILT_HEIGHT = 15
INBUILT_START = (0, 0)
INBUILT_GOAL = (19, 14)  # (20-1, 15-1)

# 2. Try loading the visual editor configuration if it exists
try:
    import config
    HAS_CUSTOM_CONFIG = True
except ImportError:
    HAS_CUSTOM_CONFIG = False


def run_experiment(map_type: str = 'custom', density: float = 0.20):
    """
    Runs A* benchmarks.
    Profiles:
      - 'custom': Loads dimensions, start, goal, and walls from visual editor.
      - 'surgical': Restores the exact original built-in environment & layout.
      - 'random': Legacy random noise testing.
    """
    
    # -----------------------------------------------------------------
    # PROFILE 1: The Original Built-in Configuration
    # -----------------------------------------------------------------
    if map_type == 'surgical':
        print("🏢 [PROFILE: INBUILT] Loading original fixed Surgical Lab layout...")
        lab = SurgicalLabEnvironment(map_type='surgical', width=INBUILT_WIDTH, height=INBUILT_HEIGHT)
        start_pos = INBUILT_START
        goal_pos = INBUILT_GOAL

    # -----------------------------------------------------------------
    # PROFILE 2: The New Visual Editor Configuration
    # -----------------------------------------------------------------
    elif map_type == 'custom':
        if HAS_CUSTOM_CONFIG:
            print("🎨 [PROFILE: CUSTOM] Loading hand-drawn layout from visual editor (config.py)...")
            lab = SurgicalLabEnvironment(
                map_type='custom', 
                width=getattr(config, 'DEFAULT_WIDTH', INBUILT_WIDTH),
                height=getattr(config, 'DEFAULT_HEIGHT', INBUILT_HEIGHT)
            )
            start_pos = getattr(config, 'START_POSITION', INBUILT_START)
            goal_pos = getattr(config, 'GOAL_POSITION', (lab.width - 1, lab.height - 1))
        else:
            print("⚠️ Warning: config.py not found! Skipping custom test run...")
            return

    # -----------------------------------------------------------------
    # PROFILE 3: Procedural Random Map
    # -----------------------------------------------------------------
    elif map_type == 'random':
        print(f"🧬 [PROFILE: RANDOM] Generating procedural map with {density*100}% obstacle density...")
        lab = SurgicalLabEnvironment(map_type='random', width=INBUILT_WIDTH, height=INBUILT_HEIGHT, obstacle_density=density)
        start_pos = (0, 0)
        goal_pos = (lab.width - 1, lab.height - 1)
        
        # Ensure start and goal positions are carved out safely
        lab.grid[start_pos[1]][start_pos[0]] = 0
        lab.grid[goal_pos[1]][goal_pos[0]] = 0

    else:
        raise ValueError(f"Unknown map profile type: {map_type}")

    # -----------------------------------------------------------------
    # EXECUTION & BENCHMARKING (Shared Engine)
    # -----------------------------------------------------------------
    planner = ForkliftPlanner(lab)
    
    if not lab.is_valid(start_pos[0], start_pos[1]) or not lab.is_valid(goal_pos[0], goal_pos[1]):
        print(f"❌ ERROR: Start {start_pos} or Goal {goal_pos} is blocked by a wall! Execution halted.")
        return

    # Compute Paths
    path_man, expanded_man, time_man = planner.a_star(start_pos, goal_pos, 'manhattan')
    path_euc, expanded_euc, time_euc = planner.a_star(start_pos, goal_pos, 'euclidean')
    
    # Print benchmarking breakdown
    print(f"\n{'Heuristic':<15} {'Path Length':<15} {'Nodes Expanded':<20} {'Time (ms)':<15}")
    print("-" * 65)
    print(f"{'Manhattan':<15} {len(path_man):<15} {expanded_man:<20} {time_man*1000:<15.2f}")
    print(f"{'Euclidean':<15} {len(path_euc):<15} {expanded_euc:<20} {time_euc*1000:<15.2f}\n")
    
    # Fire up Visualizer
    stats = {'Manhattan': expanded_man, 'Euclidean': expanded_euc}
    visualize_paths(lab, path_man, path_euc, start_pos, goal_pos, stats)


if __name__ == "__main__":
    print("🚀 STARTING TRIPLE-PROFILE COMPARISON EXPERIMENT 🚀\n")
    
    # -----------------------------------------------------------------
    # TEST RUN 1: Custom Map Layout from your browser GUI
    # -----------------------------------------------------------------
    print("=" * 70)
    print("🧪 TEST #1: RUNNING CUSTOM VISUAL EDITOR PROFILE ('config.py')")
    print("=" * 70)
    run_experiment(map_type='custom')
    
    print("\n\n") # Space things out for clean terminal readability
    
    # -----------------------------------------------------------------
    # TEST RUN 2: Inbuilt Legacy Surgical Lab Configuration
    # -----------------------------------------------------------------
    print("=" * 70)
    print("🧪 TEST #2: RUNNING INBUILT LEGACY SURGICAL MAP PROFILE")
    print("=" * 70)
    run_experiment(map_type='surgical')
    
    print("\n\n")
    
    # -----------------------------------------------------------------
    # TEST RUN 3: Procedural Random Noise Configuration
    # -----------------------------------------------------------------
    print("=" * 70)
    print("🧪 TEST #3: RUNNING PROCEDURAL RANDOM MAP PROFILE")
    print("=" * 70)
    # You can change the density float here (e.g., 0.30 = 30% filled with walls)
    run_experiment(map_type='random', density=0.25)
    
    print("\n🏁 All 3 experiments complete!")
