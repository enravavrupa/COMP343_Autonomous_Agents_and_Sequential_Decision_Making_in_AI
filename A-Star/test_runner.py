from environment import SurgicalLabEnvironment
from planner import ForkliftPlanner

def run_suite():
    # ----------------------------------------------------
    # TEST 1: Sanity/Trivial Check
    # ----------------------------------------------------
    env_trivial = SurgicalLabEnvironment(map_type='trivial')
    planner = ForkliftPlanner(env_trivial)
    
    start, goal = (1, 5), (8, 5)
    path, expanded, _ = planner.a_star(start, goal)
    
    print(f"Trivial Case: Found path of length {len(path)} expanding {expanded} nodes.")
    env_trivial.print_grid(start, goal, path)
    assert len(path) == 8, "Error: Trivial straight line calculation incorrect"

    # ----------------------------------------------------
    # TEST 2: The Island Dead End 
    # ----------------------------------------------------
    env_island = SurgicalLabEnvironment(map_type='island')
    planner_island = ForkliftPlanner(env_island)
    
    start, goal = (1, 1), (8, 8)
    path, expanded, _ = planner_island.a_star(start, goal)
    
    print(f"Island Case: Found path of length {len(path)} expanding {expanded} nodes.")
    env_island.print_grid(start, goal, path)
    assert path == [], "Error: Algorithm should return an empty list for blocked nodes!"

    # ----------------------------------------------------
    # TEST 3: The U-Trap Concave Wall
    # ----------------------------------------------------
    env_trap = SurgicalLabEnvironment(map_type='trap')
    planner_trap = ForkliftPlanner(env_trap)
    
    start, goal = (2, 5), (8, 5)
    path, expanded, _ = planner_trap.a_star(start, goal)
    
    print(f"Trap Case: Found path of length {len(path)} expanding {expanded} nodes.")
    env_trap.print_grid(start, goal, path)
    assert len(path) > 0, "Error: Failed to route around the U-trap"
    
    print("\n🎉 Awesome! All local test environments passed assertions successfully.")

if __name__ == '__main__':
    run_suite()
