import pytest
import numpy as np
from hmm_environment import WarehouseHMMEnvironment
from hmm_filter import HMMStateEstimator

# A tiny 5x3 map. Walkable spaces are only at (1,1), (1,2), and (1,3)
# (1,1) and (1,3) are dead-ends (3 surrounding walls). (1,2) is a hallway (2 surrounding walls).
MAP_TINY = """
#####
#...#
#####
"""

def test_initial_belief_is_uniform():
    """Verifies the robot starts with complete uncertainty across valid spaces."""
    env = WarehouseHMMEnvironment(MAP_TINY)
    filter_agent = HMMStateEstimator(env.num_states, env.T, env.E)
    
    # There are exactly 3 walkable cells.
    valid_states = np.sum(filter_agent.belief_state > 0)
    assert valid_states == 3, "Prior should only distribute probability to walkable cells, not walls."
    
    # The sum of all probabilities must exactly equal 1.0
    assert np.isclose(np.sum(filter_agent.belief_state), 1.0), "Initial belief state does not sum to 1.0!"

def test_predict_update_normalization():
    """Verifies that after one step of matrix math, probabilities still sum to 1.0."""
    env = WarehouseHMMEnvironment(MAP_TINY)
    filter_agent = HMMStateEstimator(env.num_states, env.T, env.E)
    
    # Take 1 arbitrary action (East = 2), observe 2 walls
    belief = filter_agent.bayesian_filter_step(action=2, observation=2)
    
    assert np.isclose(np.sum(belief), 1.0), "Your Update step failed to normalize the vector (Z constant)!"
    assert np.all(belief >= 0.0), "Probabilities cannot be negative!"

def test_robot_instant_localization():
    """Tests if the robot can instantly find itself with perfect sensors."""
    # We set sensor accuracy to 1.0 (100% perfect) to eliminate noise
    env = WarehouseHMMEnvironment(MAP_TINY, sensor_accuracy=1.0)
    filter_agent = HMMStateEstimator(env.num_states, env.T, env.E)
    
    # Action East (2). The robot perfectly senses 3 walls (which only exist at dead ends).
    # Because it moved East, the physics model should strongly predict the Eastern dead end.
    filter_agent.bayesian_filter_step(action=2, observation=3)
    
    most_likely_state = filter_agent.get_most_likely_state()
    expected_eastern_dead_end = (1, 3) 
    
    # The robot should have 100% certainty it is at (1, 3)
    assert env.state_to_coord(most_likely_state) == expected_eastern_dead_end, "Robot failed to localize at the dead end!"
