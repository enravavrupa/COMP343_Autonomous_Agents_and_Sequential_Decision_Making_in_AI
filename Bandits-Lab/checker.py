import unittest
import numpy as np
import warnings

from environment import WarehouseEnvironment
from agents import EpsilonGreedyAgent, UCBAgent


class TestWarehouseBanditSystem(unittest.TestCase):

    def test_epsilon_greedy_exploration_exploitation(self):
        """BUG FIX 1: Verifies epsilon logic explores/exploits properly."""
        agent_exploit = EpsilonGreedyAgent(n_arms=4, epsilon=0.0)
        agent_exploit.Q = np.array([0.1, 0.2, 0.9, 0.4])  # Zone 3 (index 2) is optimal
        self.assertEqual(
            agent_exploit.select_action(), 
            2, 
            "Epsilon Greedy failed: When epsilon=0, action must be argmax(Q)."
        )

    def test_epsilon_greedy_update_denominator(self):
        """BUG FIX 2: Verifies incremental update uses N[action] instead of total t."""
        agent = EpsilonGreedyAgent(n_arms=4, epsilon=0.1)
        agent.update(action=0, reward=1.0)  # N[0]=1
        agent.update(action=1, reward=0.0)  # N[1]=1, t=2
        agent.update(action=0, reward=0.0)  # N[0]=2, t=3

        # Q[0] should be 0.5. If dividing by t (3), Q[0] becomes ~0.6667
        self.assertAlmostEqual(
            agent.Q[0], 0.5, places=4,
            msg="Epsilon Greedy failed: Q update must divide by N[action], not total time t."
        )

    def test_ucb_unvisited_arms_prioritized(self):
        """BUG FIX 3: Ensures zero division is avoided and unvisited arms are selected."""
        agent = UCBAgent(n_arms=4, c=1.0)
        
        # Turn warnings into errors to catch division by zero on N=0
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                action = agent.select_action()
            except RuntimeWarning:
                self.fail("UCB Agent failed: Division by zero encountered when arms have N=0.")

        visited = []
        for _ in range(4):
            a = agent.select_action()
            visited.append(a)
            agent.update(a, reward=1.0)

        self.assertEqual(
            sorted(visited), [0, 1, 2, 3], 
            "UCB Agent failed: Must pull all arms at least once before applying UCB formula."
        )

    def test_ucb_square_root_formula(self):
        """BUG FIX 4: Validates that the square root is applied to the exploration bonus."""
        agent = UCBAgent(n_arms=2, c=2.0)
        agent.N = np.array([1.0, 4.0])
        agent.Q = np.array([0.5, 0.8])
        # Total t = sum(N) + 1 = 6.0
        # Correct bonus with sqrt: c * sqrt(ln(6) / N)
        # Arm 0: 0.5 + 2.0 * sqrt(1.79176 / 1) = 0.5 + 2.677 = 3.177
        # Arm 1: 0.8 + 2.0 * sqrt(1.79176 / 4) = 0.8 + 1.338 = 2.138
        
        # Without sqrt:
        # Arm 0: 0.5 + 2.0 * (1.79176 / 1) = 4.083
        # Arm 1: 0.8 + 2.0 * (1.79176 / 4) = 1.695
        
        # We test explicit UCB values on an arm where lack of sqrt changes choice dynamics
        agent.N = np.array([10.0, 100.0])
        agent.Q = np.array([1.0, 1.2])
        # t = 111. ln(111) = 4.7095
        # Correct (with sqrt):
        # Arm 0: 1.0 + 1.0 * sqrt(4.7095 / 10)  = 1.0 + 0.686 = 1.686
        # Arm 1: 1.2 + 1.0 * sqrt(4.7095 / 100) = 1.2 + 0.217 = 1.417 -> Picks Arm 0
        # Incorrect (without sqrt):
        # Arm 0: 1.0 + 1.0 * (4.7095 / 10)  = 1.0 + 0.4709 = 1.4709
        # Arm 1: 1.2 + 1.0 * (4.7095 / 100) = 1.2 + 0.0470 = 1.2470 -> Picks Arm 0
        
        # Direct check on bonus computation:
        t = np.sum(agent.N) + 1
        expected_bonus_0 = agent.c * np.sqrt(np.log(t) / agent.N[0])
        
        # Force a state where missing sqrt selects the WRONG action:
        agent.c = 1.0
        agent.N = np.array([2.0, 10.0])
        agent.Q = np.array([0.1, 0.4])
        # t = 13, ln(13) = 2.5649
        # WITH SQRT:   Arm 0 = 0.1 + sqrt(1.282) = 1.232 | Arm 1 = 0.4 + sqrt(0.256) = 0.906 -> Chooses 0
        # WITHOUT SQRT: Arm 0 = 0.1 + 1.282 = 1.382      | Arm 1 = 0.4 + 0.256 = 0.656

        # Target test case where action flips:
        agent.N = np.array([5.0, 20.0])
        agent.Q = np.array([0.5, 0.7])
        # t = 26, ln(26) = 3.258
        # WITH SQRT:   Arm 0 = 0.5 + sqrt(0.6516) = 1.307 | Arm 1 = 0.7 + sqrt(0.1629) = 1.103 -> Choice: 0
        # WITHOUT SQRT: Arm 0 = 0.5 + 0.6516 = 1.1516     | Arm 1 = 0.7 + 0.1629 = 0.8629 -> Choice: 0
        
        # Test case specifically sensitivity-tuned to break if sqrt is missing:
        agent.c = 0.3
        agent.N = np.array([2.0, 8.0])
        agent.Q = np.array([0.3, 0.5])
        # t = 11, ln(11) = 2.397
        # WITH SQRT:   Arm 0 = 0.3 + 0.3*sqrt(1.198) = 0.628 | Arm 1 = 0.5 + 0.3*sqrt(0.299) = 0.664 -> Chooses 1
        # WITHOUT SQRT: Arm 0 = 0.3 + 0.3*(1.198) = 0.659    | Arm 1 = 0.5 + 0.3*(0.299) = 0.589 -> Chooses 0
        
        selected = agent.select_action()
        self.assertEqual(
            selected, 1, 
            "UCB Agent failed: UCB exploration bonus is missing np.sqrt()."
        )


if __name__ == "__main__":
    unittest.main()
