import numpy as np

class WarehouseEnvironment:
    """
    Simulates a 4-zone warehouse environment modeled as a Multi-Armed Bandit.
    Default zone success rates (probabilities): Zone 1 = 0.1, Zone 2 = 0.5, Zone 3 = 0.8, Zone 4 = 0.3
    """
    def __init__(self, zone_rates=None):
        if zone_rates is None:
            self.zone_rates = np.array([0.1, 0.5, 0.8, 0.3])
        else:
            self.zone_rates = np.array(zone_rates)
            
        self.n_arms = len(self.zone_rates)
        self.state = np.zeros(self.n_arms)

    def step(self, action):
        """
        Executes one environment step for the chosen action (zone).
        Returns the updated state and the reward for the selected zone.
        """
        # Generate random event occurrences for all zones based on their probabilities
        events = np.random.rand(self.n_arms) < self.zone_rates
        self.state = events.astype(int)
        
        # The reward is 1 if the selected zone had an active event, else 0
        reward = self.state[action]
        
        return self.state, reward

    def reset(self):
        """Resets the environment state to zero."""
        self.state = np.zeros(self.n_arms)
        return self.state
