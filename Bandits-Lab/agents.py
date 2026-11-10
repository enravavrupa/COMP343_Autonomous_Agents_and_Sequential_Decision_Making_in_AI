import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1, decay_type=None, decay_rate=0.01, min_epsilon=0.01):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.decay_type = decay_type  # Options: None, 'linear', 'exponential'
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms)
        self.t = 0
        
    def select_action(self):
        # TODO: Implement epsilon-greedy action selection # done
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.Q)
            
    def update(self, action, reward):
        self.N[action] += 1
        self.t += 1
        
        # TODO: Implement sample-average update rule for Q-values # donee
        self.Q[action] = self.Q[action] + (1.0 / self.N[action]) * (reward - self.Q[action])
        
        # TODO: Implement epsilon decay schedule if decay_type is specified # done? 
        # there are no associated failure so can't tell # done
        if self.decay_type == 'linear':
            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)
        elif self.decay_type == 'exponential':
            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)


class UCBAgent:
    def __init__(self, n_arms, c=1.0):
        self.n_arms = n_arms
        self.c = c
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms)
        
    def select_action(self):
        t = np.sum(self.N) + 1
        
        # TODO: Implement UCB action selection formula # done
        if np.any(self.N == 0): 
            return np.argmin(self.N) # code is throwing errors when I try using only self.N == 0 for vagueness
        # and suggesting use of any() or all() so using the first # using argmin the index for the 0
        # .all() is also working # for loop not working
        
        bonus = self.c * np.sqrt(np.log(t) / self.N)
        ucb_values = self.Q + bonus
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (1.0 / self.N[action]) * (reward - self.Q[action])
