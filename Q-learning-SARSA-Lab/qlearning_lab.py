"""
qlearning_lab.py
==================
YOUR TASK: implement Q-learning control on the PackBot environment.

Run test_correctness.py first -- it checks qlearning_update() against a
hand-computed example before you spend any time on the full training loop.

Do not import anything from qlearning_solution.py.
"""

# Note: Marcus Koseck, “Q Learning simply explained | SARSA and Q-Learning Explanation,”  https://www.youtube.com/watch?v=MI8ByADMh20 
# was referred for revising and more deeply understanding these algorithms

# Note: Priyadarshini Tamilselvan, “Q Learning vs SARSA,”  https://medium.com/@priya61197/q-learning-vs-sarsa-b9e433dec930
# was also referred for revising and more deeply understanding these algorithms

import random
import numpy as np
from packbot_env import PackBotEnv, state_index
from rl_utils import epsilon_greedy, linear_epsilon_schedule, init_Q, wear_threshold



def qlearning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    TODO: implement the Q-learning update rule.

    Inputs:
        Q       : the (N_STATES, N_ACTIONS) value table
        s, a    : the state index and action just taken
        r       : the reward received
        s_next  : the resulting state index, or None if the episode ended
        alpha, gamma : learning rate and discount factor

    Returns:
        The NEW value that Q[s, a] should be updated to. Do not mutate Q
        inside this function -- just return the new scalar.

    Reminder from lecture: Q-learning bootstraps using max_a' Q[s_next, a'] --
    the BEST available action at s_next -- regardless of which action your
    behavior policy would actually take there. Notice there is no a_next
    parameter at all: Q-learning never needs to know what your ε-greedy
    policy actually sampled next. That is the entire difference between
    this file and sarsa_lab.py.

    Special case: if s_next is None (terminal transition), there is
    nothing to bootstrap from -- the target is just r.
    """
    # TODO: replace this line
    if s_next == None:
        new_q_val = Q[s,a] + alpha * (r - Q[s,a])
    else:
        new_q_val = Q[s,a] + alpha * ((r + (gamma * np.max(Q[s_next, :]))) - Q[s,a])
    return new_q_val
    #raise NotImplementedError("Implement qlearning_update")


def train_qlearning(num_episodes=4000, alpha=0.1, gamma=0.95, seed=0,
                     eps_start=1.0, eps_end=0.2, log_every=50):
    """
    TODO: implement the Q-learning training loop using qlearning_update() above.

    Structure to follow (the standard Q-learning control loop from lecture):

        Initialize S
        Repeat until S is terminal:
            Choose A from S using epsilon-greedy(Q)
            Take action A, observe R, S'
            Q(S, A) <- qlearning_update(Q, S, A, R, S' if S' not terminal else None, alpha, gamma)
            S <- S'

    Return (Q, episode_returns, threshold_log) to match qlearning_solution.py's
    signature -- compare_policies.py and test scripts expect this shape.
    threshold_log should be a list of (episode, {item_type: wear_threshold(...)})
    tuples, logged every `log_every` episodes.
    """
    env = PackBotEnv(seed=seed)
    rng = random.Random(seed)
    Q = init_Q()

    episode_returns = []
    threshold_log = []

    # TODO: implement the training loop described above.
    S = env.reset()
    acc = 0
    for i in range(0, num_episodes):
        epsilon = linear_epsilon_schedule(i, num_episodes, eps_start, eps_end)
        terminal = False
        while terminal == False:
            A = epsilon_greedy(Q, S, epsilon, rng)
            S_next, R, terminal, info = env.step(A)         
            Q[S, A] = qlearning_update(Q, S, A, R, S_next if terminal == False else None, alpha, gamma)
            S = S_next
            acc = acc + R
        episode_returns.append(acc) # add new reward to the G_t funciton
        if (i + 1) % log_every == 0:
            for j in [0,1,2]:
                threshold_log.append((i + 1, {i: wear_threshold(Q, j)}))
    #raise NotImplementedError("Implement train_qlearning")

    return Q, episode_returns, threshold_log


if __name__ == "__main__":
    Q, returns, thresholds = train_qlearning()
    print("Final wear thresholds by item type:", thresholds[-1][1])
    print("Mean return, last 200 episodes:", np.mean(returns[-200:]))
