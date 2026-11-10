"""
sarsa_lab.py
=============
YOUR TASK: implement SARSA control on the PackBot environment.

Run test_correctness.py first -- it checks sarsa_update() against a
hand-computed example before you spend any time on the full training loop.

Do not import anything from sarsa_solution.py.
"""

# Note: Marcus Koseck, “Q Learning simply explained | SARSA and Q-Learning Explanation,”  https://www.youtube.com/watch?v=MI8ByADMh20 
# was referred for revising and more deeply understanding these algorithms

# Note: Priyadarshini Tamilselvan, “Q Learning vs SARSA,”  https://medium.com/@priya61197/q-learning-vs-sarsa-b9e433dec930
# was also referred for revising and more deeply understanding these algorithms

import random
import numpy as np
from packbot_env import PackBotEnv, state_index
from rl_utils import epsilon_greedy, linear_epsilon_schedule, init_Q, wear_threshold


def sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma):
    """
    TODO: implement the SARSA update rule.

    Inputs:
        Q       : the (N_STATES, N_ACTIONS) value table
        s, a    : the state index and action just taken
        r       : the reward received
        s_next  : the resulting state index, or None if the episode ended
        a_next  : the SAMPLED next action (from the same behavior policy
                  you used to pick `a`), or None if the episode ended
        alpha, gamma : learning rate and discount factor

    Returns:
        The NEW value that Q[s, a] should be updated to. Do not mutate Q
        inside this function -- just return the new scalar.

    Reminder from lecture: SARSA bootstraps using the VALUE OF THE ACTION
    THAT WAS ACTUALLY SAMPLED next (a_next) -- not the best available
    action at s_next. That is the entire difference between this file
    and qlearning_lab.py.

    Special case: if s_next is None (terminal transition), there is
    nothing to bootstrap from -- the target is just r.
    """
    # TODO: replace this line
    if s_next == None:
        new_q_val = Q[s,a] + alpha * (r - Q[s,a])
    else:
        new_q_val = Q[s,a] + alpha * ((r + (gamma * Q[s_next, a_next])) - Q[s,a])
    return new_q_val
    #raise NotImplementedError("Implement sarsa_update")


def train_sarsa(num_episodes=4000, alpha=0.1, gamma=0.95, seed=0,
                 eps_start=1.0, eps_end=0.2, log_every=50):
    """
    TODO: implement the SARSA training loop using sarsa_update() above.

    Structure to follow (this is the standard SARSA control loop from
    lecture -- note that, unlike Q-learning, you pick action A' for the
    NEXT state *before* you loop back around, because you need it both
    to act next step AND to bootstrap this step's update):

        Initialize S
        Choose A from S using epsilon-greedy(Q)
        Repeat until S is terminal:
            Take action A, observe R, S'
            If S' is not terminal:
                Choose A' from S' using epsilon-greedy(Q)
                Q(S, A) <- sarsa_update(Q, S, A, R, S', A', alpha, gamma)
            Else:
                A' <- None
                Q(S, A) <- sarsa_update(Q, S, A, R, None, None, alpha, gamma)
            S, A <- S', A'

    Return (Q, episode_returns, threshold_log) to match sarsa_solution.py's
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
    
    for i in range(0, num_episodes):
        epsilon = linear_epsilon_schedule(i, num_episodes, eps_start, eps_end)
        A = epsilon_greedy(Q, S, epsilon, rng)
        terminal = False
        acc = 0
        while terminal == False:
            S_next, R, terminal, info = env.step(A)     
            A_next = epsilon_greedy(Q, S, epsilon, rng)
            Q[S, A] = sarsa_update(Q, S, A, R, S_next, A_next, alpha, gamma)
            S = S_next
            A = A_next
            acc = acc + R
            if terminal == True:
                Q[S, A] = sarsa_update(Q, S, A, R, None, None, alpha, gamma)
                
        episode_returns.append(acc) # add new reward to the G_t funciton
        if (i + 1) % log_every == 0:
            for j in [0,1,2]:
                threshold_log.append((i + 1, {i: wear_threshold(Q, j)}))
    
    # raise NotImplementedError("Implement train_sarsa")

    return Q, episode_returns, threshold_log


if __name__ == "__main__":
    Q, returns, thresholds = train_sarsa()
    print("Final wear thresholds by item type:", thresholds[-1][1])
    print("Mean return, last 200 episodes:", np.mean(returns[-200:]))
