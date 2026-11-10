from sarsa_lab import train_sarsa
from qlearning_lab import train_qlearning
from compare_policies import query_policy_action

Q_sarsa, _, _ = train_sarsa()
Q_qlearning, _, _ = train_qlearning()

print(query_policy_action(Q_sarsa, item_type=1, wear=8))
print(query_policy_action(Q_qlearning, item_type=1, wear=8))
