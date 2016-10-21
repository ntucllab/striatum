from six.moves import range
import matplotlib.pyplot as plt

from striatum.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
)
from striatum.bandit import UCB1
from striatum import simulation

def simulate_bandit():
    context_dimension = 5
    action_storage = MemoryActionStorage()
    action_storage.add([Action(i) for i in range(5)])

    # Regret Analysis
    n_rounds = 40000
    context, desired_actions = simulation.simulate_data(
        n_rounds, context_dimension, action_storage, random_state=0)
    policy = UCB1(MemoryHistoryStorage(), MemoryModelStorage(),
                  action_storage)

    for t in range(n_rounds):
        history_id, action = policy.get_action(context[t], 1)
        action_id = action[0]['action'].id
        if desired_actions[t] != action_id:
            policy.reward(history_id, {action_id: 0})
        else:
            policy.reward(history_id, {action_id: 1})

    policy.plot_avg_regret()

def main():
    simulate_bandit()
    plt.show()

if __name__ == '__main__':
    main()
