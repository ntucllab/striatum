from six.moves import range
import matplotlib.pyplot as plt

from buzzni.ai.reco.mab.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
)
from buzzni.ai.reco.mab.bandit import UCB1
from buzzni.ai.reco.mab import simulation


def main():
    context_dimension = 5
    action_storage = MemoryActionStorage()
    action_storage.add([Action(i) for i in range(5)])

    # Regret Analysis
    n_rounds = 10000
    context, desired_actions = simulation.simulate_data(
        n_rounds, context_dimension, action_storage, random_state=1)
    policy = UCB1(MemoryHistoryStorage(), MemoryModelStorage(),
                  action_storage)

    for t in range(n_rounds):
        history_id, recommendation = policy.get_action(context[t])
        action_id = recommendation.action.id
        if desired_actions[t] != action_id:
            policy.reward(history_id, {action_id: 0})
        else:
            policy.reward(history_id, {action_id: 1})

    policy.plot_avg_regret()
    plt.show()


if __name__ == '__main__':
    main()
