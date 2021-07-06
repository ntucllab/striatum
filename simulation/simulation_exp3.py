import matplotlib.pyplot as plt
import numpy as np

from buzzni.ai.reco.mab.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
)
from buzzni.ai.reco.mab.bandit import Exp3
from buzzni.ai.reco.mab import simulation


def main():
    n_rounds = 1000
    context_dimension = 5
    action_storage = MemoryActionStorage()
    action_storage.add([Action(i) for i in range(5)])
    random_state = np.random.RandomState(0)

    # Parameter tuning
    tuning_region = np.arange(0.001, 1, 0.03)
    ctr_tuning = np.zeros(shape=len(tuning_region))
    context1, desired_actions1 = simulation.simulate_data(
        n_rounds, context_dimension, action_storage, random_state=0)
    for gamma_i, gamma in enumerate(tuning_region):
        policy = Exp3(MemoryHistoryStorage(), MemoryModelStorage(),
                      action_storage, gamma=gamma, random_state=random_state)
        cum_regret = simulation.evaluate_policy(policy, context1,
                                                desired_actions1)
        ctr_tuning[gamma_i] = n_rounds - cum_regret[-1]
    ctr_tuning /= n_rounds
    gamma_opt = tuning_region[np.argmax(ctr_tuning)]
    simulation.plot_tuning_curve(tuning_region, ctr_tuning,
                                 label="gamma changes")

    # Regret Analysis
    n_rounds = 10000
    context2, desired_actions2 = simulation.simulate_data(
        n_rounds, context_dimension, action_storage, random_state=1)
    policy = Exp3(MemoryHistoryStorage(), MemoryModelStorage(),
                  action_storage, gamma=gamma_opt, random_state=random_state)

    for t in range(n_rounds):
        history_id, recommendation = policy.get_action(context2[t])
        action_id = recommendation.action.id
        if desired_actions2[t] != action_id:
            policy.reward(history_id, {action_id: 0})
        else:
            policy.reward(history_id, {action_id: 1})

    policy.plot_avg_regret()
    plt.show()


if __name__ == '__main__':
    main()
