import numpy as np
import matplotlib.pyplot as plt

from striatum import simulation
from striatum.storage import MemoryHistoryStorage, MemoryModelStorage
from striatum.bandit import linucb
from striatum.bandit.bandit import Action


def main():
    n_rounds = 1000
    n_dimensions = 5
    actions = [Action(i) for i in range(5)]

    # Parameter tuning
    tuning_region = np.arange(0, 3, 0.05)
    ctr_tuning = np.empty(shape=len(tuning_region))
    context1, desired_actions1 = simulation.simulate_data(n_rounds, n_dimensions, actions)
    for alpha_i, alpha in enumerate(tuning_region):
        policy = linucb.LinUCB(actions,
                               historystorage=MemoryHistoryStorage(),
                               modelstorage=MemoryModelStorage(),
                               alpha=alpha, n_dimensions=n_dimensions)
        cum_regret = simulation.evaluate_policy(policy, context1, desired_actions1)
        ctr_tuning[alpha_i] = n_rounds - cum_regret[-1]
    ctr_tuning /= n_rounds
    alpha_opt = tuning_region[np.argmax(ctr_tuning)]
    simulation.plot_tuning_curve(tuning_region, ctr_tuning, label="alpha changes")

    # Regret Analysis
    n_rounds = 10000
    context2, desired_actions2 = simulation.simulate_data(n_rounds, n_dimensions, actions)
    policy = linucb.LinUCB(actions,
                           historystorage=MemoryHistoryStorage(),
                           modelstorage=MemoryModelStorage(),
                           alpha=alpha_opt, n_dimensions=n_dimensions)

    for t in range(n_rounds):
        history_id, action = policy.get_action(context2[t], 1)
        action_id = action[0]['action'].action_id
        if desired_actions2[t] != action_id:
            policy.reward(history_id, {action_id: 0})
        else:
            policy.reward(history_id, {action_id: 1})

    policy.plot_avg_regret()
    plt.show()


if __name__ == '__main__':
    main()
