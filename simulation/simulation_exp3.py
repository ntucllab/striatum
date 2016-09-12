import matplotlib.pyplot as plt

from striatum.storage import history
from striatum.storage import model
from striatum.bandit import Exp3
from striatum.bandit.bandit import Action
from striatum import simulation
import numpy as np


def main():
    n_rounds = 1000
    context_dimension = 5
    actions = [Action(action_id) for action_id in range(1, 6)]

    # Parameter tuning
    tuning_region = np.arange(0.001, 1, 0.03)
    ctr_tuning = np.zeros(shape=len(tuning_region))
    context1, desired_actions1 = simulation.simulate_data(
        n_rounds, context_dimension, actions, random_state=0)
    for gamma_i, gamma in enumerate(tuning_region):
        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = Exp3(actions, historystorage, modelstorage, gamma)
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
        n_rounds, context_dimension, actions, random_state=1)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = Exp3(actions, historystorage, modelstorage, gamma=gamma_opt)

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
