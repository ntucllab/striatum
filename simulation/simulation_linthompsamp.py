from six.moves import range
import numpy as np
import matplotlib.pyplot as plt

from striatum.storage import MemoryHistoryStorage, MemoryModelStorage
from striatum.bandit import LinThompSamp
from striatum.bandit.bandit import Action
from striatum import simulation


def main():
    n_rounds = 1000
    context_dimension = 5
    actions = [Action(action_id) for action_id in range(1, 6)]
    random_state = np.random.RandomState(0)

    # Parameter tuning
    tuning_region = np.arange(0.01, 0.99, 0.1)
    ctr_delta = np.zeros(shape=len(tuning_region))
    ctr_r = np.zeros(shape=len(tuning_region))
    ctr_epsilon = np.zeros(shape=len(tuning_region))

    context1, desired_actions1 = simulation.simulate_data(
        n_rounds, context_dimension, actions, random_state=random_state)

    for param_i, param in enumerate(tuning_region):
        policy = LinThompSamp(actions,
                              MemoryHistoryStorage(), MemoryModelStorage(),
                              context_dimension=context_dimension,
                              delta=param, r=0.01, epsilon=0.5,
                              random_state=random_state)
        cum_regret = simulation.evaluate_policy(policy, context1,
                                                desired_actions1)
        ctr_delta[param_i] = n_rounds - cum_regret[-1]

        policy = LinThompSamp(actions,
                              MemoryHistoryStorage(), MemoryModelStorage(),
                              context_dimension=context_dimension,
                              delta=0.5, r=param, epsilon=0.5,
                              random_state=random_state)

        cum_regret = simulation.evaluate_policy(policy, context1,
                                                desired_actions1)
        ctr_r[param_i] = n_rounds - cum_regret[-1]

        policy = LinThompSamp(actions,
                              MemoryHistoryStorage(), MemoryModelStorage(),
                              context_dimension=context_dimension,
                              delta=0.5, r=0.01, epsilon=param,
                              random_state=random_state)
        cum_regret = simulation.evaluate_policy(policy, context1,
                                                desired_actions1)
        ctr_epsilon[param_i] = n_rounds - cum_regret[-1]

    ctr_delta /= n_rounds
    ctr_r /= n_rounds
    ctr_epsilon /= n_rounds

    delta_opt = tuning_region[np.argmax(ctr_delta)]
    r_opt = tuning_region[np.argmax(ctr_r)]
    epsilon_opt = tuning_region[np.argmax(ctr_epsilon)]

    # Plot the parameter tuning result
    plt.plot(np.arange(0.01, 0.99, 0.1), ctr_delta, 'ro-',
             label="delta changes, R = 0.01, eps = 0.5")
    plt.plot(np.arange(0.01, 0.99, 0.1), ctr_r, 'gs-',
             label="delta = 0.5, R = changes, eps = 0.5")
    plt.plot(np.arange(0.01, 0.99, 0.1), ctr_epsilon, 'b^-',
             label="delta = 0.5, R = 0.01, eps = changes")
    plt.xlabel('parameter value')
    plt.ylabel('CTR')
    plt.legend(bbox_to_anchor=(1., 0.7))
    plt.ylim([0, 1])
    plt.title("Parameter Tunning Curve - LinThompSamp")
    plt.show()

    # Regret Analysis
    n_rounds = 10000
    context2, desired_actions2 = simulation.simulate_data(
        n_rounds, context_dimension, actions, random_state=random_state)
    policy = LinThompSamp(actions,
                          MemoryHistoryStorage(), MemoryModelStorage(),
                          context_dimension=context_dimension,
                          delta=delta_opt, r=r_opt, epsilon=epsilon_opt,
                          random_state=random_state)

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
