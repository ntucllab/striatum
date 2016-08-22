from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linthompsamp
from striatum import simulation
from striatum import rewardplot as rplt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from striatum.bandit.bandit import Action


def main():

    times = 1000
    d = 5
    a1 = Action(1)
    a2 = Action(2)
    a3 = Action(3)
    a4 = Action(4)
    a5 = Action(5)
    actions = [a1, a2, a3, a4, a5]

    # Parameter tuning
    tuning_region = np.arange(0.01, 0.99, 0.1)
    ctr_delta = np.zeros(shape=(len(tuning_region), 1))
    ctr_r = np.zeros(shape=(len(tuning_region), 1))
    ctr_epsilon = np.zeros(shape=(len(tuning_region), 1))

    context1, desired_action1 = simulation.simulate_data(times, d, actions)
    i = 0

    for para in tuning_region:
        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = linthompsamp.LinThompSamp(actions, historystorage, modelstorage,
                                           d=d, delta=para, r=0.01, epsilon=0.5)
        cum_regret = simulation.evaluate_policy(policy, context1, desired_action1)
        ctr_delta[i] = times - cum_regret[-1]

        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = linthompsamp.LinThompSamp(actions, historystorage, modelstorage,
                                           d=d, delta=0.5, r=para, epsilon=0.5)

        cum_regret = simulation.evaluate_policy(policy, context1, desired_action1)
        ctr_r[i] = times - cum_regret[-1]

        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = linthompsamp.LinThompSamp(actions, historystorage, modelstorage,
                                           d=d,  delta=0.5, r=0.01, epsilon=para)
        cum_regret = simulation.evaluate_policy(policy, context1, desired_action1)
        ctr_epsilon[i] = times - cum_regret[-1]
        i += 1

    ctr_delta /= times
    ctr_r /= times
    ctr_epsilon /= times

    delta_opt = tuning_region[np.argmax(ctr_delta)]
    r_opt = tuning_region[np.argmax(ctr_r)]
    epsilon_opt = tuning_region[np.argmax(ctr_epsilon)]

    # Plot the parameter tuning result
    plt.plot(np.arange(0.01, 0.99, 0.1), ctr_delta, 'ro-',
             np.arange(0.01, 0.99, 0.1), ctr_r, 'gs-',
             np.arange(0.01, 0.99, 0.1), ctr_epsilon, 'b^-')
    line1 = mlines.Line2D([], [], color='r', marker='o',
                          label="delta changes, R = 0.01, eps = 0.5")
    line2 = mlines.Line2D([], [], color='g', marker='s',
                          label="delta = 0.5, R = changes, eps = 0.5")
    line3 = mlines.Line2D([], [], color='b', marker='^',
                          label="delta = 0.5, R = 0.01, eps = changes")
    plt.xlabel('parameter value')
    plt.ylabel('CTR')
    plt.legend(handles=[line1, line2, line3], loc='upper center', bbox_to_anchor=(0.5, -0.15))
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Parameter Tunning Curve - LinThompSamp")
    plt.show()

    # Regret Analysis
    times = 10000
    context2, desired_action2 = simulation.simulate_data(times, d, actions)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = linthompsamp.LinThompSamp(actions, historystorage, modelstorage,
                                       d=d, delta=delta_opt, r=r_opt, epsilon=epsilon_opt)

    for t in range(times):
        history_id, action = policy.get_action(context2[t], 1)
        if desired_action2[t][0] != action[0]['action'].action_id:
            policy.reward(history_id, {action[0]['action'].action_id: 0})
        else:
            policy.reward(history_id, {action[0]['action'].action_id: 1})

    rplt.plot_avg_regret(policy, history_id)


if __name__ == '__main__':
    main()
