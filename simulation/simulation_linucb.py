import numpy as np

from striatum import simulation
from striatum.storage import MemoryHistoryStorage, MemoryModelStorage
from striatum.bandit import linucb
from striatum.bandit.bandit import Action


def main():
    times = 1000
    d = 5
    actions = [Action(i) for i in range(5)]

    # Parameter tuning
    tuning_region = np.arange(0, 3, 0.05)
    ctr_tuning = np.zeros(shape=len(tuning_region))
    context1, desired_actions1 = simulation.simulate_data(times, d, actions)
    i = 0
    for alpha in tuning_region:
        policy = linucb.LinUCB(actions,
                               historystorage=MemoryHistoryStorage(),
                               modelstorage=MemoryModelStorage(),
                               alpha=alpha, d=d)
        cum_regret = simulation.evaluate_policy(policy, context1, desired_actions1)
        ctr_tuning[i] = times - cum_regret[-1]
        i += 1
    ctr_tuning /= times
    alpha_opt = tuning_region[np.argmax(ctr_tuning)]
    simulation.plot_tuning_curve(tuning_region, ctr_tuning, label="alpha changes")

    # Regret Analysis
    times = 10000
    context2, desired_actions2 = simulation.simulate_data(times, d, actions)
    historystorage = MemoryHistoryStorage()
    modelstorage = MemoryModelStorage()
    policy = linucb.LinUCB(actions, historystorage, modelstorage, alpha=alpha_opt, d=d)

    for t in range(times):
        history_id, action = policy.get_action(context2[t], 1)
        if desired_actions2[t][0] != action[0]['action'].action_id:
            policy.reward(history_id, {action[0]['action'].action_id: 0})
        else:
            policy.reward(history_id, {action[0]['action'].action_id: 1})

    policy.plot_avg_regret()



if __name__ == '__main__':
    main()
