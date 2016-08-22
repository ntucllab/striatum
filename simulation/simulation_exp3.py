from striatum.storage import history
from striatum.storage import model
from striatum.bandit import exp3
from striatum import simulation
from striatum import rewardplot as rplt
import numpy as np
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
    tuning_region = np.arange(0.001, 1, 0.03)
    ctr_tuning = np.zeros(shape=(len(tuning_region), 1))
    context1, desired_action1 = simulation.simulate_data(times, d, actions)
    i = 0
    for gamma in tuning_region:
        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = exp3.Exp3(actions, historystorage, modelstorage, gamma)
        cum_regret = simulation.evaluate_policy(policy, context1, desired_action1)
        ctr_tuning[i] = times - cum_regret[-1]
        i += 1
    ctr_tuning /= times
    gamma_opt = tuning_region[np.argmax(ctr_tuning)]
    simulation.plot_tuning_curve(tuning_region, ctr_tuning, label="gamma changes")

    # Regret Analysis
    times = 10000
    context2, desired_action2 = simulation.simulate_data(times, d, actions)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = exp3.Exp3(actions, historystorage, modelstorage, gamma=gamma_opt)

    for t in range(times):
        history_id, action = policy.get_action(context2[t], 1)
        if desired_action2[t][0] != action[0]['action'].action_id:
            policy.reward(history_id, {action[0]['action'].action_id: 0})
        else:
            policy.reward(history_id, {action[0]['action'].action_id: 1})

    rplt.plot_avg_regret(policy, history_id)


if __name__ == '__main__':
    main()
