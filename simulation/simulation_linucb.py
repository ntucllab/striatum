from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linucb
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
    tuning_region = np.arange(0, 3, 0.05)
    ctr_tuning = np.zeros(shape=(len(tuning_region), 1))
    context1, desired_action1 = simulation.simulate_data(times, d, actions)
    i = 0
    for alpha in tuning_region:
        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = linucb.LinUCB(actions, historystorage, modelstorage, alpha=alpha, d=d)
        seq_error = simulation.evaluate_policy(policy, context1, desired_action1)
        ctr_tuning[i] = times - seq_error[-1]
        i += 1
    ctr_tuning /= times
    alpha_opt = tuning_region[np.argmax(ctr_tuning)]
    simulation.plot_tuning_curve(tuning_region, ctr_tuning, label="alpha changes")

    # Regret Analysis
    times = 10000
    context2, desired_action2 = simulation.simulate_data(times, d, actions)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = linucb.LinUCB(actions, historystorage, modelstorage, alpha=alpha_opt, d=d)

    for t in range(times):
        history_id, action = policy.get_action(context2[t], 1)
        if desired_action2[t][0] != action[0]['action'].action_id:
            policy.reward(history_id, {action[0]['action'].action_id: 0})
        else:
            policy.reward(history_id, {action[0]['action'].action_id: 1})

    rplt.plot_avg_regret(policy, history_id)


if __name__ == '__main__':
    main()
