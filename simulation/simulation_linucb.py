from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linucb
from striatum import simulation
import numpy as np


def main():
    times = 1000
    d = 5
    actions = [1, 2, 3, 4, 5]

    # Parameter tunning
    tunning_region = np.arange(0, 3, 0.05)
    ctr_tunning = np.zeros(shape=(len(tunning_region), 1))
    context1, desired_action1 = simulation.data_simulation(times, d, actions)
    i = 0
    for alpha in tunning_region:
        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = linucb.LinUCB(actions, historystorage, modelstorage, alpha=alpha, d=d)
        seq_error = simulation.policy_evaluation(policy, context1, desired_action1)
        ctr_tunning[i] = times - seq_error[-1]
        i += 1
    ctr_tunning /= times
    alpha_opt = tunning_region[np.argmax(ctr_tunning)]
    simulation.tuning_plot(tunning_region, ctr_tunning, label="alpha changes")

    # Regret Analysis
    times = 10000
    context2, desired_action2 = simulation.data_simulation(times, d, actions)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = linucb.LinUCB(actions, historystorage, modelstorage, alpha=alpha_opt, d=d)
    regret = simulation.regret_calculation(simulation.policy_evaluation(policy, context2, desired_action2))
    simulation.regret_plot(times, regret, label='delta = ' + str(alpha_opt))


if __name__ == '__main__':
    main()
