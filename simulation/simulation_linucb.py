from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linucb
from striatum import simulation
import numpy as np
from striatum.bandit.bandit import Action


def main():
    times = 1000
    d = 5
    a1 = Action(1, 'a1', 'content 1')
    a2 = Action(2, 'a2', 'content 2')
    a3 = Action(3, 'a3', 'content 3')
    a4 = Action(4, 'a4', 'content 4')
    a5 = Action(5, 'a5', 'content 5')
    actions = [a1, a2, a3, a4, a5]

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
