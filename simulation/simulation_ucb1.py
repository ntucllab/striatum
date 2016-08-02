from striatum.storage import history
from striatum.storage import model
from striatum.bandit import ucb1
import simulation as sm
import numpy as np
import matplotlib.pyplot as plt


def main():
    times = 1000
    d = 5
    actions = [1, 2, 3, 4, 5]

    # Regret Analysis
    times = 20000
    context, desired_action = sm.data_simulation(times, d, actions)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = ucb1.UCB1(actions, historystorage, modelstorage)

    seq_error = sm.policy_evaluation(policy, context, desired_action)
    seq_error = [x / y for x, y in zip(seq_error, range(1, times + 1))]

    # Plot the regret analysis
    plt.plot(range(times), seq_error, 'r-')
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Regret Bound with respect to T - UCB1")
    plt.show()


if __name__ == '__main__':
    main()
