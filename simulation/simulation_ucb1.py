from striatum.storage import history
from striatum.storage import model
from striatum.bandit import ucb1
from striatum import simulation
import matplotlib.pyplot as plt
from striatum.bandit.bandit import Action


def main():
    d = 5
    a1 = Action(1)
    a2 = Action(2)
    a3 = Action(3)
    a4 = Action(4)
    a5 = Action(5)
    actions = [a1, a2, a3, a4, a5]

    # Regret Analysis
    times = 40000
    context, desired_action = simulation.data_simulation(times, d, actions)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = ucb1.UCB1(actions, historystorage, modelstorage)

    seq_error = simulation.policy_evaluation(policy, context, desired_action)
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
