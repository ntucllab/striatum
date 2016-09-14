from six.moves import range
import matplotlib.pyplot as plt

from striatum.storage import MemoryHistoryStorage, MemoryModelStorage
from striatum.bandit import ucb1
from striatum import simulation
from striatum.bandit.bandit import Action

def simulate_bandit():
    context_dimension = 5
    actions = [Action(action_id) for action_id in range(1, 6)]

    # Regret Analysis
    n_rounds = 40000
    context, desired_actions = simulation.simulate_data(
        n_rounds, context_dimension, actions, random_state=0)
    historystorage = MemoryHistoryStorage()
    modelstorage = MemoryModelStorage()
    policy = ucb1.UCB1(actions, historystorage, modelstorage)

    for t in range(n_rounds):
        history_id, action = policy.get_action(context[t], 1)
        action_id = action[0]['action'].action_id
        if desired_actions[t] != action_id:
            policy.reward(history_id, {action_id: 0})
        else:
            policy.reward(history_id, {action_id: 1})

    policy.plot_avg_regret()

def main():
    simulate_bandit()
    plt.show()

if __name__ == '__main__':
    main()
