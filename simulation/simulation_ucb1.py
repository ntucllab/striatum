from striatum.storage import history
from striatum.storage import model
from striatum.bandit import ucb1
from striatum import simulation
from striatum import rewardplot as rplt
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
    context, desired_action = simulation.simulate_data(times, d, actions)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = ucb1.UCB1(actions, historystorage, modelstorage)

    for t in range(times):
        history_id, action = policy.get_action(context[t], 1)
        if desired_action[t][0] != action[0]['action'].action_id:
            policy.reward(history_id, {action[0]['action'].action_id: 0})
        else:
            policy.reward(history_id, {action[0]['action'].action_id: 1})

    policy.plot_avg_regret()


if __name__ == '__main__':
    main()
