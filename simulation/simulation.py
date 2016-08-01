import numpy as np


def data_simulation(times, d, actions):
    context = np.random.uniform(0, 1, (times, d))
    desired_action = np.zeros(shape=(times, 1))
    n_actions = len(actions)
    for t in range(times):
        for i in range(n_actions):
            if i * d / n_actions < sum(context[t, :]) <= (i + 1) * d / n_actions:
                desired_action[t] = actions[i]
    return context, desired_action


def policy_evaluation(policy, context, desired_action):
    times = len(desired_action)
    seq_error = np.zeros(shape=(times, 1))
    for t in range(times):
        history_id, action = policy.get_action(context[t])
        if desired_action[t][0] != action:
            policy.reward(history_id, 0)
            if t == 0:
                seq_error[t] = 1.0
            else:
                seq_error[t] = seq_error[t - 1] + 1.0
        else:
            policy.reward(history_id, 1)
            if t > 0:
                seq_error[t] = seq_error[t - 1]
    return seq_error


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def main():
    return 0

if __name__ == '__main__':
    main()
