from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt


def data_simulation(times, d, actions):
    context = {}
    desired_action = np.zeros(shape=(times, 1))
    n_actions = len(actions)
    for t in range(times):
        context[t] = np.random.uniform(0, 1, (n_actions, d))
        desired_action[t] = actions[np.argmax(np.sum(context[t], axis=1))]
    return context, desired_action


def data_simulation2(times, d, actions):
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


def expert_training(history_context, history_action):
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB(), )
    logreg.fit(history_context, history_action)
    mnb.fit(history_context, history_action)
    return [logreg, mnb]


def tuning_plot(tunning_region, ctr_tunning, label):
    plt.plot(tunning_region, ctr_tunning, 'ro-', label=label)
    plt.xlabel('parameter value')
    plt.ylabel('CTR')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Parameter Tunning Curve")
    plt.show()


def regret_plot(times, regret, label):
    plt.plot(range(times), regret, 'r-', label=label)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Regret Bound with respect to T")
    plt.show()
