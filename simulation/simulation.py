from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import ucb1
from striatum.bandit import linucb
from striatum.bandit import linthompsamp
from striatum.bandit import exp3
from striatum.bandit import exp4p
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


def policy_generation(bandit, action_context, actions):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = 0
    if bandit == 'Exp4P':
        models = expert_training(action_context)
        policy = exp4p.Exp4P(actions, historystorage, modelstorage, models, delta=0.5, pmin=None)

    elif bandit == 'LinUCB':
        policy = linucb.LinUCB(actions, historystorage, modelstorage, 0.5, 20)

    elif bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(actions, historystorage, modelstorage,
                                           d=20, delta=0.9, r=0.01, epsilon=0.5)

    elif bandit == 'UCB1':
        policy = ucb1.UCB1(actions, historystorage, modelstorage)

    elif bandit == 'Exp3':
        policy = exp3.Exp3(actions, historystorage, modelstorage, gamma=0.2)

    elif bandit == 'random':
        policy = 0

    return policy


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def parameter_tunning():
    return 0


def expert_training(self):
    history_context, history_action = self.data_simulation()
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB(), )
    logreg.fit(history_context, history_action)
    mnb.fit(history_context, history_action)
    return [logreg, mnb]


def main():
    return 0

if __name__ == '__main__':
    main()
