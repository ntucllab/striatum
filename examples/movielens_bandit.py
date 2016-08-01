"""
The script uses real-world data to conduct contextual bandit experiments. Here we use
MovieLens 10M Dataset, which is released by GroupLens at 1/2009. Please fist pre-process
datasets (use "movielens_preprocess.py"), and then you can run this example.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import ucb1
from striatum.bandit import linucb
from striatum.bandit import linthompsamp
from striatum.bandit import exp4p
from striatum.bandit import exp3
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def get_data():
    streaming_batch = pd.read_csv('streaming_batch.csv', sep='\t', names=['user_id'], engine='c')
    user_feature = pd.read_csv('user_feature.csv', sep='\t', header=0, index_col=0, engine='c')
    actions = list(pd.read_csv('actions.csv', sep='\t', header=0, engine='c')['movie_id'])
    reward_list = pd.read_csv('reward_list.csv', sep='\t', header=0, engine='c')
    action_context = pd.read_csv('action_context.csv', sep='\t', header=0, engine='c')
    return streaming_batch, user_feature, actions, reward_list, action_context


def expert_training(action_context):
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB(), )
    logreg.fit(action_context.iloc[:, 2:], action_context.iloc[:, 1])
    mnb.fit(action_context.iloc[:, 2:], action_context.iloc[:, 1])
    return [logreg, mnb]


def policy_generation(bandit, action_context, actions):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()

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


def policy_evaluation(policy, bandit, streaming_batch, user_feature, reward_list, actions):
    times = len(streaming_batch)
    seq_error = np.zeros(shape=(times, 1))
    if bandit in ['LinUCB', 'LinThompSamp', 'UCB1', 'Exp3']:

        for t in range(times):
            feature = user_feature[user_feature.index == streaming_batch.iloc[t, 0]]
            full_context = pd.DataFrame(np.repeat(np.array(feature), 50, axis=0)).as_matrix()
            history_id, action = policy.get_action(full_context)
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[t, 0]]

            if action not in list(watched_list['movie_id']):
                policy.reward(history_id, 0)
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                policy.reward(history_id, 1)
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    elif bandit == 'Exp4P':
        for t in range(times):
            feature = user_feature[user_feature.index == streaming_batch.iloc[t, 0]]
            history_id, action = policy.get_action(list(feature.iloc[0]))
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[t, 0]]

            if action not in list(watched_list['movie_id']):
                policy.reward(history_id, 0)
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                policy.reward(history_id, 1)
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    elif bandit == 'random':
        for t in range(times):
            action = actions[np.random.randint(0, len(actions)-1)]
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[t, 0]]

            if action not in list(watched_list['movie_id']):
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    return seq_error


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def main():
    streaming_batch, user_feature, actions, reward_list, action_context = get_data()
    streaming_batch_small = streaming_batch.iloc[0:10000]

    # conduct regret analyses
    experiment_bandit = ['LinUCB', 'LinThompSamp', 'Exp4P', 'UCB1', 'Exp3', 'random']
    regret = {}
    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    i = 0
    for bandit in experiment_bandit:
        policy = policy_generation(bandit, action_context, actions)
        seq_error = policy_evaluation(policy, bandit, streaming_batch_small, user_feature, reward_list, actions)
        regret[bandit] = regret_calculation(seq_error)
        plt.plot(range(len(streaming_batch_small)), regret[bandit], c=col[i], ls='-',
                 marker='.', label=bandit)
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Regret Bound with respect to T")
        i += 1
    plt.show()


if __name__ == '__main__':
    main()
