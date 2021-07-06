# -*- coding: utf-8 -*-
"""
==============================
Contextual bandit on MovieLens
==============================
The script uses real-world data to conduct contextual bandit experiments. Here we use
MovieLens 10M Dataset, which is released by GroupLens at 1/2009. Please fist pre-process
datasets (use "movielens_preprocess.py"), and then you can run this example.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from buzzni.ai.reco.mab.storage import history
from buzzni.ai.reco.mab.storage import model
from buzzni.ai.reco.mab.storage import action
from buzzni.ai.reco.mab.bandit import ucb1
from buzzni.ai.reco.mab.bandit import linucb
from buzzni.ai.reco.mab.bandit import linthompsamp
from buzzni.ai.reco.mab.bandit import exp4p
from buzzni.ai.reco.mab.bandit import exp3
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def get_data():
    streaming_batch = pd.read_csv('streaming_batch.csv', sep='\t', engine='c')
    user_feature = pd.read_csv('user_feature.csv', sep='\t', header=0, index_col=0, engine='c')
    actions_id = list(pd.read_csv('actions.csv', sep='\t', header=0, engine='c')['movie_id'])
    reward_list = pd.read_csv('reward_list.csv', sep='\t', header=0, engine='c')
    action_context = pd.read_csv('action_context.csv', sep='\t', header=0, engine='c')

    actions = []
    for key in actions_id:
        actions.append(action.Action(key))
    actionstorage = action.MemoryActionStorage()
    actionstorage.add(actions)
    return streaming_batch, user_feature, actionstorage, reward_list, action_context


def train_expert(action_context):
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB(), )
    logreg.fit(action_context.iloc[:, 2:], action_context.iloc[:, 1])
    mnb.fit(action_context.iloc[:, 2:], action_context.iloc[:, 1])
    return [logreg, mnb]


def get_advice(context, actions_id, experts):
    advice = {}
    for time in context.keys():
        advice[time] = {}
        for i in range(len(experts)):
            prob = experts[i].predict_proba(context[time])[0]
            advice[time][i] = {}
            for j in range(len(prob)):
                advice[time][i][actions_id[j]] = prob[j]
    return advice


def policy_generation(bandit, actionstorage):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()

    if bandit == 'Exp4P':
        policy = exp4p.Exp4P(action_storage=actionstorage,
                             history_storage=historystorage,
                             model_storage=modelstorage,
                             delta=0.5,
                             pmin=None)

    elif bandit == 'LinUCB':
        policy = linucb.LinUCB(action_storage=actionstorage,
                               history_storage=historystorage,
                               model_storage=modelstorage,
                               alpha=0.3,
                               context_dimension=18)

    elif bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(action_storage=actionstorage,
                                           history_storage=historystorage,
                                           model_storage=modelstorage,
                                           context_dimension=18,
                                           delta=0.61,
                                           R=0.01,
                                           epsilon=0.71)

    elif bandit == 'UCB1':
        policy = ucb1.UCB1(action_storage=actionstorage,
                           history_storage=historystorage,
                           model_storage=modelstorage)

    elif bandit == 'Exp3':
        policy = exp3.Exp3(action_storage=actionstorage,
                           history_storage=historystorage,
                           model_storage=modelstorage,
                           gamma=0.2)

    elif bandit == 'random':
        policy = 0

    return policy


def policy_evaluation(policy, bandit, streaming_batch, user_feature, reward_list, actionstorage, action_context=None):
    times = len(streaming_batch)
    seq_error = np.zeros(shape=(times, 1))
    actions_id = [action_id for action_id in actionstorage.iterids()]
    if bandit in ['LinUCB', 'LinThompSamp', 'UCB1', 'Exp3']:
        for t in range(times):
            feature = np.array(user_feature[user_feature.index == int(streaming_batch.iloc[t, 0])])[0]
            full_context = {}
            for action_id in actions_id:
                full_context[action_id] = feature
            history_id, action = policy.get_action(full_context, 1)
            watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t, 0])]

            if action[0].action.id not in list(watched_list['movie_id']):
                policy.reward(history_id, {action[0].action.id: 0.0})
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                policy.reward(history_id, {action[0].action.id: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    elif bandit == 'Exp4P':
        for t in range(times):
            feature = user_feature[user_feature.index == int(streaming_batch.iloc[t, 0])]
            experts = train_expert(action_context)
            advice = {}
            for i in range(len(experts)):
                prob = experts[i].predict_proba(feature)[0]
                advice[i] = {}
                for j in range(len(prob)):
                    advice[i][actions_id[j]] = prob[j]
            history_id, action = policy.get_action(advice)
            watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t, 0])]

            if action[0].action.id not in list(watched_list['movie_id']):
                policy.reward(history_id, {action[0].action.id: 0.0})
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                policy.reward(history_id, {action[0].action.id: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    elif bandit == 'random':
        for t in range(times):
            action = actions_id[np.random.randint(0, len(actions)-1)]
            watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t, 0])]

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
    streaming_batch, user_feature, actionstorage, reward_list, action_context = get_data()
    streaming_batch_small = streaming_batch.iloc[0:10000]

    # conduct regret analyses
    experiment_bandit = ['LinUCB', 'LinThompSamp', 'Exp4P', 'UCB1', 'Exp3', 'random']
    regret = {}
    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    i = 0
    for bandit in experiment_bandit:
        policy = policy_generation(bandit, actionstorage)
        seq_error = policy_evaluation(policy, bandit, streaming_batch_small, user_feature, reward_list,
                                      actionstorage, action_context)
        regret[bandit] = regret_calculation(seq_error)
        plt.plot(range(len(streaming_batch_small)), regret[bandit], c=col[i], ls='-', label=bandit)
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