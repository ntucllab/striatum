import six
from six.moves import range, zip
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt

from buzzni.ai.reco.mab.storage import MemoryHistoryStorage, MemoryModelStorage
from buzzni.ai.reco.mab.bandit.exp4p import Exp4P
from buzzni.ai.reco.mab.storage import Action
from buzzni.ai.reco.mab import simulation


def train_expert(history_context, history_action):
    n_round = len(history_context)
    history_context = np.array([history_context[t] for t in range(n_round)])
    history_action = np.array([history_action[t] for t in range(n_round)])
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB())
    logreg.fit(history_context, history_action)
    mnb.fit(history_context, history_action)
    return [logreg, mnb]


def get_advice(context, action_ids, experts):
    advice = {}
    for t, context_t in six.viewitems(context):
        advice[t] = {}
        for exp_i, expert in enumerate(experts):
            prob = expert.predict_proba(context_t[np.newaxis, :])[0]
            advice[t][exp_i] = {}
            for action_id, action_prob in zip(action_ids, prob):
                advice[t][exp_i][action_id] = action_prob
    return advice


def main():  # pylint: disable=too-many-locals
    n_rounds = 1000
    context_dimension = 5
    actions = [Action(i) for i in range(5)]

    action_ids = [0, 1, 2, 3, 4]
    context1, desired_actions1 = simulation.simulate_data(
        3000, context_dimension, actions, "Exp4P", random_state=0)
    experts = train_expert(context1, desired_actions1)

    # Parameter tuning
    tuning_region = np.arange(0.01, 1, 0.05)
    ctr_tuning = np.empty(len(tuning_region))
    advice1 = get_advice(context1, action_ids, experts)

    for delta_i, delta in enumerate(tuning_region):
        historystorage = MemoryHistoryStorage()
        modelstorage = MemoryModelStorage()
        policy = Exp4P(actions, historystorage, modelstorage,
                       delta=delta, p_min=None)
        cum_regret = simulation.evaluate_policy(policy, advice1,
                                                desired_actions1)
        ctr_tuning[delta_i] = n_rounds - cum_regret[-1]
    ctr_tuning /= n_rounds
    delta_opt = tuning_region[np.argmax(ctr_tuning)]
    simulation.plot_tuning_curve(tuning_region, ctr_tuning,
                                 label="delta changes")

    # Regret Analysis
    n_rounds = 10000
    context2, desired_actions2 = simulation.simulate_data(
        n_rounds, context_dimension, actions, "Exp4P", random_state=1)
    advice2 = get_advice(context2, action_ids, experts)
    historystorage = MemoryHistoryStorage()
    modelstorage = MemoryModelStorage()
    policy = Exp4P(actions, historystorage, modelstorage,
                   delta=delta_opt, p_min=None)

    for t in range(n_rounds):
        history_id, action = policy.get_action(advice2[t], 1)
        action_id = action[0]['action'].action_id
        if desired_actions2[t] != action_id:
            policy.reward(history_id, {action_id: 0})
        else:
            policy.reward(history_id, {action_id: 1})

    policy.plot_avg_regret()
    plt.show()


if __name__ == '__main__':
    main()
