import six

from striatum.storage import history
from striatum.storage import model
from striatum.bandit import exp4p
from striatum import simulation
from striatum import rewardplot as rplt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from striatum.bandit.bandit import Action


def train_expert(history_context, history_action):
    history_context = np.array(list(six.viewvalues(history_context)))
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB(), )
    logreg.fit(history_context, history_action)
    mnb.fit(history_context, history_action)
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


def main():
    n_rounds = 1000
    context_dimension = 5
    actions = [Action(i) for i in range(5)]
    actions_id = [1, 2, 3, 4, 5]
    history_context, history_action = simulation.simulate_data(
        3000, context_dimension, actions, "Exp4P", random_state=0)
    experts = train_expert(history_context, history_action)

    # Parameter tuning
    tuning_region = np.arange(0.01, 1, 0.05)
    ctr_tuning = np.zeros(shape=(len(tuning_region), 1))
    context1, desired_actions1 = simulation.simulate_data(n_rounds, context_dimension, actions, "Exp4P")
    advice1 = get_advice(context1, actions_id, experts)
    i = 0
    for delta in tuning_region:
        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = exp4p.Exp4P(actions, historystorage, modelstorage, delta=delta, pmin=None)
        cum_regret = simulation.evaluate_policy(policy, advice1, desired_actions1)
        ctr_tuning[i] = n_rounds - cum_regret[-1]
        i += 1
    ctr_tuning /= n_rounds
    delta_opt = tuning_region[np.argmax(ctr_tuning)]
    simulation.plot_tuning_curve(tuning_region, ctr_tuning, label="delta changes")

    # Regret Analysis
    n_rounds = 10000
    context2, desired_actions2 = simulation.simulate_data(n_rounds, context_dimension, actions, "Exp4P")
    advice2 = get_advice(context2, actions_id, experts)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = exp4p.Exp4P(actions, historystorage, modelstorage, delta=delta_opt, pmin=None)

    for t in range(n_rounds):
        history_id, action = policy.get_action(advice2[t], 1)
        action_id = action[0]['action'].action_id
        if desired_actions2[t] != action_id:
            policy.reward(history_id, {action_id: 0})
        else:
            policy.reward(history_id, {action_id: 1})

    policy.plot_avg_regret()


if __name__ == '__main__':
    main()
