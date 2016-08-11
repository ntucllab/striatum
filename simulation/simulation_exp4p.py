from striatum.storage import history
from striatum.storage import model
from striatum.bandit import exp4p
from striatum import simulation
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from striatum.bandit.bandit import Action


def train_expert(history_context, history_action):
    history_context = np.array([history_context[i] for i in history_context.keys()])
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
    times = 1000
    d = 5
    a1 = Action(1, 'a1', 'content 1')
    a2 = Action(2, 'a2', 'content 2')
    a3 = Action(3, 'a3', 'content 3')
    a4 = Action(4, 'a4', 'content 4')
    a5 = Action(5, 'a5', 'content 5')
    actions = [a1, a2, a3, a4, a5]
    actions_id = [1, 2, 3, 4, 5]
    history_context, history_action = simulation.data_simulation(3000, d, actions, "Exp4P")
    experts = train_expert(history_context, history_action)

    # Parameter tunning
    tunning_region = np.arange(0.01, 1, 0.05)
    ctr_tunning = np.zeros(shape=(len(tunning_region), 1))
    context1, desired_action1 = simulation.data_simulation(times, d, actions, "Exp4P")
    advice1 = get_advice(context1, actions_id, experts)
    i = 0
    for delta in tunning_region:
        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = exp4p.Exp4P(actions, historystorage, modelstorage, delta=delta, pmin=None)
        seq_error = simulation.policy_evaluation(policy, advice1, desired_action1)
        ctr_tunning[i] = times - seq_error[-1]
        i += 1
    ctr_tunning /= times
    delta_opt = tunning_region[np.argmax(ctr_tunning)]
    simulation.tuning_plot(tunning_region, ctr_tunning, label="delta changes")

    # Regret Analysis
    times = 10000
    context2, desired_action2 = simulation.data_simulation(times, d, actions, "Exp4P")
    advice2 = get_advice(context2, actions_id, experts)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = exp4p.Exp4P(actions, historystorage, modelstorage, delta=delta_opt, pmin=None)
    regret = simulation.regret_calculation(simulation.policy_evaluation(policy, advice2, desired_action2))
    simulation.regret_plot(times, regret, label='delta = ' + str(delta_opt))


if __name__ == '__main__':
    main()
