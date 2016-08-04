from striatum.storage import history
from striatum.storage import model
from striatum.bandit import exp4p
from striatum import simulation
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np


def expert_training(history_context, history_action):
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB(), )
    logreg.fit(history_context, history_action)
    mnb.fit(history_context, history_action)
    return [logreg, mnb]


def main():
    times = 1000
    d = 5
    actions = [1, 2, 3, 4, 5]
    history_context, history_action = simulation.data_simulation2(2000, d, actions)
    models = expert_training(history_context, history_action)

    # Parameter tunning
    tunning_region = np.arange(0.01, 1, 0.05)
    ctr_tunning = np.zeros(shape=(len(tunning_region), 1))
    context1, desired_action1 = simulation.data_simulation2(times, d, actions)
    i = 0
    for delta in tunning_region:
        historystorage = history.MemoryHistoryStorage()
        modelstorage = model.MemoryModelStorage()
        policy = exp4p.Exp4P(actions, historystorage, modelstorage, models, delta=delta, pmin=None)
        seq_error = simulation.policy_evaluation(policy, context1, desired_action1)
        ctr_tunning[i] = times - seq_error[-1]
        i += 1
    ctr_tunning /= times
    delta_opt = tunning_region[np.argmax(ctr_tunning)]
    simulation.tuning_plot(tunning_region, ctr_tunning, label="delta changes")

    # Regret Analysis
    times = 10000
    context2, desired_action2 = simulation.data_simulation2(times, d, actions)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    policy = exp4p.Exp4P(actions, historystorage, modelstorage, models, delta=delta_opt, pmin=None)
    regret = simulation.regret_calculation(simulation.policy_evaluation(policy, context2, desired_action2))
    simulation.regret_plot(times, regret, label='delta = ' + str(delta_opt))


if __name__ == '__main__':
    main()
