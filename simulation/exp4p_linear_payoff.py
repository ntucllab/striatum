from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import exp4p


class Exp4PLinearPayoff:

    def __init__(self, t, d, actions):
        self.T = t
        self.d = d
        self.actions = actions

    def data_simulation(self):
        context = np.random.uniform(0, 1, (self.T, self.d))
        desired_action = np.zeros(shape=(self.T, 1))
        n_actions = len(self.actions)
        for t in range(self.T):
            for i in range(n_actions):
                if i * self.d/n_actions < sum(context[t, :]) <= (i+1) * self.d/n_actions:
                    desired_action[t] = self.actions[i]
        return context, desired_action

    def expert_training(self):
        history_context, history_action = self.data_simulation()
        logreg = OneVsRestClassifier(LogisticRegression())
        mnb = OneVsRestClassifier(MultinomialNB(),)
        logreg.fit(history_context, history_action)
        mnb.fit(history_context, history_action)
        return [logreg, mnb]

    def policy_evaluation(self, policy, context, desired_action, models, delta):
        if policy != 'EXP4P':
            print("We don't support other bandit algorithms now!")
        else:
            historystorage = history.MemoryHistoryStorage()
            modelstorage = model.MemoryModelStorage()
            sum_error = 0
            policy = exp4p.Exp4P(self.actions, historystorage, modelstorage, models, delta, pmin=None)
            for t in range(self.T):
                history_id, action = policy.get_action(context[t, :].tolist())
                if desired_action[t][0] != action:
                    policy.reward(history_id, 0)
                    sum_error += 1
                else:
                    policy.reward(history_id, 1)
            return self.T - sum_error

    def parameter_tuning(self):
        tunning_region = np.arange(0.01, 1, 0.05)
        ctr = np.zeros(shape=(len(tunning_region), 1))
        context, desired_action = self.data_simulation()
        models = self.expert_training()
        i = 0
        for delta in tunning_region:
            ctr[i] = self.policy_evaluation('EXP4P', context, desired_action, models, delta)
            i += 1
        ctr = ctr / self.T
        plt.plot(tunning_region, ctr, 'ro-', label="delta changes")
        plt.xlabel('parameter value')
        plt.ylabel('CTR')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Parameter Tunning Curve - EXP4.P")

if __name__ == '__main__':
    simulation = Exp4PLinearPayoff(1000, 5, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()
