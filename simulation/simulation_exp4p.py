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
        self.t = t
        self.d = d
        self.actions = actions

    def data_simulation(self):
        context = np.random.uniform(0, 1, (self.t, self.d))
        desired_action = np.zeros(shape=(self.t, 1))
        n_actions = len(self.actions)
        for t in range(self.t):
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
            seq_error = np.zeros(shape=(self.t, 1))
            policy = exp4p.Exp4P(self.actions, historystorage, modelstorage, models, delta, pmin=None)
            for t in range(self.t):
                history_id, action = policy.get_action(context[t])
                if desired_action[t][0] != action:
                    policy.reward(history_id, 0)
                    # sum_error += 1
                    if t == 0:
                        seq_error[t] = 1.0
                    else:
                        seq_error[t] = seq_error[t - 1] + 1.0
                else:
                    policy.reward(history_id, 1)
                    if t > 0:
                        seq_error[t] = seq_error[t - 1]
            return seq_error

    def parameter_tuning(self):
        tunning_region = np.arange(0.01, 1, 0.05)
        ctr = np.zeros(shape=(len(tunning_region), 1))
        context, desired_action = self.data_simulation()
        models = self.expert_training()
        i = 0
        for delta in tunning_region:
            seq_error = self.policy_evaluation('EXP4P', context, desired_action, models, delta)
            ctr[i] = self.t - seq_error[-1]
            i += 1
        ctr = ctr / self.t
        plt.figure(1)
        plt.subplot(211)
        plt.plot(tunning_region, ctr, 'ro-', label="delta changes")
        plt.xlabel('parameter value')
        plt.ylabel('CTR')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Parameter Tunning Curve - EXP4.P")
        plt.show()

    def regret_bound(self):
        context, desired_action = self.data_simulation()
        models = self.expert_training()
        seq_error = self.policy_evaluation('EXP4P', context, desired_action, models, delta=0.98)
        seq_error = [x/y for x, y in zip(seq_error, range(1, self.t + 1))]
        plt.figure(1)
        plt.subplot(211)
        plt.plot(range(self.t), seq_error, 'r-', label="delta=0.5")
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Regret Bound with respect to T - EXP4P")
        plt.show()


if __name__ == '__main__':
    simulation = Exp4PLinearPayoff(1000, 5, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()
    simulation2 = Exp4PLinearPayoff(10000, 5, [1, 2, 3, 4, 5])
    simulation2.regret_bound()
