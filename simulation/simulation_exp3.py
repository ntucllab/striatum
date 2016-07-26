import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import exp3


class Exp3LinearPayoff:

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

    def policy_evaluation(self, policy, context, desired_action, gamma):
        if policy != 'EXP3':
            print("We don't support other bandit algorithms now!")
        else:
            historystorage = history.MemoryHistoryStorage()
            modelstorage = model.MemoryModelStorage()
            seq_error = np.zeros(shape=(self.t, 1))
            policy = exp3.Exp3(self.actions, historystorage, modelstorage, gamma)
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
        tunning_region = np.arange(0.001, 1, 0.03)
        ctr = np.zeros(shape=(len(tunning_region), 1))
        context, desired_action = self.data_simulation()
        i = 0
        for gamma in tunning_region:
            seq_error = self.policy_evaluation('EXP3', context, desired_action, gamma)
            ctr[i] = self.t - seq_error[-1]
            i += 1
        ctr = ctr / self.t
        plt.figure(1)
        plt.subplot(211)
        plt.plot(tunning_region, ctr, 'ro-', label="gamma changes")
        plt.xlabel('parameter value')
        plt.ylabel('CTR')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Parameter Tunning Curve - EXP3")
        plt.show()

    def regret_bound(self):
        context, desired_action = self.data_simulation()
        seq_error = self.policy_evaluation('EXP3', context, desired_action, gamma=0.18)
        seq_error = [x/y for x, y in zip(seq_error, range(1, self.t + 1))]
        plt.figure(1)
        plt.subplot(211)
        plt.plot(range(self.t), seq_error, 'r-', label="gamma=0.18")
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Regret Bound with respect to T - EXP3")
        plt.show()

if __name__ == '__main__':
    simulation = Exp3LinearPayoff(1000, 5, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()
    simulation2 = Exp3LinearPayoff(10000, 5, [1, 2, 3, 4, 5])
    simulation2.regret_bound()
