import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linucb


class LinUCBLinearPayoff:

    def __init__(self, t, d, actions):
        self.t= t
        self.d = d
        self.actions = actions

    def data_simulation(self):
        context = {}
        desired_action = np.zeros(shape=(self.t, 1))
        n_actions = len(self.actions)
        for t in range(self.t):
            context[t] = np.random.uniform(0, 1, (n_actions, self.d))
            desired_action[t] = self.actions[np.argmax(np.sum(context[t], axis=1))]
        return context, desired_action

    def policy_evaluation(self, policy, context, desired_action, alpha):
        if policy != 'LinUCB':
            print("We don't support other bandit algorithms now!")
        else:
            historystorage = history.MemoryHistoryStorage()
            modelstorage = model.MemoryModelStorage()
            # sum_error = 0
            policy = linucb.LinUCB(self.actions, historystorage, modelstorage, alpha, self.d)
            seq_error = np.zeros(shape=(self.t, 1))
            for t in range(self.t):
                history_id, action = policy.get_action(context[t])
                if desired_action[t][0] != action:
                    policy.reward(history_id, 0)
                    # sum_error += 1
                    if t == 0:
                        seq_error[t] = 1.0
                    else:
                        seq_error[t] = seq_error[t-1] + 1.0
                else:
                    policy.reward(history_id, 1)
                    if t > 0:
                        seq_error[t] = seq_error[t-1]
            return seq_error

    def parameter_tuning(self):
        tunning_region = np.arange(0, 3, 0.05)
        ctr = np.zeros(shape=(len(tunning_region), 1))
        context, desired_action = self.data_simulation()
        i = 0
        for alpha in tunning_region:
            seq_error = self.policy_evaluation('LinUCB', context, desired_action, alpha)
            ctr[i] = self.t - seq_error[-1]
            i += 1
        ctr = ctr / self.t
        plt.figure(1)
        plt.subplot(211)
        plt.plot(tunning_region, ctr, 'ro-', label="alpha changes")
        plt.xlabel('parameter value')
        plt.ylabel('CTR')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Parameter Tunning Curve - LinUCB")
        plt.show()

    def regret_bound(self):
        context, desired_action = self.data_simulation()
        seq_error = self.policy_evaluation('LinUCB', context, desired_action, alpha=0.42)
        seq_error = [x/y for x, y in zip(seq_error, range(1, self.t + 1))]
        plt.figure(1)
        plt.subplot(211)
        plt.plot(range(self.t), seq_error, 'r-', label="alpha =0.5")
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Regret Bound with respect to T - LinUCB")
        plt.show()

if __name__ == '__main__':
    simulation = LinUCBLinearPayoff(1000, 5, [1, 2, 3, 4, 5])
    simulation2 = LinUCBLinearPayoff(10000, 5, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()
    simulation2.regret_bound()
