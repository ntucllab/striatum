import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linucb


class LinUCBLinearPayoff:

    def __init__(self, t, d, actions):
        self.T = t
        self.d = d
        self.actions = actions

    def data_simulation(self):
        context = {}
        desired_action = np.zeros(shape=(self.T, 1))
        n_actions = len(self.actions)
        for t in range(self.T):
            context[t] = np.random.uniform(0, 1, (n_actions, self.d))
            desired_action[t] = self.actions[np.argmax(np.sum(context[t], axis=1))]
        return context, desired_action

    def policy_evaluation(self, policy, context, desired_action, alpha):
        if policy != 'LinUCB':
            print("We don't support other bandit algorithms now!")
        else:
            historystorage = history.MemoryHistoryStorage()
            modelstorage = model.MemoryModelStorage()
            sum_error = 0
            policy = linucb.LinUCB(self.actions, historystorage, modelstorage, alpha, self.d)
            for t in range(self.T):
                history_id, action = policy.get_action(context[t])
                if desired_action[t][0] != action:
                    policy.reward(history_id, 0)
                    sum_error += 1
                else:
                    policy.reward(history_id, 1)
            return self.T - sum_error

    def parameter_tuning(self):
        tunning_region = np.arange(0, 3, 0.05)
        ctr = np.zeros(shape=(len(tunning_region), 1))
        context, desired_action = self.data_simulation()
        i = 0
        for alpha in tunning_region:
            ctr[i] = self.policy_evaluation('LinUCB', context, desired_action, alpha)
            i += 1
        ctr = ctr / self.T
        plt.plot(tunning_region, ctr, 'ro-', label="alpha changes")
        plt.xlabel('parameter value')
        plt.ylabel('CTR')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Parameter Tunning Curve - LinUCB")

if __name__ == '__main__':
    simulation = LinUCBLinearPayoff(1000, 5, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()
