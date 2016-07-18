import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linthompsamp


class LinearPayoffSimulation:
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

    def policy_evaluation(self, policy, context, desired_action, delta, r, epsilon):
        if policy != 'LinThompSamp':
            print("We don't support other bandit algorithms now!")
        else:
            historystorage = history.MemoryHistoryStorage()
            modelstorage = model.MemoryModelStorage()
            sum_error = 0
            policy = linthompsamp.LinThompSamp(self.actions, historystorage, modelstorage,
                                               self.d, delta, r, epsilon)
            for t in range(self.T):
                history_id, action = policy.get_action(context[t])
                if desired_action[t][0] != action:
                    policy.reward(history_id, 0)
                    sum_error += 1
                else:
                    policy.reward(history_id, 1)
            return self.T - sum_error

    def parameter_tuning(self):
        tunning_region = np.arange(0.01, 0.99, 0.1)
        ctr_delta = np.zeros(shape=(len(tunning_region), 1))
        ctr_r = np.zeros(shape=(len(tunning_region), 1))
        ctr_epsilon = np.zeros(shape=(len(tunning_region), 1))
        context, desired_action = self.data_simulation()
        i = 0
        for para in tunning_region:
            ctr_delta[i] = self.policy_evaluation('LinThompSamp', context, desired_action,
                                                  delta=para, r=0.01, epsilon=0.5)
            ctr_r[i] = self.policy_evaluation('LinThompSamp', context, desired_action,
                                              delta=0.5, r=para, epsilon=0.5)
            ctr_epsilon[i] = self.policy_evaluation('LinThompSamp', context, desired_action,
                                                    delta=0.5, r=0.01, epsilon=para)
            i += 1

        ctr_delta = ctr_delta / self.T
        ctr_r = ctr_r / self.T
        ctr_epsilon = ctr_epsilon / self.T

        plt.plot(np.arange(0.01, 0.99, 0.1), ctr_delta, 'ro-',
                 np.arange(0.01, 0.99, 0.1), ctr_r, 'gs-',
                 np.arange(0.01, 0.99, 0.1), ctr_epsilon, 'b^-')
        line1 = mlines.Line2D([], [], color='r', marker='o',
                              label="delta changes, R = 0.01, eps = 0.5")
        line2 = mlines.Line2D([], [], color='g', marker='s',
                              label="delta = 0.5, R = changes, eps = 0.5")
        line3 = mlines.Line2D([], [], color='b', marker='^',
                              label="delta = 0.5, R = 0.01, eps = changes")
        plt.xlabel('parameter value')
        plt.ylabel('CTR')
        plt.legend(handles=[line1, line2, line3], loc='upper center', bbox_to_anchor=(0.5, -0.15))
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Parameter Tunning Curve")


if __name__ == '__main__':
    simulation = LinearPayoffSimulation(1000, 5, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()
