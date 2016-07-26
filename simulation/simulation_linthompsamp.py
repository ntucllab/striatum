import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
sys.path.append("..")
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linthompsamp


class LinThompSampLinearPayoff:
    def __init__(self, t, d, actions):
        self.t = t
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

    def policy_evaluation(self, policy, context, desired_action, delta, r, epsilon):
        if policy != 'LinThompSamp':
            print("We don't support other bandit algorithms now!")
        else:
            historystorage = history.MemoryHistoryStorage()
            modelstorage = model.MemoryModelStorage()
            policy = linthompsamp.LinThompSamp(self.actions, historystorage, modelstorage,
                                               self.d, delta, r, epsilon)
            seq_error = np.zeros(shape=(self.t, 1))
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
        tunning_region = np.arange(0.01, 0.99, 0.1)
        ctr_delta = np.zeros(shape=(len(tunning_region), 1))
        ctr_r = np.zeros(shape=(len(tunning_region), 1))
        ctr_epsilon = np.zeros(shape=(len(tunning_region), 1))
        context, desired_action = self.data_simulation()
        i = 0
        for para in tunning_region:
            seq_error = self.policy_evaluation('LinThompSamp', context, desired_action,
                                               delta=para, r=0.01, epsilon=0.5)
            ctr_delta[i] = self.t - seq_error[-1]
            seq_error = self.policy_evaluation('LinThompSamp', context, desired_action,
                                               delta=0.5, r=para, epsilon=0.5)
            ctr_r[i] = self.t - seq_error[-1]
            seq_error = self.policy_evaluation('LinThompSamp', context, desired_action,
                                               delta=0.5, r=0.01, epsilon=para)
            ctr_epsilon[i] = self.t - seq_error[-1]
            i += 1

        ctr_delta = ctr_delta / self.t
        ctr_r = ctr_r / self.t
        ctr_epsilon = ctr_epsilon / self.t

        plt.figure(1)
        plt.subplot(211)
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
        plt.title("Parameter Tunning Curve - LinThompSamp")
        plt.show()

    def regret_bound(self):
        context, desired_action = self.data_simulation()
        seq_error = self.policy_evaluation('LinThompSamp', context, desired_action,
                                           delta=0.9, r=0.01, epsilon=0.5)
        seq_error = [x/y for x, y in zip(seq_error, range(1, self.t + 1))]
        plt.figure(1)
        plt.subplot(211)
        plt.plot(range(self.t), seq_error, 'r-', label="delta=0.9, r=0.01, epsilon=0.5")
        plt.xlabel('time')
        plt.ylabel('regret')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.title("Regret Bound with respect to T - LinThompSamp")
        plt.show()

if __name__ == '__main__':
    simulation = LinThompSampLinearPayoff(1000, 5, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()
    simulation2 = LinThompSampLinearPayoff(10000, 5, [1, 2, 3, 4, 5])
    simulation2.regret_bound()
