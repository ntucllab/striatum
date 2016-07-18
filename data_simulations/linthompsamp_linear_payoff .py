
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linthompsamp
import numpy as np
import matplotlib.pyplot as plt


class LinearPayoffSimulation:
    Linear payoff 
    def __init__(self, T, d, actions):
        self.T = T
        self.d = d
        self.actions = actions

    def data_simulation(self):
        context = {}
        desired_action = np.zeros(shape = (self.T, 1))
        n_actions = len(self.actions)
        for t in range(self.T):
            context[t] = np.random.uniform(0, 1, (len(n_actions),self.d))
            desired_action[t] = self.actions[np.argmax(np.sum(context[t], axis=1))]
        return context, desired_action

    def policy_evaluation(self, policy, context, desired_action, delta, R, epsilon):
        if policy != 'LinThompSamp':
            print("We don't support other bandit algorithms now!")
        else:
            HistoryStorage = history.MemoryHistoryStorage()
            ModelStorage = model.MemoryModelStorage()
            sum_error = 0
            policy = linthompsamp.LinThompSamp(self.actions, HistoryStorage, ModelStorage,
                                               self.d, delta, R, epsilon)
            for t in range(self.T):
                history_id, action = policy.get_action(context[t])
                if desired_action[t][0] != action:
                    policy.reward(history_id, 0)
                    sum_error = sum_error + 1
                else:
                    policy.reward(history_id, 1)
            return (self.T - sum_error)

    def parameter_tuning(self):
        tunning_region = np.arange(0.01, 0.99, 0.1)
        CTR_delta = np.zeros(shape=(len(tunning_region), 1))
        CTR_R =  np.zeros(shape=(len(tunning_region), 1))
        CTR_epsilon = np.zeros(shape=(len(tunning_region), 1))
        context, desired_action = self.data_simulation()
        i = 0
        for delta in np.arange(0.01, 0.99, 0.1):
            CTR_delta[i] = self.policy_evaluation('LinThompSamp', context, desired_action, delta=delta, R=0.5, epsilon=0.5)
            i = i + 1


        CTR_delta = CTR_delta / self.T
        plt.plot(np.arange(0.01, 0.99, 0.1), CTR_delta, label="delta change, R = 0.5, epsilon = 0.05")
        CTR_R = CTR_R / self.T
        CTR_epslion = epsilon / self.T

        plt.plot(tunning_region, CTR_R, label="delta = 0.5, R change, epsilon = 0.05")
        plt.plot(tunning_region, CTR_epslion, label="delta = 0.5, R = 0.5, epsilon change")
        plt.legend()


if __name__ == '__main__':
    simulation = LinearPayoffSimulation(1000, 10, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()'''
