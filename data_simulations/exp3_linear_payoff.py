import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import exp3


class LinearPayoffSimulation:

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

    def policy_evaluation(self, policy, context, desired_action, gamma):
        if policy != 'EXP3':
            print("We don't support other bandit algorithms now!")
        else:
            historystorage = history.MemoryHistoryStorage()
            modelstorage = model.MemoryModelStorage()
            sum_error = 0
            policy = exp3.Exp3(self.actions, historystorage, modelstorage, gamma)
            for t in range(self.T):
                history_id, action = policy.get_action(context[t, :].tolist())
                if desired_action[t][0] != action:
                    policy.reward(history_id, 0)
                    sum_error += 1
                else:
                    policy.reward(history_id, 1)
            return self.T - sum_error

    def parameter_tuning(self):
        ctr = np.zeros(shape=(len(np.arange(0.001, 1, 0.03)), 1))
        context, desired_action = self.data_simulation()
        i = 0
        for gamma in np.arange(0.001, 1, 0.03):
            ctr[i] = self.policy_evaluation('EXP3', context, desired_action, gamma)
            i += 1
        ctr = ctr / self.T
        plt.plot(np.arange(0.001, 1, 0.03), ctr)

if __name__ == '__main__':
    simulation = LinearPayoffSimulation(1000, 4, [1, 2, 3, 4, 5])
    simulation.parameter_tuning()
