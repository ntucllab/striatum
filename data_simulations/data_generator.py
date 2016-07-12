import striatum
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linucb
import numpy as np
import matplotlib.pyplot as plt


class LinearPayoffSimulation():
    ''' Linear payoff '''
    def __init__(self, d, actions):
        self.HistoryStorage = history.MemoryHistoryStorage()
        self.ModelStorage = model.MemoryModelStorage()
        self.d = d
        self.actions = actions

    def data_simulation(self, T, d):
        context = np.random.uniform(0, 1, (T, d))
        desired_action = np.zeros(shape = (T, 1))
        for t in range(T):
            if sum(context[t,:]) < 1:
                desired_action[t] = 1
            elif sum(context[t,:]) < 2.5:
                desired_action[t] = 2
            else:
                desired_action[t] = 3
        return context, desired_action

    def data_evaluation(self, policy, context, desired_action, alpha,d):
        if policy != 'LinUCB':
            print("We don't support other bandit algorithms now!")
        else:
            sum_error = 0
            policy = linucb.LinUCB(self.actions, self.HistoryStorage, self.ModelStorage, alpha, d)
            for t in range(T):
                history_id, action = policy.get_action(context[t, :].tolist())
            if desired_action[t] != action:
                policy.reward(history_id, 0)
                sum_error = sum_error + 1
            else:
                policy.reward(history_id, 1)
        return 1 - sum_error / T

