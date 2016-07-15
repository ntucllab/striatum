
import logging

from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class Exp3(BaseBandit):

    def __init__(self, actions, HistoryStorage, ModelStorage, gamma):
        super(Exp3, self).__init__(HistoryStorage, ModelStorage, actions)

        self.last_history_id = -1
        self.n_actions = len(self.actions)          # number of actions (i.e. K in the paper)
        self.exp3_ = None

        # gamma in (0,1]
        if not isinstance(gamma, float):
            raise ValueError("gamma should be float, the one"
                             "given is: %f" % gamma)
        elif (gamma <= 0) or (gamma > 1):
            raise ValueError("gamma should be in (0, 1], the one"
                            "given is: %f" % gamma)
        else:
            self.gamma = gamma

        # Initialize the model storage
        query_vector = np.zeros(self.n_actions)     # probability distribution for action recommendation)
        w = np.ones(self.n_actions)                 # weight vector
        self._ModelStorage.save_model({'query_vector': query_vector, 'w': w})

    def exp3(self):

        """The generator which implements the main part of Exp3.
        """

        while True:
            w = self._ModelStorage.get_model()['w']
            w_sum = np.sum(w)
            query_vector = (1 - self.gamma) * w / w_sum + self.gamma / self.n_actions

            self._ModelStorage.save_model({'query_vector': query_vector, 'w': w})

            action_idx = np.random.choice(np.arange(len(self.actions)), size=1, p = query_vector/sum(query_vector))[0]
            action_max = self.actions[action_idx]
            yield action_max

        raise StopIteration

    def get_action(self, context):
        """Return the action to perform

        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context available.

        Returns
        -------
        history_id : int
            The history id of the action.

        action : Actions object
            The action to perform.
        """
        if self.exp4_ is None:
            self.exp3_ = self.exp3()
            action_max = self.exp3_.next()
        else:
            action_max = self.exp3_.send(context)

        self.last_history_id = self.last_history_id + 1
        self._HistoryStorage.add_history(np.transpose(np.array([context])), action_max, reward=None)
        return self.last_history_id, action_max

    def reward(self, history_id, reward):
        """Reward the preivous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        reward : float
            A float representing the feedback given to the action, the higher
            the better.
        """

        reward_action = self._HistoryStorage.unrewarded_histories[history_id].action
        reward_action_idx = self.actions.index(reward_action)
        w_old = self._ModelStorage.get_model()['w']
        query_vector = self._ModelStorage.get_model()['query_vector']

        # Update the model
        rhat = np.zeros(self.n_actions)
        rhat[reward_action_idx] = reward/query_vector[reward_action_idx]
        w_new = w_old * np.exp(self.gamma * rhat / self.n_actions)

        self._ModelStorage.save_model({'query_vector': query_vector, 'w': w_new})

        # Update the history
        self._HistoryStorage.add_reward(history_id, reward)
