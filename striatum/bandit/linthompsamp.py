import logging
import math

from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class LinThompSamp (BaseBandit):

    def __init__(self, actions, HistoryStorage, ModelStorage, d=6, delta=0.5, R=0.5, epsilon=0.1):
        super(LinThompSamp, self).__init__(HistoryStorage, ModelStorage, actions)

        self.last_history_id = -1
        self.linthompsamp_ = None
        self.d = d

        # 0 < delta <= 1
        if not isinstance(delta, float):
            raise ValueError("delta should be float")
        elif (delta < 0) or (delta > 1):
            raise ValueError("delta should be in (0, 1]")
        else:
            self.delta = delta

        # R > 0
        if not isinstance(R, float):
            raise ValueError("R should be float")
        elif R <= 0:
            raise ValueError("R should be positive")
        else:
            self.R = R

        # 0 < epsilon < 1
        if not isinstance(epsilon, float):
            raise ValueError("epsilon should be float")
        elif (epsilon < 0) or (epsilon > 1):
            raise ValueError("epsilon should be in (0, 1)")
        else:
            self.epsilon = epsilon

        # model initialization
        self.t = 0
        v = 0
        B = np.identity(self.d)
        muhat = np.matrix(np.zeros(self.d)).T
        f = np.matrix(np.zeros(self.d)).T
        self._ModelStorage.save_model({'B': B, 'muhat': muhat, 'f': f})

    def linthompsamp(self):

        while True:
            context = yield
            self.t += 1
            B = self._ModelStorage.get_model()['B']
            muhat = self._ModelStorage.get_model()['muhat']
            v = self.R * np.sqrt(24 / self.epsilon * self.d * np.log(self.t / self.delta))
            mu = np.random.multivariate_normal(np.array(muhat.T)[0], v**2 * np.linalg.inv(B), 1)[0]
            action_max = self.actions[np.argmax(np.dot(np.array(context), np.array(mu)))]
            yield action_max
        raise StopIteration


    def get_action(self, context):
        """Return the action to perform
        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context avaliable.
        Returns
        -------
        history_id : int
            The history id of the action.
        action : Actions object
            The action to perform.
        """
        if self.linthompsamp_ is None:
            self.linthompsamp_ = self.linthompsamp()
            self.linthompsamp_.next()
            action_max = self.linthompsamp_.send(context)
        else:
            self.linthompsamp_.next()
            action_max = self.linthompsamp_.send(context)

        self.last_history_id = self.last_history_id + 1
        self._HistoryStorage.add_history(context, action_max, reward=None)
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

        context = self._HistoryStorage.unrewarded_histories[history_id].context
        reward_action = self._HistoryStorage.unrewarded_histories[history_id].action
        reward_action_idx = self.actions.index(reward_action)

        # Update the model
        B = self._ModelStorage.get_model()['B']
        f = self._ModelStorage.get_model()['f']
        B += np.dot(np.matrix(context)[reward_action_idx].T, np.matrix(context)[reward_action_idx])
        f += reward * np.matrix(context)[reward_action_idx].T
        muhat = np.dot(np.linalg.inv(B),f)
        self._ModelStorage.save_model({'B': B, 'muhat': muhat, 'f': f})

        # Update the history
        self._HistoryStorage.add_reward(history_id, reward)
