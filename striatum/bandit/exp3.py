""" Exp3: Exponential-weight algorithm for Exploration and Exploitation
This module contains a class that implements EXP3, a bandit algorithm that randomly choose an action
according to a learned probability distribution.
"""

import logging
import six
from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class Exp3(BaseBandit):

    """Exp3 with pre-trained supervised learning algorithm.

        Parameters
        ----------
        actions : array-like
            Actions (arms) for recommendation.
        historystorage: a HistoryStorage object
            The place where we store the histories of contexts and rewards.
        modelstorage: a ModelStorage object
            The place where we store the model parameters.
        gamma: float, 0 < gamma <= 1
            The parameter used to control the minimum chosen probability for each action.

        Attributes
        ----------
        exp3\_ : 'exp3' object instance
            The contextual bandit algorithm instances

        References
        ----------
        .. [1]  Peter Auer, Nicolo Cesa-Bianchi, et al. "The non-stochastic multi-armed bandit problem ."
                SIAM Journal of Computing. 2002.
        """

    def __init__(self, actions, historystorage, modelstorage, gamma):
        super(Exp3, self).__init__(historystorage, modelstorage, actions)

        self.n_actions = len(self._actions)  # number of actions (i.e. K in the paper)
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
        query_vector = np.zeros(self.n_actions)  # probability distribution for action recommendation)
        w = np.ones(self.n_actions)  # weight vector
        self._modelstorage.save_model({'query_vector': query_vector, 'w': w})

    def exp3(self):

        """The generator which implements the main part of Exp3.
        """

        while True:
            w = self._modelstorage.get_model()['w']
            w_sum = np.sum(w)
            query_vector = (1 - self.gamma) * w / w_sum + self.gamma / self.n_actions

            self._modelstorage.save_model({'query_vector': query_vector, 'w': w})

            action_idx = np.random.choice(np.arange(len(self._actions)), size=1, p=query_vector / sum(query_vector))[0]
            action_max = self._actions[action_idx]
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

        if self.exp3_ is None:
            self.exp3_ = self.exp3()
            action_max = six.next(self.exp3_)
        else:
            action_max = six.next(self.exp3_)

        history_id = self._historystorage.add_history(np.transpose(np.array([context])), action_max, reward=None)
        return history_id, action_max

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

        reward_action = self._historystorage.unrewarded_histories[history_id].action
        reward_action_idx = self._actions.index(reward_action)
        w_old = self._modelstorage.get_model()['w']
        query_vector = self._modelstorage.get_model()['query_vector']

        # Update the model
        rhat = np.zeros(self.n_actions)
        rhat[reward_action_idx] = reward / query_vector[reward_action_idx]
        w_new = w_old * np.exp(self.gamma * rhat / self.n_actions)

        self._modelstorage.save_model({'query_vector': query_vector, 'w': w_new})

        # Update the history
        self._historystorage.add_reward(history_id, reward)
