"""Upper Confidence Bound 1
This module contains a class that implements UCB1 algorithm, a famous multi-armed bandit algorithm without context.
"""

import logging
import numpy as np
import six
from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class UCB1(BaseBandit):

    """Upper Confidence Bound 1

    Parameters
    ----------
    actions : {array-like, None}
        Actions (arms) for recommendation
    historystorage: a :py:mod:'striatum.storage.HistoryStorage' object
        The object where we store the histories of contexts and rewards.
    modelstorage: a :py:mod:'straitum.storage.ModelStorage' object
        The object where we store the model parameters.

    Attributes
    ----------
    ucb1\_ : 'ucb1' object instance
        The multi-armed bandit algorithm instances.

    References
    ----------
    .. [1]  Peter Auer, et al. "Finite-time Analysis of the Multiarmed Bandit Problem."
            Machine Learning, 47. 2002.
    """

    def __init__(self, actions, historystorage, modelstorage):
        super(UCB1, self).__init__(historystorage, modelstorage, actions)
        self.ucb1_ = None
        empirical_reward = {}
        n_actions = {}
        for key in self._actions:
            empirical_reward[key] = 1.0
            n_actions[key] = 1.0
        n_total = float(len(self._actions))
        self._modelstorage.save_model({'empirical_reward': empirical_reward,
                                      'n_actions': n_actions, 'n_total': n_total})

    def ucb1(self):
        while True:
            empirical_reward = np.array(
                [self._modelstorage.get_model()['empirical_reward'][action] for action in self._actions])
            n_actions = np.array([self._modelstorage.get_model()['n_actions'][action] for action in self._actions])
            n_total = self._modelstorage.get_model()['n_total']
            action_max = self._actions[np.argmax(empirical_reward/n_actions + np.sqrt(2*np.log(n_total)/n_actions))]
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
        if self.ucb1_ is None:
            self.ucb1_ = self.ucb1()
            action_max = six.next(self.ucb1_)
        else:
            action_max = six.next(self.ucb1_)

        # update the history
        history_id = self._historystorage.add_history(context, action_max, reward=None)
        return history_id, action_max

    def reward(self, history_id, reward):
        """Reward the previous action with reward.
        Parameters
        ----------
        history_id : int
            The history id of the action to reward.
        reward : float
            A float representing the feedback given to the action, the higher
            the better.
        """
        reward_action = self._historystorage.unrewarded_histories[history_id].action

        # Update the model
        empirical_reward = self._modelstorage.get_model()['empirical_reward']
        n_actions = self._modelstorage.get_model()['n_actions']
        n_total = self._modelstorage.get_model()['n_total']
        empirical_reward[reward_action] += 1.0
        n_actions[reward_action] += 1.0
        n_total += 1.0
        self._modelstorage.save_model({'empirical_reward': empirical_reward,
                                       'n_actions': n_actions, 'n_total': n_total})
        # Update the history
        self._historystorage.add_reward(history_id, reward)
