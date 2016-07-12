
import logging

import numpy as np

from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class UCB1(BaseBandit):

    def __init__(self, actions, HistoryStorage, ModelStorage):
        super(UCB1, self).__init__(HistoryStorage, ModelStorage, actions)
        self.last_history_id = -1
        empirical_reward = {}
        n_actions = {}
        for key in self._actions:
            empirical_reward[key] = 1.0
            n_actions[key] = 1.0
        n_total = float(len(self._actions))
        self._ModelStorage.save_model({'empirical_reward': empirical_reward,
                                      'n_actions': n_actions, 'n_total': n_total})


    def ucb1(self):
        while True:
            empirical_reward = np.array([self._ModelStorage.get_model()['empirical_reward'][action] for action in self._actions])
            n_actions = np.array([self._ModelStorage.get_model()['n_actions'][action] for action in self._actions])
            n_total = self._ModelStorage.get_model()['n_total']
            action_max = self._actions[np.argmax(empirical_reward/n_actions + np.sqrt(2*np.log(n_total)/n_actions))]
            yield action_max

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
        if context is not None:
            LOGGER.warning("UCB1 does not support context.")

        # learn the model
        learn = self.ucb1()
        learn.next()
        action_max = learn.send(context)

        # update the history
        self.last_history_id = self.last_history_id + 1
        self._HistoryStorage.add_history(None, action_max, reward = None)
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

        # Update the model
        empirical_reward = self._ModelStorage.get_model()['empirical_reward']
        n_actions = self._ModelStorage.get_model()['n_actions']
        n_total = self._ModelStorage.get_model()['n_total']
        empirical_reward[reward_action] += 1.0
        n_actions[reward_action] += 1.0
        n_total += 1.0
        self._ModelStorage.save_model({'empirical_reward': empirical_reward,
                                       'n_actions': n_actions, 'n_total': n_total})
        # Update the history
        self._HistoryStorage.add_reward(history_id, reward)
