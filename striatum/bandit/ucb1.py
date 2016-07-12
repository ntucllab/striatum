
import logging

import numpy as np

from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class UCB1(BaseBandit):

    def __init__(self, actions, HistoryStorage, ModelStorage):
        super(UCB1, self).__init__(HistoryStorage, ModelStorage, actions)
        self.last_history_id = None
        empirical_reward = {}
        n_actions = {}
        for key in self._actions:
            empirical_reward[key] = 1
            n_actions[key] = 1
        n_total = len(self._actions)
        self._ModelStorage.save_model({'empirical_reward': empirical_reward,
                                      'n_actions': n_actions, 'n_total': n_total})


    def ucb1(self):
        def upper_bound(t, n_plays):
            return np.sqrt(2 * np.log(t) / n_plays)

        t = 0
        n_arms = len(self.actions)
        n_plays = np.zeros(n_arms)
        empirical_reward = np.zeros(n_arms)

        for i in range(n_arms):
            reward = yield i
            empirical_reward[i] += reward
            t += 1
            n_plays[i] += 1

        while True:
            ucbs = empirical_reward / n_plays + upper_bound(t, n_plays)
            choice = np.argmax(ucbs)
            reward = yield choice
            empirical_reward[choice] += reward
            n_plays[i] += 1
            t += 1

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

        if self.last_reward is None:
            raise ValueError("The last reward have not been passed in.")

        # learn the model
        action_max = self.ucb1_.send(self.last_reward)

        # update the history
        self.last_history_id = self.last_history_id + 1
        self._HistoryStorage.add_history(action_max, reward=None)
        return self.last_history_id, action_max

        return history_id, action

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
        if history_id != self.last_history_id:
            raise ValueError("The history_id should be the same as last one.")
        self.last_reward = reward
        self.storage.reward(history_id, reward)
