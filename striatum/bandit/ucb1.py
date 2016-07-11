
import logging

import numpy as np

from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class UCB1(BaseBandit):

    def __init__(self, actions, history_storage, model_storage):
        super(UCB1, self).__init__(history_storage, model_storage, actions)
        self._history_storage = history_storage
        self._model_storage = model_storage
        self._actions = actions
        for key in self._actions:
            self._model_storage._model['empirical_rewards'][key] = np.identity(len(self._actions))
            self._model_storage._model['n_plays'][key] = np.identity(len(self._actions))

    def ucb1(self):
        while True:
            reward = self._model_storage._model['empirical_rewards']
            plays = self._model_storage._model['n_plays']
            action_max = self._actions[np.argmax(reward/plays+np.sqrt(2 * np.log(self.history_id+1) / plays))]
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
        learn = self.ucb()
        action_max = learn.next()
        self.last_history_id = self.last_history_id + 1
        self._history_storage.add_history(np.transpose(np.array([context])), action_max, reward=None)
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
        reward_action = self._history_storage.unrewarded_histories[history_id].action

        # Update the model
        self._model_storage._model['emperical_rewards'][reward_action] += 1
        self._model_storage._model['n_plays'][reward_action] += 1

        # Update the history
        self._history_storage.add_reward(history_id, reward)
