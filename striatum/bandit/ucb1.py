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
        action_times = {}
        for action_id in self.action_ids:
            empirical_reward[action_id] = 1.0
            action_times[action_id] = 1.0
        total_time = float(len(self._actions))
        self._modelstorage.save_model({'empirical_reward': empirical_reward,
                                      'action_times': action_times, 'total_time': total_time})

    def ucb1(self):
        while True:
            empirical_reward = self._modelstorage.get_model()['empirical_reward']
            action_times = self._modelstorage.get_model()['action_times']
            total_time = self._modelstorage.get_model()['total_time']

            estimated_reward = {}
            uncertainty = {}
            score = {}
            for action_id in self.action_ids:
                estimated_reward[action_id] = empirical_reward[action_id]/action_times[action_id]
                uncertainty[action_id] = np.sqrt(2*np.log(total_time)/action_times[action_id])
                score[action_id] = estimated_reward[action_id] + uncertainty[action_id]
            yield estimated_reward, uncertainty, score

        raise StopIteration

    def get_action(self, context, n_actions=1):
        """Return the action to perform

        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context available.

        n_actions: int
            Number of actions wanted to recommend users.

        Returns
        -------
        history_id : int
            The history id of the action.

        action : list of dictionaries
            In each dictionary, it will contains {Action object, estimated_reward, uncertainty}
        """

        if self.ucb1_ is None:
            self.ucb1_ = self.ucb1()
            estimated_reward, uncertainty, score = six.next(self.ucb1_)
        else:
            estimated_reward, uncertainty, score = six.next(self.ucb1_)

        action_recommendation = []
        actions_recommendation_ids = sorted(score, key=score.get, reverse=True)[:n_actions]
        for action_id in actions_recommendation_ids:
            action_id = int(action_id)
            action = [action for action in self._actions if action.action_id == action_id][0]
            action_recommendation.append({'action': action, 'estimated_reward': estimated_reward[action_id],
                                     'uncertainty': uncertainty[action_id], 'score': score[action_id]})

        history_id = self._historystorage.add_history(context, action_recommendation, reward=None)
        return history_id, action_recommendation

    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """

        # Update the model
        empirical_reward = self._modelstorage.get_model()['empirical_reward']
        action_times = self._modelstorage.get_model()['action_times']
        total_time = self._modelstorage.get_model()['total_time']
        for action_id, reward_tmp in rewards.items():
            empirical_reward[action_id] += reward_tmp
            action_times[action_id] += 1.0
            total_time += 1.0
            self._modelstorage.save_model({'empirical_reward': empirical_reward,
                                           'action_times': action_times, 'total_time': total_time})
        # Update the history
        self._historystorage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation
        """
        action_ids = [actions[i].action_id for i in range(len(actions))]
        self._actions.extend(actions)

        empirical_reward = self._modelstorage.get_model()['empirical_reward']
        action_times = self._modelstorage.get_model()['action_times']
        total_time = self._modelstorage.get_model()['total_time']

        for action_id in self.action_ids:
            empirical_reward[action_id] = 1.0
            action_times[action_id] = 1.0

        self._modelstorage.save_model({'empirical_reward': empirical_reward,
                                       'action_times': action_times, 'total_time': total_time})
