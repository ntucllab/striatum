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
        for action_id in self._actions_id:
            empirical_reward[action_id] = 1.0
            n_actions[action_id] = 1.0
        n_total = float(len(self._actions))
        self._modelstorage.save_model({'empirical_reward': empirical_reward,
                                      'n_actions': n_actions, 'n_total': n_total})

    def ucb1(self):
        while True:
            empirical_reward = self._modelstorage.get_model()['empirical_reward']
            n_actions = self._modelstorage.get_model()['n_actions']
            n_total = self._modelstorage.get_model()['n_total']

            estimated_reward = {}
            uncertainty = {}
            score = {}
            for action_id in self._actions_id:
                estimated_reward[action_id] = empirical_reward[action_id]/n_actions[action_id]
                uncertainty[action_id] = np.sqrt(2*np.log(n_total)/n_actions[action_id])
                score[action_id] = estimated_reward[action_id] + uncertainty[action_id]
            yield estimated_reward, uncertainty, score

        raise StopIteration

    def get_action(self, context, n_action=1):
        """Return the action to perform

            Parameters
            ----------
            context : {array-like, None}
                The context of current state, None if no context available.

            n_action: int
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

        action_recommend = []
        actions_recommend_id = sorted(score, key=score.get, reverse=True)[:n_action]
        for action_id in actions_recommend_id:
            action_id = int(action_id)
            action = [action for action in self._actions if action.action_id == action_id][0]
            action_recommend.append({'action': action, 'estimated_reward': estimated_reward[action_id],
                                     'uncertainty': uncertainty[action_id], 'score': score[action_id]})

        history_id = self._historystorage.add_history(context, action_recommend, reward=None)
        return history_id, action_recommend

    def reward(self, history_id, reward):
        """Reward the previous action with reward.

            Parameters
            ----------
            history_id : int
                The history id of the action to reward.

            reward : dictionary
                The dictionary {action_id, reward}, where reward is a float.
        """

        # Update the model
        empirical_reward = self._modelstorage.get_model()['empirical_reward']
        n_actions = self._modelstorage.get_model()['n_actions']
        n_total = self._modelstorage.get_model()['n_total']
        for action_id, reward_tmp in reward.items():
            empirical_reward[action_id] += reward_tmp
            n_actions[action_id] += 1.0
            n_total += 1.0
            self._modelstorage.save_model({'empirical_reward': empirical_reward,
                                           'n_actions': n_actions, 'n_total': n_total})
        # Update the history
        self._historystorage.add_reward(history_id, reward)

    def add_action(self, actions):
        """ Add new actions (if needed).

            Parameters
            ----------
            actions : list
                A list of Action objects for recommendation
        """
        actions_id = [actions[i].action_id for i in range(len(actions))]
        self._actions.extend(actions)
        self._actions_id.extend(actions_id)

        empirical_reward = self._modelstorage.get_model()['empirical_reward']
        n_actions = self._modelstorage.get_model()['n_actions']
        n_total = self._modelstorage.get_model()['n_total']

        for action_id in self._actions_id:
            empirical_reward[action_id] = 1.0
            n_actions[action_id] = 1.0

        self._modelstorage.save_model({'empirical_reward': empirical_reward,
                                       'n_actions': n_actions, 'n_total': n_total})
