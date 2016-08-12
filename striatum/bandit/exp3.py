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
        query_vector = {}
        w = {}
        for action_id in self._actions_id:
            query_vector[action_id] = 0     # probability distribution for action recommendation)
            w[action_id] = 1                # weight vector
        self._modelstorage.save_model({'query_vector': query_vector, 'w': w})

    def exp3(self):

        """The generator which implements the main part of Exp3.
        """

        while True:
            w = self._modelstorage.get_model()['w']
            w_sum = np.sum(w.values())

            for action_id in self._actions_id:
                query_vector = (1 - self.gamma) * w[action_id] / w_sum + self.gamma / len(self._actions_id)

            self._modelstorage.save_model({'query_vector': query_vector, 'w': w})

            estimated_reward = {}
            uncertainty = {}
            score = {}
            for i in range(self.k):
                estimated_reward[self._actions_id[i]] = query_vector[i]
                uncertainty[self._actions_id[i]] = 0
                score[self._actions_id[i]] = query_vector[i]
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
                In each dictionary, it will contains {rank: Action object, estimated_reward, uncertainty}
        """

        if self.exp3_ is None:
            self.exp3_ = self.exp3()
            estimated_reward, uncertainty, score = six.next(self.exp3_)
        else:
            estimated_reward, uncertainty, score = six.next(self.exp3_)

        action_recommend = []
        actions_recommend_id = np.random.choice(self._actions_id, size=n_action, p=score.values(), replace=False)

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

        w = self._modelstorage.get_model()['w']
        query_vector = self._modelstorage.get_model()['query_vector']
        actions_id = query_vector.keys()

        # Update the model
        for action_id, reward_tmp in reward.items():
            rhat = {}
            for i in actions_id:
                rhat[i] = 0.0
            rhat[action_id] = reward_tmp / query_vector[action_id]
            w[action_id] *= np.exp(self.gamma * rhat / len(self._actions_id))

        self._modelstorage.save_model({'query_vector': query_vector, 'w': w})

        # Update the history
        self._historystorage.add_reward(history_id, reward)

    def add_action(self, actions):
        """ Add new actions (if needed).

            Parameters
            ----------
            actions : list
                Actions (arms) for recommendation
        """

        actions_id = [actions[i].action_id for i in range(len(actions))]
        w = self._modelstorage.get_model()['w']
        query_vector = self._modelstorage.get_model()['query_vector']

        for action_id in actions_id:
            query_vector[action_id] = 0
            w[action_id] = 1  # weight vector

        self._actions.extend(actions)
        self._actions_id.extend(actions_id)
