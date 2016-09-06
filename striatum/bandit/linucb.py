"""LinUCB with Disjoint Linear Models

This module contains a class that implements LinUCB with disjoint linear model,
a contextual bandit algorithm assuming the reward function is a linear function
of the context.
"""

import logging

import numpy as np
import six

from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class LinUCB(BaseBandit):
    r"""LinUCB with Disjoint Linear Models

    Parameters
    ----------
    actions : list of Action objects
        List of actions to be chosen from.

    historystorage: a HistoryStorage object
        The place where we store the histories of contexts and rewards.

    modelstorage: a ModelStorage object
        The place where we store the model parameters.

    alpha: float
        The constant determines the width of the upper confidence bound.

        d: int
        The dimension of the context.

    Attributes
    ----------
    linucb\_ : 'linucb' object instance
        The contextual bandit algorithm instances.

    References
    ----------
    .. [1]  Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
            News Article Recommendation." In Proceedings of the 19th
            International Conference on World Wide Web (WWW), 2010.
    """

    def __init__(self, actions, historystorage, modelstorage, alpha, d=1):
        super(LinUCB, self).__init__(historystorage, modelstorage, actions)
        self.last_reward = None
        self.alpha = alpha
        self.d = d
        self.linucb_ = None

        # Initialize LinUCB Model Parameters

        # dictionary - For any action a in actions,
        # matrix_a[a] = (DaT*Da + I) the ridge reg solution
        matrix_a = {}
        # dictionary - The inverse of each matrix_a[a] for action a in actions
        matrix_ainv = {}
        # dictionary - The cumulative return of action a, given the context xt.
        b = {}
        # dictionary - The coefficient vector of actiona with
        # linear model b = dot(xt, theta)
        theta = {}

        for action_id in self.action_ids:
            matrix_a[action_id] = np.identity(self.d)
            matrix_ainv[action_id] = np.identity(self.d)
            b[action_id] = np.zeros((self.d, 1))
            theta[action_id] = np.zeros((self.d, 1))

        self._modelstorage.save_model({'matrix_a': matrix_a,
                                       'matrix_ainv': matrix_ainv,
                                       'b': b,
                                       'theta': theta})

    @property
    def linucb(self):
        """The generator implementing the disjoint LINUCB algorithm.
        """

        while True:
            context = yield
            matrix_ainv_tmp = self._modelstorage.get_model()['matrix_ainv']
            theta_tmp = self._modelstorage.get_model()['theta']

            # The recommended action should maximize the Linear UCB.
            estimated_reward = {}
            uncertainty = {}
            score = {}
            for action_id in self.action_ids:
                context_tmp = np.array(context[action_id])
                estimated_reward[action_id] = float(
                    np.dot(context_tmp, theta_tmp[action_id]))
                uncertainty[action_id] = float(self.alpha * np.sqrt(
                    np.dot(np.dot(context_tmp, matrix_ainv_tmp[action_id]),
                           context_tmp.T)))
                score[action_id] = estimated_reward[action_id] + \
                    uncertainty[action_id]
            yield estimated_reward, uncertainty, score

        raise StopIteration

    def get_action(self, context, n_actions=1):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {action_id: context} of different actions.

        n_actions: int
            Number of actions wanted to recommend users.

        Returns
        -------
        history_id : int
            The history id of the action.

        action_recommendation : list of dictionaries
            Each dictionary contains {Action object, estimated_reward,
            uncertainty}
        """

        if context is None:
            raise ValueError("LinUCB requires contexts for all actions!")

        if self.linucb_ is None:
            self.linucb_ = self.linucb
            six.next(self.linucb_)
            estimated_reward, uncertainty, score = self.linucb_.send(context)
        else:
            six.next(self.linucb_)
            estimated_reward, uncertainty, score = self.linucb_.send(context)

        action_recommendation = []
        action_recommendation_ids = sorted(score, key=score.get,
                                           reverse=True)[:n_actions]
        for action_id in action_recommendation_ids:
            action_id = int(action_id)
            action = [action for action in self._actions
                      if action.action_id == action_id][0]
            action_recommendation.append({
                'action': action,
                'estimated_reward': estimated_reward[action_id],
                'uncertainty': uncertainty[action_id],
                'score': score[action_id]})

        history_id = self._historystorage.add_history(context,
                                                      action_recommendation,
                                                      reward=None)
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

        context = self._historystorage.unrewarded_histories[history_id].context

        # Update the model
        matrix_a = self._modelstorage.get_model()['matrix_a']
        matrix_ainv = self._modelstorage.get_model()['matrix_ainv']
        b = self._modelstorage.get_model()['b']
        theta = self._modelstorage.get_model()['theta']

        for action_id, reward_tmp in rewards.items():
            context_tmp = np.matrix(context[action_id])
            matrix_a[action_id] += np.dot(context_tmp.T, context_tmp)
            matrix_ainv[action_id] = np.linalg.solve(
                matrix_a[action_id], np.identity(self.d))
            b[action_id] += reward_tmp * context_tmp.T
            theta[action_id] = np.dot(matrix_ainv[action_id], b[action_id])
        self._modelstorage.save_model({
            'matrix_a': matrix_a, 'matrix_ainv': matrix_ainv,
            'b': b, 'theta': theta})

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

        matrix_a = self._modelstorage.get_model()['matrix_a']
        matrix_ainv = self._modelstorage.get_model()['matrix_ainv']
        b = self._modelstorage.get_model()['b']
        theta = self._modelstorage.get_model()['theta']

        for action_id in action_ids:
            matrix_a[action_id] = np.identity(self.d)
            matrix_ainv[action_id] = np.identity(self.d)
            b[action_id] = np.zeros((self.d, 1))
            theta[action_id] = np.zeros((self.d, 1))

        self._modelstorage.save_model({
            'matrix_a': matrix_a, 'matrix_ainv': matrix_ainv,
            'b': b, 'theta': theta})
