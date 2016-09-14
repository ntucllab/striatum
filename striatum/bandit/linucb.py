"""LinUCB with Disjoint Linear Models

This module contains a class that implements LinUCB with disjoint linear model,
a contextual bandit algorithm assuming the reward function is a linear function
of the context.
"""
import logging

import six
import numpy as np

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

    context_dimension: int
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

    def __init__(self, actions, historystorage, modelstorage, alpha,
                 context_dimension=1):
        super(LinUCB, self).__init__(historystorage, modelstorage, actions)
        self.alpha = alpha
        self.context_dimension = context_dimension

        # Initialize LinUCB Model Parameters
        model = {
            # dictionary - For any action a in actions,
            # A[a] = (DaT*Da + I) the ridge reg solution
            'A': {},
            # dictionary - The inverse of each A[a] for action a
            # in actions
            'A_inv': {},
            # dictionary - The cumulative return of action a, given the
            # context xt.
            'b': {},
            # dictionary - The coefficient vector of actiona with
            # linear model b = dot(xt, theta)
            'theta': {},
        }
        for action_id in self.action_ids:
            self._init_action_model(model, action_id)

        self._modelstorage.save_model(model)

    def _init_action_model(self, model, action_id):
        model['A'][action_id] = np.identity(self.context_dimension)
        model['A_inv'][action_id] = np.identity(self.context_dimension)
        model['b'][action_id] = np.zeros((self.context_dimension, 1))
        model['theta'][action_id] = np.zeros((self.context_dimension, 1))

    def _linucb_score(self, context):
        """disjoint LINUCB algorithm.
        """
        model = self._modelstorage.get_model()
        A_inv = model['A_inv']  # pylint: disable=invalid-name
        theta = model['theta']

        # The recommended actions should maximize the Linear UCB.
        estimated_reward = {}
        uncertainty = {}
        score = {}
        for action_id in self.action_ids:
            action_context = np.reshape(context[action_id], (-1, 1))
            estimated_reward[action_id] = float(
                theta[action_id].T.dot(action_context))
            uncertainty[action_id] = float(
                self.alpha * np.sqrt(action_context.T
                                     .dot(A_inv[action_id])
                                     .dot(action_context)))
            score[action_id] = (estimated_reward[action_id]
                                + uncertainty[action_id])
        return estimated_reward, uncertainty, score

    def get_action(self, context, n_actions=1):
        """Return the action to perform

        Parameters
        ----------
        context : dict
            Contexts {action_id: context} of different actions.

        n_actions: int
            Number of actions wanted to recommend users.

        Returns
        -------
        history_id : int
            The history id of the action.

        action_recommendation : list of dict
            Each dict contains {Action object, estimated_reward,
            uncertainty}
        """
        if not isinstance(context, dict):
            raise ValueError("LinUCB requires context dict for all actions!")

        estimated_reward, uncertainty, score = self._linucb_score(context)

        action_recommendation = []
        action_recommendation_ids = sorted(score, key=score.get,
                                           reverse=True)[:n_actions]

        for action_id in action_recommendation_ids:
            action = self.get_action_with_id(action_id)
            action_recommendation.append({
                'action': action,
                'estimated_reward': estimated_reward[action_id],
                'uncertainty': uncertainty[action_id],
                'score': score[action_id],
            })

        history_id = self._historystorage.add_history(
            context, action_recommendation, reward=None)
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
        context = (self._historystorage
                   .get_unrewarded_history(history_id)
                   .context)

        # Update the model
        model = self._modelstorage.get_model()
        A = model['A']  # pylint: disable=invalid-name
        A_inv = model['A_inv']  # pylint: disable=invalid-name
        b = model['b']
        theta = model['theta']

        for action_id, reward in six.viewitems(rewards):
            action_context = np.reshape(context[action_id], (-1, 1))
            A[action_id] += action_context.dot(action_context.T)
            A_inv[action_id] = np.linalg.inv(A[action_id])
            b[action_id] += reward * action_context
            theta[action_id] = A_inv[action_id].dot(b[action_id])
        self._modelstorage.save_model({
            'A': A,
            'A_inv': A_inv,
            'b': b,
            'theta': theta,
        })

        # Update the history
        self._historystorage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation
        """
        self._actions.extend(actions)
        model = self._modelstorage.get_model()

        for action in actions:
            self._init_action_model(model, action.action_id)

        self._modelstorage.save_model(model)
