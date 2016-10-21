""" Thompson Sampling with Linear Payoff
In This module contains a class that implements Thompson Sampling with Linear
Payoff. Thompson Sampling with linear payoff is a contexutal multi-armed bandit
algorithm which assume the underlying relationship between rewards and contexts
is linear. The sampling method is used to balance the exploration and
exploitation. Please check the reference for more details.
"""
import logging
import six
from six.moves import zip

import numpy as np

from .bandit import BaseBandit
from ..utils import get_random_state

LOGGER = logging.getLogger(__name__)


class LinThompSamp(BaseBandit):
    r"""Thompson sampling with linear payoff.

    Parameters
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    delta: float, 0 < delta < 1
        With probability 1 - delta, LinThompSamp satisfies the theoretical
        regret bound.

    R: float, R >= 0
        Assume that the residual  :math:`ri(t) - bi(t)^T \hat{\mu}`
        is R-sub-gaussian. In this case, R^2 represents the variance for
        residuals of the linear model :math:`bi(t)^T`.

    epsilon: float, 0 < epsilon < 1
        A  parameter  used  by  the  Thompson Sampling algorithm.
        If the total trials T is known, we can choose epsilon = 1/ln(T).

    Attributes
    ----------
    linthomp\_ : 'linthomp' object instance
        The contextual bandit algorithm instances

    References
    ----------
    .. [1]  Shipra Agrawal, and Navin Goyal. "Thompson Sampling for Contextual
            Bandits with Linear Payoffs." Advances in Neural Information
            Processing Systems 24. 2011.
    """

    def __init__(self, history_storage, model_storage, action_storage,
                 context_dimension=128, delta=0.5, R=0.01, epsilon=0.5,
                 random_state=None):
        super(LinThompSamp, self).__init__(history_storage, model_storage,
                                           action_storage)
        self.random_state = get_random_state(random_state)
        self.context_dimension = context_dimension

        # 0 < delta < 1
        if not isinstance(delta, float):
            raise ValueError("delta should be float")
        elif (delta < 0) or (delta >= 1):
            raise ValueError("delta should be in (0, 1]")
        else:
            self.delta = delta

        # R > 0
        if not isinstance(R, float):
            raise ValueError("R should be float")
        elif R <= 0:
            raise ValueError("R should be positive")
        else:
            self.R = R  # pylint: disable=invalid-name

        # 0 < epsilon < 1
        if not isinstance(epsilon, float):
            raise ValueError("epsilon should be float")
        elif (epsilon < 0) or (epsilon > 1):
            raise ValueError("epsilon should be in (0, 1)")
        else:
            self.epsilon = epsilon

        # model initialization
        B = np.identity(self.context_dimension)  # pylint: disable=invalid-name
        mu_hat = np.zeros(shape=(self.context_dimension, 1))
        f = np.zeros(shape=(self.context_dimension, 1))
        self._model_storage.save_model({'B': B, 'mu_hat': mu_hat, 'f': f})

    def _linthompsamp_score(self, context):
        """Thompson Sampling"""
        action_ids = list(six.viewkeys(context))
        context_array = np.asarray([context[action_id]
                                    for action_id in action_ids])
        model = self._model_storage.get_model()
        B = model['B']  # pylint: disable=invalid-name
        mu_hat = model['mu_hat']
        v = self.R * np.sqrt(24 / self.epsilon
                             * self.context_dimension
                             * np.log(1 / self.delta))
        mu_tilde = self.random_state.multivariate_normal(
            mu_hat.flat, v**2 * np.linalg.inv(B))[..., np.newaxis]
        estimated_reward_array = context_array.dot(mu_hat)
        score_array = context_array.dot(mu_tilde)

        estimated_reward_dict = {}
        uncertainty_dict = {}
        score_dict = {}
        for action_id, estimated_reward, score in zip(
                action_ids, estimated_reward_array, score_array):
            estimated_reward_dict[action_id] = float(estimated_reward)
            score_dict[action_id] = float(score)
            uncertainty_dict[action_id] = float(score - estimated_reward)
        return estimated_reward_dict, uncertainty_dict, score_dict

    def get_action(self, context, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {action_id: context} of different actions.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        action_recommendation : list of dictionaries
            In each dictionary, it contains {Action object,
            estimated_reward, uncertainty}.
        """
        if not isinstance(context, dict):
            raise ValueError(
                "LinThompSamp requires context dict for all actions!")
        if n_actions == -1:
            n_actions = self._action_storage.count()

        estimated_reward, uncertainty, score = self._linthompsamp_score(context)

        if n_actions is None:
            action_recommendation_id = max(score, key=score.get)
            action_recommendation = {
                'action': self._action_storage.get(action_recommendation_id),
                'estimated_reward': estimated_reward[action_recommendation_id],
                'uncertainty': uncertainty[action_recommendation_id],
                'score': score[action_recommendation_id],
            }
        else:
            action_recommendation_ids = sorted(score, key=score.get,
                                               reverse=True)[:n_actions]
            action_recommendation = []  # pylint: disable=redefined-variable-type
            for action_id in action_recommendation_ids:
                action_recommendation.append({
                    'action': self._action_storage.get(action_id),
                    'estimated_reward': estimated_reward[action_id],
                    'uncertainty': uncertainty[action_id],
                    'score': score[action_id],
                })

        history_id = self._history_storage.add_history(
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
        context = (self._history_storage
                   .get_unrewarded_history(history_id)
                   .context)

        # Update the model
        model = self._model_storage.get_model()
        B = model['B']  # pylint: disable=invalid-name
        f = model['f']

        for action_id, reward in six.viewitems(rewards):
            context_t = np.reshape(context[action_id], (-1, 1))
            B += context_t.dot(context_t.T)  # pylint: disable=invalid-name
            f += reward * context_t
            mu_hat = np.linalg.inv(B).dot(f)
        self._model_storage.save_model({'B': B, 'mu_hat': mu_hat, 'f': f})

        # Update the history
        self._history_storage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action oBjects for recommendation
        """
        self._action_storage.add(actions)
