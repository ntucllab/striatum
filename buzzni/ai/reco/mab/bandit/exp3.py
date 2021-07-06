""" Exp3: Exponential-weight algorithm for Exploration and Exploitation
This module contains a class that implements EXP3, a bandit algorithm that
randomly choose an action according to a learned probability distribution.
"""
import logging

import numpy as np
import six

from .bandit import BaseBandit
from ..utils import get_random_state

LOGGER = logging.getLogger(__name__)


class Exp3(BaseBandit):
    r"""Exp3 algorithm.

    Parameters
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    recommendation_cls : class (default: None)
        The class used to initiate the recommendations. If None, then use
        default Recommendation class.

    gamma: float, 0 < gamma <= 1
        The parameter used to control the minimum chosen probability for each
        action.

    random_state: {int, np.random.RandomState} (default: None)
        If int, np.random.RandomState will used it as seed. If None, a random
        seed will be used.

    References
    ----------
    .. [1]  Peter Auer, Nicolo Cesa-Bianchi, et al. "The non-stochastic
            multi-armed bandit problem ." SIAM Journal of Computing. 2002.
    """

    def __init__(self, history_storage, model_storage, action_storage,
                 recommendation_cls=None, gamma=0.3, random_state=None):
        super(Exp3, self).__init__(history_storage, model_storage,
                                   action_storage, recommendation_cls)
        self.random_state = get_random_state(random_state)

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
        w = {action_id: 1. for action_id in self._action_storage.iterids()}
        self._model_storage.save_model({'w': w})

    def _exp3_probs(self):
        """Exp3 algorithm.
        """
        w = self._model_storage.get_model()['w']
        w_sum = sum(six.viewvalues(w))

        probs = {}
        n_actions = self._action_storage.count()
        for action_id in self._action_storage.iterids():
            probs[action_id] = ((1 - self.gamma) * w[action_id]
                                / w_sum
                                + self.gamma / n_actions)

        return probs

    def get_action(self, context=None, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context available.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        if self._action_storage.count() == 0:
            return self._get_action_with_empty_action_storage(context,
                                                              n_actions)

        probs = self._exp3_probs()
        if n_actions == -1:
            n_actions = self._action_storage.count()

        action_ids = list(six.viewkeys(probs))
        prob_array = np.asarray([probs[action_id]
                                 for action_id in action_ids])
        recommendation_ids = self.random_state.choice(
            action_ids, size=n_actions, p=prob_array, replace=False)

        if n_actions is None:
            recommendations = self._recommendation_cls(
                action=self._action_storage.get(recommendation_ids),
                estimated_reward=probs[recommendation_ids],
                uncertainty=probs[recommendation_ids],
                score=probs[recommendation_ids],
            )
        else:
            recommendations = []  # pylint: disable=redefined-variable-type
            for action_id in recommendation_ids:
                recommendations.append(self._recommendation_cls(
                    action=self._action_storage.get(action_id),
                    estimated_reward=probs[action_id],
                    uncertainty=probs[action_id],
                    score=probs[action_id],
                ))

        history_id = self._history_storage.add_history(context, recommendations)
        return history_id, recommendations

    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

            Parameters
            ----------
            history_id : int
                The history id of the action to reward.

            rewards : dictionary
                The dictionary {action_id, reward}, where reward is a float.
        """
        w = self._model_storage.get_model()['w']
        history = self._history_storage.get_unrewarded_history(history_id)
        n_actions = self._action_storage.count()
        if isinstance(history.recommendations, list):
            recommendations = history.recommendations
        else:
            recommendations = [history.recommendations]
        probs = {rec.action.id: rec.estimated_reward
                 for rec in recommendations
                 if rec.action.id in rewards}

        # Update the model
        for action_id, reward in rewards.items():
            w[action_id] *= np.exp(
                self.gamma * (reward / probs[action_id]) / n_actions)

        self._model_storage.save_model({'w': w})

        # Update the history
        self._history_storage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation
        """
        self._action_storage.add(actions)

        w = self._model_storage.get_model()['w']

        for action in actions:
            w[action.id] = 1.0  # weight vector

        self._model_storage.save_model({'w': w})

    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        w = self._model_storage.get_model()['w']
        del w[action_id]
        self._model_storage.save_model({'w': w})
        self._action_storage.remove(action_id)
