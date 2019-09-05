"""LinUCB with Hybrid Linear Models

This module contains a class that implements LinUCB with hybrid linear model,
a contextual bandit algorithm assuming the reward function is a linear function
of the context."""

import logging
import six
import numpy as np
from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)
EPS = 1e-04


class HybLinUCB(BaseBandit):
    """LinUCB with Hybrid Linear Models

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

        shared_context_dimension: int
            The dimension of the context shared between actions.

        unshared_context_dimension: int
            The dimension of the context not shared between actions.

        is_shared_context_first: bool
            Whether the shared context comes first in the input.

        alpha: float
            The constant determines the width of the upper confidence bound.

        References
        ----------
        .. [1]  Lihong Li, et al. "A Contextual-Bandit Approach to Personalized
                News Article Recommendation." In Proceedings of the 19th
                International Conference on World Wide Web (WWW), 2010."""

    def __init__(self, history_storage, model_storage, action_storage, recommendation_cls=None,
                 shared_context_dimension=64, unshared_context_dimension=64, is_shared_context_first=True, alpha=0.5):

        super(HybLinUCB, self).__init__(history_storage, model_storage, action_storage, recommendation_cls)

        self.alpha = alpha
        self.shared_context_dimension = shared_context_dimension
        self.unshared_context_dimension = unshared_context_dimension

        if is_shared_context_first:
            self.shared_context_boundary_idx = [0, self.shared_context_dimension]
            self.unshared_context_boundary_idx = [self.shared_context_dimension,
                                                  self.shared_context_dimension + self.unshared_context_dimension]
        else:
            self.shared_context_boundary_idx = [self.unshared_context_dimension,
                                                self.shared_context_dimension + self.unshared_context_dimension]
            self.unshared_context_boundary_idx = [0, self.unshared_context_dimension]

        model = {
            'A0': np.identity(self.shared_context_dimension),
            'b0': np.zeros((self.shared_context_dimension, 1)),
            'beta': np.zeros((self.shared_context_dimension, 1)),
            'A': {},
            'B': {},
            'b': {},
            'theta': {}
        }

        for action_id in self._action_storage.iterids():
            self._init_action_model(model, action_id)

        self._model_storage.save_model(model)

    def _init_action_model(self, model, action_id=None):
        model['A'][action_id] = np.identity(self.unshared_context_dimension)
        model['B'][action_id] = np.zeros((self.unshared_context_dimension, self.shared_context_dimension))
        model['b'][action_id] = np.zeros((self.unshared_context_dimension, 1))
        model['theta'][action_id] = np.zeros((self.unshared_context_dimension, 1))

    def _linucb_score(self, shared_context, unshared_context):
        """Hybrid LinUCB algorithm."""
        model = self._model_storage.get_model()

        A0 = model['A0']
        b0 = model['b0']
        A = model['A']
        B = model['B']
        b = model['b']

        A0_inv = np.linalg.inv(A0)
        beta = A0_inv.dot(b0)

        theta = {}
        estimated_reward = {}
        uncertainty = {}
        score = {}

        for action_id in self._action_storage.iterids():
            A_inv = np.linalg.inv(A[action_id])
            theta[action_id] = A_inv.dot(b[action_id] - B[action_id].dot(beta))
            s = max(np.linalg.multi_dot([shared_context[action_id].T, A0_inv, shared_context[action_id]])
                    - 2.0 * np.linalg.multi_dot([shared_context[action_id].T, A0_inv, B[action_id].T, A_inv,
                                                 unshared_context[action_id]])
                    + np.linalg.multi_dot([unshared_context[action_id].T, A_inv, unshared_context[action_id]])
                    + np.linalg.multi_dot([unshared_context[action_id].T, A_inv, B[action_id], A0_inv, B[action_id].T,
                                           A_inv, unshared_context[action_id]]), EPS)

            estimated_reward[action_id] = (shared_context[action_id].T.dot(beta)
                                           + unshared_context[action_id].T.dot(theta[action_id])).item()
            uncertainty[action_id] = (self.alpha * np.sqrt(s)).item()
            score[action_id] = estimated_reward[action_id] + uncertainty[action_id]

        self._model_storage.save_model({
            'A0': A0,
            'b0': b0,
            'beta': beta,
            'A': A,
            'B': B,
            'b': b,
            'theta': theta
        })
        return estimated_reward, uncertainty, score

    def get_action(self, context, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : dict
            Contexts {action_id: context} of different actions.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty, score}."""
        if self._action_storage.count() == 0:
            return self._get_action_with_empty_action_storage(context, n_actions)

        if not isinstance(context, dict):
            raise ValueError('LinUCB requires context dict for all actions!')

        if n_actions == -1:
            n_actions = self._action_storage.count()

        shared_context = {}
        unshared_context = {}

        for action_id in self._action_storage.iterids():
            action_context = np.reshape(context[action_id], (-1, 1))
            shared_context[action_id] = action_context[
                                        self.shared_context_boundary_idx[0]:self.shared_context_boundary_idx[1]]
            unshared_context[action_id] = action_context[
                                          self.unshared_context_boundary_idx[0]:self.unshared_context_boundary_idx[1]]

        estimated_reward, uncertainty, score = self._linucb_score(shared_context, unshared_context)

        if n_actions is None:
            recommendation_id = max(score, key=score.get)
            recommendations = self._recommendation_cls(
                action=self._action_storage.get(recommendation_id),
                estimated_reward=estimated_reward[recommendation_id],
                uncertainty=uncertainty[recommendation_id],
                score=score[recommendation_id]
            )
        else:
            recommendation_ids = sorted(score, key=score.get, reverse=True)[:n_actions]
            recommendations = []
            for action_id in recommendation_ids:
                recommendations.append(self._recommendation_cls(
                    action=self._action_storage.get(action_id),
                    estimated_reward=estimated_reward[action_id],
                    uncertainty=uncertainty[action_id],
                    score=score[action_id]
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
            The dictionary {action_id, reward}, where reward is a float."""
        context = self._history_storage.get_unrewarded_history(history_id).context
        model = self._model_storage.get_model()

        A0 = model['A0']
        b0 = model['b0']
        beta = model['beta']
        A = model['A']
        B = model['B']
        b = model['b']
        theta = model['theta']

        shared_context = {}
        unshared_context = {}

        for action_id, reward in six.viewitems(rewards):
            action_context = np.reshape(context[action_id], (-1, 1))
            shared_context[action_id] = action_context[
                                        self.shared_context_boundary_idx[0]:self.shared_context_boundary_idx[1]]
            unshared_context[action_id] = action_context[
                                          self.unshared_context_boundary_idx[0]:self.unshared_context_boundary_idx[1]]

            A_inv = np.linalg.inv(A[action_id])
            A0 += np.linalg.multi_dot([B[action_id].T, A_inv, B[action_id]])
            b0 += np.linalg.multi_dot([B[action_id].T, A_inv, b[action_id]])
            A[action_id] += unshared_context[action_id].dot(unshared_context[action_id].T)
            B[action_id] += unshared_context[action_id].dot(shared_context[action_id].T)
            b[action_id] += reward * unshared_context[action_id]
            A0 += shared_context[action_id].dot(shared_context[action_id].T) - np.linalg.multi_dot([
                B[action_id].T, A_inv, B[action_id]])
            b0 += reward * shared_context[action_id] - np.linalg.multi_dot([
                B[action_id].T, A_inv, b[action_id]])

        self._model_storage.save_model({
            'A0': A0,
            'b0': b0,
            'beta': beta,
            'A': A,
            'B': B,
            'b': b,
            'theta': theta
        })

        self._history_storage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation."""
        new_action_ids = self._action_storage.add(actions)
        model = self._model_storage.get_model()

        for action_id in new_action_ids:
            self._init_action_model(model, action_id)

        self._model_storage.save_model(model)

    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove."""
        model = self._model_storage.get_model()

        del model['A'][action_id]
        del model['B'][action_id]
        del model['b'][action_id]
        del model['theta'][action_id]

        self._model_storage.save_model(model)
        self._action_storage.remove(action_id)
