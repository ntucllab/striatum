import logging
import six
import numpy as np
from joblib import Parallel, delayed
from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class HybridLinUCB(BaseBandit):

    def __init__(self, history_storage, model_storage, action_storage, recommendation_cls=None,
                 shared_context_dimension=64, unshared_context_dimension=64, alpha=0.5, n_jobs=None):

        super(HybridLinUCB, self).__init__(history_storage, model_storage, action_storage, recommendation_cls)

        self.alpha = alpha
        self.shared_context_dimension = shared_context_dimension
        self.unshared_context_dimension = unshared_context_dimension
        self.n_jobs = n_jobs

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

        action_ids = list(self._action_storage.iterids())

        res = Parallel(n_jobs=self.n_jobs)(delayed(self._get_score_of_each_action)(
            A0_inv, beta, A[action_id], B[action_id], b[action_id]
            , shared_context[action_id], unshared_context[action_id]
        ) for action_id in action_ids)

        for action_id, (action_theta, action_estimated_reward, action_uncertainty, action_score
                        ) in zip(action_ids, res):
            theta[action_id] = action_theta
            estimated_reward[action_id] = action_estimated_reward
            uncertainty[action_id] = action_uncertainty
            score[action_id] = action_score

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

    def _get_score_of_each_action(self, A0_inv, beta, A, B, b, shared_context, unshared_context):
        A_inv = np.linalg.inv(A)
        theta = A_inv.dot(b - B.dot(beta))
        s = np.linalg.multi_dot([shared_context.T, A0_inv, shared_context]) - 2.0 * np.linalg.multi_dot(
            [shared_context.T, A0_inv, B.T, A_inv, unshared_context]) + np.linalg.multi_dot(
            [unshared_context.T, A_inv, unshared_context]) + np.linalg.multi_dot(
            [unshared_context.T, A_inv, B, A0_inv, B.T, A_inv, unshared_context])

        estimated_reward = (shared_context.T.dot(beta) + unshared_context.T.dot(theta)).item()
        uncertainty = (self.alpha * np.sqrt(s)).item()
        score = estimated_reward + uncertainty
        return theta, estimated_reward, uncertainty, score

    def get_action(self, context, n_actions=None):
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
            shared_context[action_id] = action_context[:self.shared_context_dimension]
            unshared_context[action_id] = action_context[
                                          self.shared_context_dimension:
                                          self.shared_context_dimension + self.unshared_context_dimension]

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
            shared_context[action_id] = action_context[:self.shared_context_dimension]
            unshared_context[action_id] = action_context[
                                          self.shared_context_dimension:
                                          self.shared_context_dimension + self.unshared_context_dimension]

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
        new_action_ids = self._action_storage.add(actions)
        model = self._model_storage.get_model()

        for action_id in new_action_ids:
            self._init_action_model(model, action_id)

        self._model_storage.save_model(model)

    def remove_action(self, action_id):
        model = self._model_storage.get_model()

        del model['A'][action_id]
        del model['B'][action_id]
        del model['b'][action_id]
        del model['theta'][action_id]

        self._model_storage.save_model(model)
        self._action_storage.remove(action_id)
