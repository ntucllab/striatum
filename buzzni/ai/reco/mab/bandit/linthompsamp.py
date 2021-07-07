""" Thompson Sampling with Linear Payoff
In This module contains a class that implements Thompson Sampling with Linear
Payoff. Thompson Sampling with linear payoff is a contexutal multi-armed bandit
algorithm which assume the underlying relationship between rewards and contexts
is linear. The sampling method is used to balance the exploration and
exploitation. Please check the reference for more details.
"""
import logging

import numpy as np

from buzzni.ai.reco.mab.bandit.bandit import LinearBandit
from buzzni.ai.reco.mab.utils import get_random_state

LOGGER = logging.getLogger(__name__)


class LinThompSamp(LinearBandit):
    r"""Thompson sampling with linear payoff.

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

    alpha: float
            The constant determines the width of the upper confidence bound. (follow tf-agents implement)

    gamma: float
        forgetting factor in [0.0, 1.0]. When set to 1.0, the algorithm does not forget.

    tikhonov_weight: float
        tikhonov regularization term.

    random_state: {int, np.random.RandomState} (default: None)
        If int, np.random.RandomState will used it as seed. If None, a random
        seed will be used.

    References
    ----------
    .. [1]  Shipra Agrawal, and Navin Goyal. "Thompson Sampling for Contextual
            Bandits with Linear Payoffs." Advances in Neural Information
            Processing Systems 24. 2011.
    """

    def __init__(self, history_storage, model_storage, action_storage,
                 recommendation_cls=None, context_dimension=128,
                 alpha=1.0, gamma=1.0, tikhonov_weight=1.0, random_state=None):
        self.alpha = alpha
        self.tikhonov_weight = tikhonov_weight
        self.random_state = get_random_state(random_state)

        if gamma < 0.0 or gamma > 1.0:
            raise ValueError('Forgetting factor `gamma` must be in [0.0, 1.0].')

        super(LinThompSamp, self).__init__(model_name="LinThompSamp",
                                           history_storage=history_storage,
                                           model_storage=model_storage,
                                           action_storage=action_storage,
                                           context_dimension=context_dimension,
                                           gamma=gamma,
                                           recommendation_cls=recommendation_cls)

    def _score(self, context, model):
        """disjoint LINUCB algorithm.
        """
        '''
        tf-agents 구현체 따름
        # 계산
            - https://github.com/tensorflow/agents/blob/3719a8780d7054984ddc528dbfe99b51b9f10062/tf_agents/bandits/policies/linear_bandit_policy.py#L251
        '''

        A = model['A']  # pylint: disable=invalid-name
        b = model['b']  # pylint: disable=invalid-name

        # The recommended actions should maximize the Linear UCB.
        estimated_reward = {}
        uncertainty = {}
        score = {}
        for action_id in self._action_storage.iterids():
            action_context = np.reshape(context[action_id], (-1, 1))  # user feature: (dim, 1)

            A_inv_x = np.linalg.inv(A[action_id] + self.tikhonov_weight * np.identity(self.context_dimension)).dot(
                action_context)  # (dim, 1)
            estimated_reward[action_id] = float(b[action_id].T.dot(A_inv_x))
            uncertainty[action_id] = float(np.sqrt(action_context.T.dot(A_inv_x)))
            score[action_id] = np.random.normal(loc=estimated_reward[action_id],
                                                scale=self.alpha * uncertainty[action_id])
        return estimated_reward, uncertainty, score
