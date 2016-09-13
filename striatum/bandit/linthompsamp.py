""" Thompson Sampling with Linear Payoff
In This module contains a class that implements Thompson Sampling with Linear
Payoff. Thompson Sampling with linear payoff is a contexutal multi-armed bandit
algorithm which assume the underlying relationship between rewards and contexts
is linear. The sampling method is used to balance the exploration and
exploitation. Please check the reference for more details.
"""

import logging
import six

import numpy as np

from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class LinThompSamp(BaseBandit):
    r"""Thompson sampling with linear payoff.

    Parameters
    ----------
    actions : list of Action objects
        List of actions to be chosen from.

    historystorage: a HistoryStorage object
        The place where we store the histories of contexts and rewards.

    modelstorage: a ModelStorage object
        The place where we store the model parameters.

    delta: float, 0 < delta < 1
        With probability 1 - delta, LinThompSamp satisfies the theoretical
        regret bound.

    r: float, r >= 0
        Assume that the residual  :math:`ri(t) - bi(t)^T \hat{\mu}`
        is r-sub-gaussian. In this case, r^2 represents the variance for
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

    def __init__(self, actions, historystorage, modelstorage,
                 context_dimension, delta=0.5, r=0.5, epsilon=0.1,
                 random_state=None):
        super(LinThompSamp, self).__init__(historystorage, modelstorage,
                                           actions)
        if random_state is None:
            random_state = np.random.RandomState()
        elif not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(seed=random_state)
        self.random_state = random_state
        self.linthompsamp_ = None
        self.context_dimension = context_dimension

        # 0 < delta < 1
        if not isinstance(delta, float):
            raise ValueError("delta should be float")
        elif (delta < 0) or (delta >= 1):
            raise ValueError("delta should be in (0, 1]")
        else:
            self.delta = delta

        # R > 0
        if not isinstance(r, float):
            raise ValueError("R should be float")
        elif r <= 0:
            raise ValueError("R should be positive")
        else:
            self.R = r  # pylint: disable=invalid-name

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
        self._modelstorage.save_model({'B': B, 'mu_hat': mu_hat, 'f': f})

    @property
    def linthompsamp(self):
        """linThompSamp generator"""

        while True:
            context = yield
            action_ids = list(context.keys())
            context_tmp = np.matrix(list(context.values()))
            B = self._modelstorage.get_model()['B']  # pylint: disable=invalid-name
            mu_hat = self._modelstorage.get_model()['mu_hat']
            v = self.R * np.sqrt(24 / self.epsilon * self.context_dimension *
                                 np.log(1 / self.delta))
            mu = self.random_state.multivariate_normal(
                np.array(mu_hat.T)[0], v**2 * np.linalg.inv(B), 1)[0]
            estimated_reward_tmp = np.dot(context_tmp, np.array(mu_hat)).tolist()
            score_tmp = np.dot(context_tmp, np.array(mu)).tolist()[0]

            estimated_reward = {}
            uncertainty = {}
            score = {}
            for i in range(len(action_ids)):
                estimated_reward[action_ids[i]] = float(
                    estimated_reward_tmp[i][0])
                score[action_ids[i]] = float(score_tmp[i])
                uncertainty[action_ids[i]] = score[action_ids[i]] \
                    - estimated_reward[action_ids[i]]
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
            In each dictionary, it contains {Action object,
            estimated_reward, uncertainty}.
        """

        if context is None:
            raise ValueError("LinThompSamp requires contexts for all actions!")

        if self.linthompsamp_ is None:
            self.linthompsamp_ = self.linthompsamp
            six.next(self.linthompsamp_)
            estimated_reward, uncertainty, score = self.linthompsamp_.send(
                context)
        else:
            six.next(self.linthompsamp_)
            estimated_reward, uncertainty, score = self.linthompsamp_.send(
                context)

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
                'score': score[action_id],
            })

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
        B = self._modelstorage.get_model()['B']  # pylint: disable=invalid-name
        f = self._modelstorage.get_model()['f']

        for action_id, reward_tmp in rewards.items():
            context_tmp = np.matrix(context[action_id])
            B += np.dot(context_tmp.T, context_tmp)  # pylint: disable=invalid-name
            f += reward_tmp * context_tmp.T
            mu_hat = np.dot(np.linalg.inv(B), f)
        self._modelstorage.save_model({'B': B, 'mu_hat': mu_hat, 'f': f})

        # Update the history
        self._historystorage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action oBjects for recommendation
        """

        self._actions.extend(actions)
