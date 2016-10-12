""" EXP4.P: An extention to exponential-weight algorithm for exploration and
exploitation. This module contains a class that implements EXP4.P, a contextual
bandit algorithm with expert advice.
"""

import logging

import six
from six.moves import zip
import numpy as np

from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class Exp4P(BaseBandit):
    r"""Exp4.P with pre-trained supervised learning algorithm.

    Parameters
    ----------
    actions : list of Action objects
        List of actions to be chosen from.

    historystorage: a HistoryStorage object
        The place where we store the histories of contexts and rewards.

    modelstorage: a ModelStorage object
        The place where we store the model parameters.

    delta: float, 0 < delta <= 1
        With probability 1 - delta, LinThompSamp satisfies the theoretical
        regret bound.

    p_min: float, 0 < p_min < 1/k
        The minimum probability to choose each action.

    Attributes
    ----------
    exp4p\_ : 'exp4p' object instance
        The contextual bandit algorithm instances

    References
    ----------
    .. [1]  Beygelzimer, Alina, et al. "Contextual bandit algorithms with
            supervised learning guarantees." International Conference on
            Artificial Intelligence and Statistics (AISTATS). 2011u.
    """

    def __init__(self, actions, historystorage, modelstorage, delta=0.1,
                 p_min=None, max_rounds=10000):
        super(Exp4P, self).__init__(historystorage, modelstorage, actions)
        self.n_total = 0
        # number of actions (i.e. K in the paper)
        self.n_actions = len(self._actions)
        self.exp4p_ = None
        self.max_rounds = max_rounds

        # delta > 0
        if not isinstance(delta, float):
            raise ValueError("delta should be float, the one"
                             "given is: %f" % p_min)
        self.delta = delta

        # p_min in [0, 1/k]
        if p_min is None:
            self.p_min = np.sqrt(np.log(10) / self.n_actions / self.max_rounds)
        elif not isinstance(p_min, float):
            raise ValueError("p_min should be float, the one"
                             "given is: %f" % p_min)
        elif (p_min < 0) or (p_min > (1. / self.n_actions)):
            raise ValueError("p_min should be in [0, 1/k], the one"
                             "given is: %f" % p_min)
        else:
            self.p_min = p_min

        # Initialize the model storage

        model = {
            # probability distribution for action recommendation
            'action_probs': {},
            # weight vector for each expert
            'w': {},
        }
        self._modelstorage.save_model(model)

    def _exp4p_score(self, context):
        """The main part of Exp4.P.
        """
        advisor_ids = list(six.viewkeys(context))

        w = self._modelstorage.get_model()['w']
        if len(w) == 0:
            for i in advisor_ids:
                w[i] = 1
        w_sum = sum(six.viewvalues(w))

        action_probs_list = []
        for action_id in self.action_ids:
            weighted_exp = [w[advisor_id] * context[advisor_id][action_id]
                            for advisor_id in advisor_ids]
            prob_vector = np.sum(weighted_exp) / w_sum
            action_probs_list.append((1 - self.n_actions * self.p_min)
                                     * prob_vector
                                     + self.p_min)
        action_probs_list = np.asarray(action_probs_list)
        action_probs_list /= action_probs_list.sum()

        estimated_reward = {}
        uncertainty = {}
        score = {}
        for action_id, action_prob in zip(self.action_ids, action_probs_list):
            estimated_reward[action_id] = action_prob
            uncertainty[action_id] = 0
            score[action_id] = action_prob
        self._modelstorage.save_model(
            {'action_probs': estimated_reward, 'w': w})

        return estimated_reward, uncertainty, score

    def get_action(self, context=None, n_actions=1):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {expert_id: {action_id: expert_prediction}} of
            different actions.

        n_actions: int
            Number of actions wanted to recommend users.

        Returns
        -------
        history_id : int
            The history id of the action.

        action_recommendation : list of dictionaries
            In each dictionary, it will contains {Action object,
            estimated_reward, uncertainty}.
        """
        estimated_reward, uncertainty, score = self._exp4p_score(context)

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

        self.n_total += 1
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

        model = self._modelstorage.get_model()
        w = model['w']
        action_probs = model['action_probs']
        action_ids = list(six.viewkeys(six.next(six.itervalues(context))))

        # Update the model
        for action_id, reward in six.viewitems(rewards):
            y_hat = {}
            v_hat = {}
            for i in six.viewkeys(context):
                y_hat[i] = (context[i][action_id] * reward
                            / action_probs[action_id])
                v_hat[i] = sum(
                    [context[i][k] / action_probs[k] for k in action_ids])
                w[i] = w[i] * np.exp(
                    self.p_min / 2
                    * (y_hat[i] + v_hat[i]
                       * np.sqrt(np.log(len(context) / self.delta)
                                 / (len(action_ids) * self.max_rounds))))

        self._modelstorage.save_model({
            'action_probs': action_probs, 'w': w})

        # Update the history
        self._historystorage.add_reward(history_id, rewards)
