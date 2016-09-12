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
                 p_min=None):
        super(Exp4P, self).__init__(historystorage, modelstorage, actions)
        self.n_total = 0
        self.n_actions = len(self._actions)  # number of actions (i.e. K in the paper)
        self.exp4p_ = None

        # delta > 0
        if not isinstance(delta, float):
            raise ValueError("delta should be float, the one"
                             "given is: %f" % p_min)
        self.delta = delta

        # p_min in [0, 1/k]
        if p_min is None:
            self.p_min = np.sqrt(np.log(10) / self.n_actions / 10000)
        elif not isinstance(p_min, float):
            raise ValueError("p_min should be float, the one"
                             "given is: %f" % p_min)
        elif (p_min < 0) or (p_min > (1. / self.n_actions)):
            raise ValueError("p_min should be in [0, 1/k], the one"
                             "given is: %f" % p_min)
        else:
            self.p_min = p_min

        # Initialize the model storage

        # probability distribution for action recommendation
        query_vector = np.zeros(self.n_actions)
        # weight vector for each expert
        w = {}
        self._modelstorage.save_model({'query_vector': query_vector, 'w': w})

    def _exp4p_score(self, context):
        """The generator which implements the main part of Exp4.P.
        """
        advisor_ids = list(six.viewkeys(context))

        w = self._modelstorage.get_model()['w']
        if len(w) == 0:
            for i in advisor_ids:
                w[i] = 1
        w_sum = sum(six.viewvalues(w))

        query_vector = []
        for action_id in self.action_ids:
            weighted_exp = np.asarray(
                [w[advisor_id] * context[advisor_id][action_id]
                 for advisor_id in advisor_ids])
            prob_vector = np.sum(weighted_exp / w_sum)
            query_vector.append((1 - self.n_actions * self.p_min) * prob_vector
                                + self.p_min)
        query_vector /= sum(query_vector)
        self._modelstorage.save_model(
            {'query_vector': query_vector, 'w': w})

        estimated_reward = {}
        uncertainty = {}
        score = {}
        for action_id, action_prob in zip(self.action_ids, query_vector):
            estimated_reward[action_id] = action_prob
            uncertainty[action_id] = 0
            score[action_id] = action_prob

        return estimated_reward, uncertainty, score

    def get_action(self, context=None, n_actions=1):
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
            In each dictionary, it will contains {Action object,
            estimated_reward, uncertainty}.
        """
        estimated_reward, uncertainty, score = self._exp4p_score(context)

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

        w_old = self._modelstorage.get_model()['w']
        query_vector_tmp = self._modelstorage.get_model()['query_vector']
        context = self._historystorage.unrewarded_histories[history_id].context
        action_ids = list(context[list(context.keys())[0]].keys())

        query_vector = {}
        for k in range(len(query_vector_tmp)):
            query_vector[action_ids[k]] = query_vector_tmp[k]

        # Update the model
        for action_id, reward_tmp in rewards.items():
            rhat = {}
            for i in action_ids:
                rhat[i] = 0.0
            w_new = {}
            yhat = {}
            vhat = {}
            rhat[action_id] = reward_tmp / query_vector[action_id]
            for i in context.keys():
                yhat[i] = np.dot(list(context[i].values()), list(rhat.values()))
                vhat[i] = sum(
                    [context[i][k] / np.array(list(query_vector.values()))
                     for k in action_ids])
                w_new[i] = w_old[i] + np.exp(self.p_min / 2 * (
                    yhat[i] + vhat[i] * np.sqrt(np.log(
                        len(context) / self.delta) / self.n_actions / self.n_total)
                ))

        self._modelstorage.save_model({
            'query_vector': query_vector, 'w': w_new})

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
