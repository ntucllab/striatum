""" EXP4.P: An Extention to Exponential-weight algorithm for Exploration and Exploitation
This module contains a class that implements EXP4.P, a contextual bandit algorithm with expert advice.
"""

import logging
import six
from striatum.bandit.bandit import BaseBandit
import numpy as np
LOGGER = logging.getLogger(__name__)


class Exp4P(BaseBandit):
    """Exp4.P with pre-trained supervised learning algorithm.

    Parameters
    ----------
    actions : list of Action objects
        List of actions to be chosen from.

    historystorage: a HistoryStorage object
        The place where we store the histories of contexts and rewards.

    modelstorage: a ModelStorage object
        The place where we store the model parameters.

    delta: float, 0 < delta <= 1
        With probability 1 - delta, LinThompSamp satisfies the theoretical regret bound.

    pmin: float, 0 < pmin < 1/k
        The minimum probability to choose each action.

    Attributes
    ----------
    exp4p\_ : 'exp4p' object instance
        The contextual bandit algorithm instances

    References
    ----------
    .. [1]  Beygelzimer, Alina, et al. "Contextual bandit algorithms with supervised learning guarantees."
            International Conference on Artificial Intelligence and Statistics (AISTATS). 2011u.
    """

    def __init__(self, actions, historystorage, modelstorage, delta=0.1, pmin=None):
        super(Exp4P, self).__init__(historystorage, modelstorage, actions)
        self.n_total = 0
        self.k = len(self._actions)          # number of actions (i.e. K in the paper)
        self.exp4p_ = None

        # delta > 0
        if not isinstance(delta, float):
            raise ValueError("delta should be float, the one"
                             "given is: %f" % pmin)
        self.delta = delta

        # p_min in [0, 1/k]
        if pmin is None:
            self.pmin = np.sqrt(np.log(10) / self.k / 10000)
        elif not isinstance(pmin, float):
            raise ValueError("pmin should be float, the one"
                             "given is: %f" % pmin)
        elif (pmin < 0) or (pmin > (1. / self.k)):
            raise ValueError("pmin should be in [0, 1/k], the one"
                             "given is: %f" % pmin)
        else:
            self.pmin = pmin

        # Initialize the model storage
        query_vector = np.zeros(self.k)     # probability distribution for action recommendation)
        w = {}                              # weight vector for each expert
        self._modelstorage.save_model({'query_vector': query_vector, 'w': w})

    def exp4p(self):
        """The generator which implements the main part of Exp4.P.
        """

        while True:
            context = yield
            advisors_id = context.keys()

            w = self._modelstorage.get_model()['w']
            if w == {}:
                for i in advisors_id:
                    w[i] = 1
            w_sum = np.sum(list(w.values()))

            query_vector = [(1 - self.k * self.pmin) *
                            np.sum(np.array([w[i] * context[i][action_id] for i in advisors_id])/w_sum) +
                            self.pmin for action_id in self.action_ids]
            query_vector /= sum(query_vector)
            self._modelstorage.save_model({'query_vector': query_vector, 'w': w})

            estimated_reward = {}
            uncertainty = {}
            score = {}
            for i in range(self.k):
                estimated_reward[self.action_ids[i]] = query_vector[i]
                uncertainty[self.action_ids[i]] = 0
                score[self.action_ids[i]] = query_vector[i]
            yield estimated_reward, uncertainty, score

        raise StopIteration

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
            In each dictionary, it will contains {Action object, estimated_reward, uncertainty}.
        """

        if self.exp4p_ is None:
            self.exp4p_ = self.exp4p()
            six.next(self.exp4p_)
            estimated_reward, uncertainty, score = self.exp4p_.send(context)

        else:
            six.next(self.exp4p_)
            estimated_reward, uncertainty, score = self.exp4p_.send(context)

        action_recommendation = []
        action_recommendation_ids = sorted(score, key=score.get, reverse=True)[:n_actions]

        for action_id in action_recommendation_ids:
            action_id = int(action_id)
            action = [action for action in self._actions if action.action_id == action_id][0]
            action_recommendation.append({'action': action, 'estimated_reward': estimated_reward[action_id],
                                          'uncertainty': uncertainty[action_id], 'score': score[action_id]})

        self.n_total += 1
        history_id = self._historystorage.add_history(context, action_recommendation, reward=None)
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
        action_ids = context[context.keys()[0]].keys()

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
            rhat[action_id] = reward_tmp/query_vector[action_id]
            for i in context.keys():
                yhat[i] = np.dot(context[i].values(), rhat.values())
                vhat[i] = sum([context[i][k]/np.array(query_vector.values()) for k in action_ids])
                w_new[i] = w_old[i] + np.exp(self.pmin / 2 * (
                    yhat[i] + vhat[i] * np.sqrt(
                        np.log(len(context) / self.delta) / self.k / self.n_total
                        )
                    )
                )

        self._modelstorage.save_model({'query_vector': query_vector, 'w': w_new})

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
