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
    actions : {array-like, None}
        Actions (arms) for recommendation
    historystorage: a HistoryStorage object
        The place where we store the histories of contexts and rewards.
    modelstorage: a ModelStorage object
        The place where we store the model parameters.
    models: the list of pre-trained supervised learning model objects.
        Use historical contents and rewards to train several multi-class classification models as experts.
        We strongly recommend to use scikit-learn package to pre-train the experts.
    delta: float, 0 < delta <= 1
        With probability 1 - delta, LinThompSamp satisfies the theoretical regret bound.
    pmin: float, 0 < pmin < 1/n_actions
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

    def __init__(self, actions, historystorage, modelstorage, models, delta=0.1, pmin=None):
        super(Exp4P, self).__init__(historystorage, modelstorage, actions)
        self.models = models
        self.n_total = 0
        self.n_experts = len(self.models)           # number of experts (i.e. N in the paper)
        self.n_actions = len(self._actions)          # number of actions (i.e. K in the paper)
        self.exp4p_ = None

        # delta > 0
        if not isinstance(delta, float):
            raise ValueError("delta should be float, the one"
                             "given is: %f" % pmin)
        self.delta = delta

        # p_min in [0, 1/n_actions]
        if pmin is None:
            self.pmin = np.sqrt(np.log(self.n_experts) / self.n_actions / 10000)
        elif not isinstance(pmin, float):
            raise ValueError("pmin should be float, the one"
                             "given is: %f" % pmin)
        elif (pmin < 0) or (pmin > (1. / self.n_actions)):
            raise ValueError("pmin should be in [0, 1/n_actions], the one"
                             "given is: %f" % pmin)
        else:
            self.pmin = pmin

        # Initialize the model storage
        query_vector = np.zeros(self.n_actions)     # probability distribution for action recommendation)
        w = np.ones(self.n_experts)                 # weight vector for each expert
        advice = np.zeros((self.n_experts, self.n_actions))
        self._modelstorage.save_model({'query_vector': query_vector, 'w': w, 'advice': advice})

    def exp4p(self):

        """The generator which implements the main part of Exp4.P.
        """

        while True:
            context = yield

            advice = np.zeros((self.n_experts, self.n_actions))
            # get the expert advice (probability)
            for i, model in enumerate(self.models):
                if len(model.classes_) != len(self._actions):
                    proba = model.predict_proba([context])
                    k = 0
                    for action in self._actions:
                        if action in model.classes_:
                            action_idx = self._actions.index(action)
                            advice[i, action_idx] = proba[0][k]
                            k += 1
                        else:
                            action_idx = self._actions.index(action)
                            advice[i, action_idx] = self.pmin
                else:
                    advice[i, :] = model.predict_proba([context])

            # choice vector, shape = (self.K, )
            w = self._modelstorage.get_model()['w']
            w_sum = np.sum(w)
            p_temp = (1 - self.n_actions * self.pmin) * w / w_sum + self.pmin

            # query vector, shape= = (self.n_unlabeled, )
            query_vector = np.dot(p_temp, advice)
            self._modelstorage.save_model({'query_vector': query_vector, 'w': w, 'advice': advice})

            # give back the
            action_idx = np.random.choice(np.arange(len(self._actions)), size=1, p=query_vector/sum(query_vector))[0]
            action_max = self._actions[action_idx]
            yield action_max

        raise StopIteration

    def get_action(self, context):
        """Return the action to perform
        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context available.
        Returns
        -------
        history_id : int
            The history id of the action.
        action : Actions object
            The action to perform.
        """
        if self.exp4p_ is None:
            self.exp4p_ = self.exp4p()
            six.next(self.exp4p_)
            action_max = self.exp4p_.send(context)
        else:
            six.next(self.exp4p_)
            action_max = self.exp4p_.send(context)

        self.n_total += 1
        history_id = self._historystorage.add_history(np.transpose(np.array([context])), action_max, reward=None)
        return history_id, action_max

    def reward(self, history_id, reward):
        """Reward the preivous action with reward.
        Parameters
        ----------
        history_id : int
            The history id of the action to reward.
        reward : float
            A float representing the feedback given to the action, the higher
            the better.
        """

        reward_action = self._historystorage.unrewarded_histories[history_id].action
        reward_action_idx = self._actions.index(reward_action)
        w_old = self._modelstorage.get_model()['w']
        query_vector = self._modelstorage.get_model()['query_vector']
        advice = self._modelstorage.get_model()['advice']

        # Update the model
        rhat = np.zeros(self.n_actions)
        rhat[reward_action_idx] = reward/query_vector[reward_action_idx]
        yhat = np.dot(advice, rhat)
        vhat = np.zeros(self.n_experts)
        for i in range(self.n_experts):
            for j in range(self.n_actions):
                vhat[i] = vhat[i] + advice[i, j] / query_vector[j]

        w_new = w_old * np.exp(
                           self.pmin / 2 * (
                                yhat + vhat * np.sqrt(
                                    np.log(self.n_experts / self.delta) / self.n_actions / self.n_total
                                )
                            )
                        )

        self._modelstorage.save_model({'query_vector': query_vector, 'w': w_new, 'advice': advice})

        # Update the history
        self._historystorage.add_reward(history_id, reward)
