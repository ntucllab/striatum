
import logging

from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class Exp4P(BaseBandit):

    r"""Multi-armed bandit algorithm Exp4.P.

    Parameters
    ----------
    storage
    actions
    max_iter : int
    delta : float, optional (default=0.1)
    pmin : float, optional (default=)

    Attributes
    ----------
    t : int
        The current round this instance is at.

    N : int
        The number of arms (actions) in this exp4.p instance.

    query_models\_ : list of :py:mod:`libact.query_strategies` object instance
        The underlying active learning algorithm instances.

    References
    ----------
    .. [1] Beygelzimer, Alina, et al. "Contextual bandit algorithms with
           supervised learning guarantees." In Proceedings on the International
           Conference on Artificial Intelligence and Statistics (AISTATS),
           2011u.
    """

    def __init__(self, storage, actions, max_iter, models,
                 delta=0.1, pmin=None):
        super(Exp4P, self).__init__(storage, actions)

        self.last_reward = None
        self.last_history_id = None

        self.models = models

        # max iters
        if not isinstance(max_iter, int):
            raise ValueError("max_iter should be int, the one"
                             "given is: %f" % pmin)
        self.max_iter = max_iter

        # delta > 0
        if not isinstance(delta, float):
            raise ValueError("delta should be float, the one"
                             "given is: %f" % pmin)
        self.delta = delta

        # p_min in [0, 1/n_arms]
        if pmin is None:
            self.pmin = np.sqrt(np.log(self.N) / self.K / self.T)
        elif not isinstance(pmin, float):
            raise ValueError("pmin should be float, the one"
                             "given is: %f" % pmin)
        elif (pmin < 0) or (pmin > (1. / len(actions))):
            raise ValueError("pmin should be in [0, 1/n_actions], the one"
                             "given is: %f" % pmin)
        else:
            self.pmin = pmin

        self.exp4p_ = None

        self.last_action_idx = None

    def exp4p(self, x):
        """The generator which implements the main part of Exp4.P.

        Parameters
        ----------
        reward: float
            The reward value.

        Yields
        ------
        q: array-like, shape = [K]
            The query vector which tells ALBL what kind of distribution if
            should sample from the unlabeled pool.
        """
        n_exports = len(self.models)
        n_arms = len(self.advices)
        w = np.ones(n_exports)
        advice = np.zeros((n_exports, n_arms))

        while True:
            for i, model in enumerate(self.models):
                advice[i] = model.predict_proba(x)

            # choice vector, shape = (self.K, )
            W = np.sum(w)
            p = (1 - n_arms * self.pmin) * w / W + self.pmin

            # query vector, shape= = (self.n_unlabeled, )
            query_vector = np.dot(p, advice)

            reward, action_idx = yield query_vector

            # shape = (n_exports, 1)
            rhat = reward * advice[:, action_idx] / query_vector[action_idx]

            # shape = ()
            yhat = np.dot(advice, rhat)
            vhat = 1 / p
            w = w * np.exp(
                self.pmin / 2 * (
                    yhat + vhat * np.sqrt(
                        np.log(n_exports / self.delta) / n_arms / self.max_iter
                    )
                )
            )

        raise StopIteration

    def get_action(self, context):
        """Return the action to perform

        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context avaliable.

        Returns
        -------
        history_id : int
            The history id of the action.

        action : Actions object
            The action to perform.
        """
        if self.exp4p_ is None:
            self.exp4p_ = self.exp4p(context)
            query_vector = self.exp4p_.next()
        else:
            query_vector = self.exp4p_.send(self.last_reward,
                                            self.last_action_idx)

        if self.last_reward is None:
            raise ValueError("The last reward have not been passed in.")

        action_idx = np.random.choice(
            np.arange(len(self.actions)),
            size=1,
            p=query_vector
        )[0]

        history_id = self.storage.add_history(context,
                                              self.actions[action_idx],
                                              reward=None)
        self.last_history_id = history_id
        self.last_reward = None
        self.last_action_idx = action_idx

        return history_id, self.actions[action_idx]

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
        if history_id != self.last_history_id:
            raise ValueError("The history_id should be the same as last one.")

        if not isinstance(reward, float):
            raise ValueError("reward should be a float.")

        if reward > 1. or reward < 0.:
            LOGGER.warning("reward passing in should be between 0 and 1"
                           "to maintain theoratical guarantee.")

        self.last_reward = reward
        self.storage.reward(history_id, reward)
