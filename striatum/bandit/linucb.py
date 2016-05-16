
import logging

from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class LinUCB(BaseBandit):

    """UCB with Linear Hypotheses
    """

    def __init__(self, actions, storage, alpha=1.0):
        super(LinearUCB, self).__init__(storage, actions)

        self.last_reward = None
        self.last_history_id = None

        self.linucb_ = None
        self.alpha_ = alpha

    def linucb(self, x):
        n_arms = len(self.actions)
        A = np.eye(n_arms)
        b = np.ones(n_arms) / np.sqrt(float(n_arms))

        while True:
            invA = np.linalg.pinv(A)
            theta = np.dot(invA, b)

            p = np.dot(theta, x) + \
                self.alpha * np.sqrt(
                    np.einsum('ij,ji->i', np.dot(x.T, invA), x))

            # next context x and last reward
            at = np.random.choice(np.where(p == np.max(p))[0])
            x_new, reward = yield at

            A = A + np.dot(x[:, at], x[:, at].T)
            b = b + reward * x[:, at]

            x = x_new

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
        if self.linucb_ is None:
            self.linucb_ = self.linucb(context)
        if self.last_reward is None:
            raise ValueError("The last reward have not been passed in.")
        action = self.linucb_.send(context, self.last_reward)

        self.last_reward = None

        history_id = self.storage.add_history(None, action, reward=None)
        self.last_history_id = history_id

        return history_id, action

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

        if reward >= 1. or reward <= 0:
            LOGGER.warning("reward passing in should be between 0 and 1"
                           "to maintain theoratical guarantee.")

        self.last_reward = reward
        self.storage.reward(history_id, reward)
