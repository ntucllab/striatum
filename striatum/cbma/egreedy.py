"""Epsilon Greedy under CBMA Setting
This module contains a class that implements epsilon-greedy algorithm under CBMA setting.
"""

import six
import logging
from striatum.cbma.cbma import BaseCbma
import numpy as np

LOGGER = logging.getLogger(__name__)


class EpsilonGreedy(BaseCbma):
    """Epsilon-greedy algorithm under cbma setting.

    Parameters
    ----------
    actions : {array-like, None}
        Actions (arms) for recommendation
    historystorage: a :py:mod:'striatum.storage.HistoryStorage' object
        The object where we store the histories of contexts and rewards.
    modelstorage: a :py:mod:'straitum.storage.ModelStorage' object
        The object where we store the model parameters.
    epsilon: float
        The probability of random exploration.
    d: int
        The dimension of the context.

    Attributes
    ----------
    greedy\_ : 'greedy' object instance
        The contextual bandit algorithm instances.

    References
    ----------
    .. [1]  Y-H Chang, and H-T Lin. "Pairwise Regression with Upper Confidence Boun for Contextual Bandit with
    Multiple Actions."  In Proceedings of the Conference on Technologies and Applications for Artificial Intelligence
    (TAAI), pages 19--24, December 2013.
    """

    def __init__(self, actions, historystorage, modelstorage, epsilon=0.01, d=1):
        super(EpsilonGreedy, self).__init__(historystorage, modelstorage, actions)
        self.last_reward = None
        self.epsilon = epsilon
        self.d = d
        self.greedy_ = None

        # Initialize LinUCB Model Parameters
        matrix_a = np.identity(self.d)
        matrix_ainv = np.identity(self.d)
        b = np.zeros((self.d, 1))
        theta = np.zeros((self.d, 1))
        self._modelstorage.save_model({'matrix_a': matrix_a, 'matrix_ainv': matrix_ainv, 'b': b, 'theta': theta})

    def greedy(self):
        """The generator implementing the greedy recommendation.
        """
        while True:
            context = yield
            context = np.matrix(context)
            theta_tmp = self._modelstorage.get_model()['theta']
            score = np.dot(context, theta_tmp)
            yield score
        raise StopIteration

    def get_action(self, n_recommend, context):
        """Return the action to perform

        Parameters
        ----------
        n_recommend: int
            Number of actions wanted to recommend users.

        context : {array-like, None}
            The context of current state, None if no context avaliable.

        Returns
        -------

        actions : Actions object
            The actions to perform. (Number of actions = n_recommend.)

        score : dictionary
            The dictionary with actions as key and scores as value.

        """

        if self.greedy_ is None:
            self.greedy_ = self.greedy()
            six.next(self.greedy_)
            score = self.greedy_.send(context)
        else:
            six.next(self.greedy_)
            score = self.greedy_.send(context)

        # implement the epsilon part
        indicator = int(np.random.choice([1, 2], size=1, p=[self.epsilon, 1-self.epsilon]))
        if indicator == 1:
            index = np.random.random_integers(0, len(self._actions) - 1, size=n_recommend)
            actions = [self._actions[i] for i in index]
        else:
            actions = [self._actions[i] for i in np.array(score).argsort()[-n_recommend:][::-1]]

        history_id = self._historystorage.add_history(context, actions, reward=None)
        final_score = {}
        for i in range(len(self._actions)):
            final_score[self._actions[i]] = score[i]
        return history_id, actions, final_score

    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary with actions as keys and rewards as values.
        """

        # Update the model
        matrix_a = self._modelstorage.get_model()['matrix_a']
        matrix_ainv = self._modelstorage.get_model()['matrix_ainv']
        b = self._modelstorage.get_model()['b']
        theta = self._modelstorage.get_model()['theta']

        for action, reward in rewards.items():
            reward_action_idx = self._actions.index(action)
            context = self._historystorage.unrewarded_histories[history_id].context[reward_action_idx]
            context = np.matrix(context)
            matrix_a += np.dot(context.T, context)
            matrix_ainv = np.linalg.solve(matrix_a, np.identity(self.d))
            b += reward * context.T
            theta = np.dot(matrix_ainv, b)
        self._modelstorage.save_model({'matrix_a': matrix_a, 'matrix_ainv': matrix_ainv, 'b': b, 'theta': theta})

        # Update the history
        self._historystorage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : {array-like, None}
            New actions (arms) for recommendation
        """
        self._actions.extend(actions)
