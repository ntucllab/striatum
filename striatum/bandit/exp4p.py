
import logging

from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class Exp4P(BaseBandit):

    """Multi-armed bandit algorithm Exp4.P.

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

    def __init__(self, actions, HistoryStorage, ModelStorage, models,
                 delta=0.1, pmin=None):
        super(Exp4P, self).__init__(HistoryStorage, ModelStorage, actions)

        self.last_history_id = -1
        self.models = models
        self.n_total = 0
        self.n_experts = len(self.models)           # number of experts (i.e. N in the paper)
        self.n_actions = len(self.actions)          # number of actions (i.e. K in the paper)

        # delta > 0
        if not isinstance(delta, float):
            raise ValueError("delta should be float, the one"
                             "given is: %f" % pmin)
        self.delta = delta

        # p_min in [0, 1/n_actions]
        if pmin is None:
            self.pmin = np.sqrt(np.log(self.n_experts) / self.n_experts / 10000)
        elif not isinstance(pmin, float):
            raise ValueError("pmin should be float, the one"
                             "given is: %f" % pmin)
        elif (pmin < 0) or (pmin > (1. / len(actions))):
            raise ValueError("pmin should be in [0, 1/n_actions], the one"
                             "given is: %f" % pmin)
        else:
            self.pmin = pmin

        # Initialize the model storage
        query_vector = np.zeros(self.n_actions)     # probability distribution for action recommendation)
        w = np.ones(self.n_experts)                 # weight vector for each expert
        advice = np.zeros((self.n_experts, self.n_actions))
        self._ModelStorage.save_model({'query_vector': query_vector, 'w': w, 'advice': advice})

    def exp4p(self):

        """The generator which implements the main part of Exp4.P.
        """

        while True:
            context = yield

            advice = np.zeros((self.n_experts, self.n_actions))
            # get the expert advice (probability)
            for i, model in enumerate(self.models):
                advice[i, :] = model.predict_proba(context)

            # choice vector, shape = (self.K, )
            w = self._ModelStorage.get_model()['w']
            w_sum = np.sum(w)
            p_temp = (1 - self.n_actions * self.pmin) * w / w_sum + self.pmin

            # query vector, shape= = (self.n_unlabeled, )
            query_vector = np.dot(p_temp, advice)
            self._ModelStorage.save_model({'query_vector': query_vector, 'w': w, 'advice': advice})

            # give back the
            action_idx = np.random.choice(np.arange(len(self.actions)), size=1, p = query_vector/sum(query_vector))[0]
            action_max = self.actions[action_idx]
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
        learn = self.exp4p()
        learn.next()
        action_max = learn.send(context)
        self.n_total = self.n_total + 1
        self.last_history_id = self.last_history_id + 1
        self._HistoryStorage.add_history(np.transpose(np.array([context])), action_max, reward=None)
        return self.last_history_id, action_max

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

        reward_action = self._HistoryStorage.unrewarded_histories[history_id].action
        w_old = self._ModelStorage.get_model()['w']
        query_vector = self._ModelStorage.get_model()['query_vector']
        advice = self._ModelStorage.get_model()['advice']

        # Update the model
        rhat = np.zeros(self.n_actions)
        rhat[reward_action] = reward/query_vector[reward_action]
        yhat = np.dot(advice, rhat)
        vhat = np.zeros(self.n_experts)
        for i in range(self.n_experts):
            for j in range(self.n_actions):
                vhat[i] = vhat[i] + advice[i,j]/query_vector[j]

        w_new = w_old * np.exp(
                           self.pmin / 2 * (
                                yhat + vhat * np.sqrt(
                                    np.log(self.n_experts / self.delta) / self.n_actions / self.n_total
                                )
                            )
                        )

        self._ModelStorage.save_model({'query_vector': query_vector, 'w': w_new, 'advice': advice})

        # Update the history
        self._HistoryStorage.add_reward(history_id, reward)
