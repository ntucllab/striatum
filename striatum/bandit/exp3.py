
import logging

from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class Exp3(BaseBandit):

    def __init__(self, actions, HistoryStorage, ModelStorage, gamma):
        super(Exp3, self).__init__(HistoryStorage, ModelStorage, actions)

        self.last_history_id = -1
        self.n_actions = len(self.actions)          # number of actions (i.e. K in the paper)

        # gamma in (0,1]
        if not isinstance(gamma, float):
            raise ValueError("gamma should be float, the one"
                             "given is: %f" % gamma)
        elif (gamma <= 0) or (gamma > 1):
            raise ValueError("gamma should be in (0, 1], the one"
                            "given is: %f" % gamma)
        else:
            self.gamma = gamma

        # Initialize the model storage
        query_vector = np.zeros(self.n_actions)     # probability distribution for action recommendation)
        w = np.ones(self.n_actions)                 # weight vector
        self._ModelStorage.save_model({'query_vector': query_vector, 'w': w})

    def exp4p(self):

        """The generator which implements the main part of Exp3.
        """

        while True:
            context = yield

            advice = np.zeros((self.n_experts, self.n_actions))
            # get the expert advice (probability)
            for i, model in enumerate(self.models):
                if len(model.classes_) != len(self.actions):
                    proba = model.predict_proba([context])
                    k = 0
                    for action in self.actions:
                        if action in model.classes_:
                            action_idx = self.actions.index(action)
                            advice[i,action_idx] = proba[0][k]
                            k = k + 1
                        else:
                            action_idx = self.actions.index(action)
                            advice[i, action_idx] = self.pmin
                else:
                    advice[i, :] = model.predict_proba([context])

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

        reward_action = self._HistoryStorage.unrewarded_histories[history_id].action
        reward_action_idx = self.actions.index(reward_action)
        w_old = self._ModelStorage.get_model()['w']
        query_vector = self._ModelStorage.get_model()['query_vector']
        advice = self._ModelStorage.get_model()['advice']

        # Update the model
        rhat = np.zeros(self.n_actions)
        rhat[reward_action_idx] = reward/query_vector[reward_action_idx]
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
