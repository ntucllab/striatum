import logging
from striatum.bandit.bandit import BaseBandit
import numpy as np

# TODO: How to deal with new actions?

LOGGER = logging.getLogger(__name__)

class LinUCB(BaseBandit):

    """UCB with Linear Hypotheses
    """

    def __init__(self, actions, HistoryStorage, ModelStorage, alpha, d = 1):
        """ Initialize the LinUCB model.
        :param actions:
        :param HistoryStorage:
        :param ModelStorage:
        :param alpha:
        :param d:
        """

        super(LinUCB, self).__init__(HistoryStorage, ModelStorage, actions)
        self._actions = np.array(actions)
        self._HistoryStorage = HistoryStorage
        self._ModelStorage = ModelStorage
        self.last_reward = None
        self.last_history_id = -1
        self.alpha = alpha
        self.d = d
        # Initialize LinUCB Model Parameters
        Aa = {}
        AaI = {}
        ba = {}
        theta = {}
        for key in self._actions:
            Aa[key] = np.identity(self.d)
            AaI[key] = np.identity(self.d)
            ba[key] = np.zeros((self.d, 1))
            theta[key] = np.zeros((self.d, 1))
        self._ModelStorage.save_model({'Aa': Aa, 'AaI': AaI, 'ba': ba, 'theta': theta})

    def linucb(self):

        """The generator implementing the linear LINUCB algorithm.
        """

        while True:
            context = yield
            xaT = np.array([context])
            xa = np.transpose(xaT)
            AaI_tmp = np.array([self._ModelStorage.get_model()['AaI'][action] for action in self._actions])
            theta_tmp = np.array([self._ModelStorage.get_model()['theta'][action] for action in self._actions])

            # The recommended action should maximize the Linear UCB.
            action_max = self._actions[np.argmax(np.dot(xaT, theta_tmp) +
                                                 self.alpha * np.sqrt(np.dot(np.dot(xaT, AaI_tmp), xa)))]
            yield action_max

    def reward(self, history_id, reward):

        """Reward the preivous action with reward.

            Parameters
            ----------
            history_id : int
                The history id of the action to reward.
            reward : float
                A float representing the feedback given to the action, the higher the better.

        """

        context = self._HistoryStorage.unrewarded_histories[history_id].context
        reward_action = self._HistoryStorage.unrewarded_histories[history_id].action

        # Update the model
        self._ModelStorage._model['Aa'][reward_action] += np.dot(context, np.transpose(context))
        self._ModelStorage._model['AaI'][reward_action] = np.linalg.solve(self._ModelStorage._model['Aa'][reward_action], np.identity(self.d))
        self._ModelStorage._model['ba'][reward_action] += reward * context
        self._ModelStorage._model['theta'][reward_action] = np.dot(self._ModelStorage._model['AaI'][reward_action], self._ModelStorage._model['ba'][reward_action])

        # Update the history
        self._HistoryStorage.add_reward(history_id, reward)


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

        learn = self.linucb()
        learn.next()
        action_max = learn.send(context)
        self.last_history_id = self.last_history_id + 1
        self._HistoryStorage.add_history(np.transpose(np.array([context])), action_max, reward = None)
        return self.last_history_id, action_max
