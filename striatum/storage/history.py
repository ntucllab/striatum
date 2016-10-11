"""
History storage
"""
from abc import abstractmethod
from datetime import datetime


class History(object):
    """action/reward history entry"""

    def __init__(self, history_id, action_time, context, action,
                 reward_time=None, reward=None):
        """
        history_id : int
        action_time : datetime
        context : {array-like, None}
        action : Action object
        reward_time : datetime
        reward : {float, None}
        """
        self.history_id = history_id
        self.action_time = action_time
        self.context = context
        self.action = action
        self.reward_time = reward_time
        self.reward = reward

    def update_reward(self, reward_time, reward):
        """update reward_time and reward"""
        self.reward_time = reward_time
        self.reward = reward


class HistoryStorage(object):
    """The object to store the history of context, actions and rewards."""
    @abstractmethod
    def get_history(self, history_id):
        """Get the preivous context, action and reward with history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        history: History object
        """
        pass

    @abstractmethod
    def get_unrewarded_history(self, history_id):
        """Get the previous unrewarded context, action and reward with
        history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        history: History object

        Raise
        -----
        KeyError
        """
        pass

    @abstractmethod
    def add_history(self, context, action, reward=None):
        """Add a history record.

        Parameters
        ----------
        context : {array-like, None}
        action : Action object
        reward : {float, None}, optional (default: None)

        Raise
        -----
        """
        pass

    @abstractmethod
    def add_reward(self, history_id, reward):
        """Add reward to a history record.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        reward : float

        Raise
        -----
        """
        pass


class MemoryHistoryStorage(HistoryStorage):
    """HistoryStorage that store all data in memory"""

    def __init__(self):
        self.histories = {}
        self.unrewarded_histories = {}
        self.n_histories = 0

    def get_history(self, history_id):
        """Get the previous context, action and reward with history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        history: History object

        Raise
        -----
        KeyError
        """
        return self.histories[history_id]

    def get_unrewarded_history(self, history_id):
        """Get the previous unrewarded context, action and reward with
        history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        history: History object

        Raise
        -----
        KeyError
        """
        return self.unrewarded_histories[history_id]

    def add_history(self, context, action, reward=None):
        """Add a history record.

        Parameters
        ----------
        context : {array-like, None}
        action : Action object
        reward : {float, None}, optional (default: None)

        Raise
        -----
        """
        action_time = datetime.now()
        history_id = self.n_histories
        if reward is None:
            history = History(history_id, action_time, context, action)
            self.unrewarded_histories[history_id] = history
        else:
            reward_time = action_time
            history = History(history_id, action_time, context, action,
                              reward_time, reward)
            self.histories[history_id] = history
        self.n_histories += 1
        return history_id

    def add_reward(self, history_id, reward):
        """Add reward to a history record.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        reward : float

        Raise
        -----
        KeyError
        """
        reward_time = datetime.now()
        history = self.unrewarded_histories.pop(history_id)
        history.update_reward(reward_time, reward)
        self.histories[history.history_id] = history
