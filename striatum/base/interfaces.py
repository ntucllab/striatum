"""
Base interfaces
"""
from abc import abstractmethod


class Action(object):

    """The action object
    """

    @abstractmethod
    def __init__(self):
        pass


class Storage(object):

    """The object to store history context, actions and rewards.
    """

    @abstractmethod
    def __init__(self):
        pass

    @property
    def history(self):
        """dictionary of history_id mapping to tuple (timestamp, context,
        action, reward)"""
        return self._history

    @property
    def unrewarded_history(self):
        """dictionary of history_id mapping to tuple (timestamp, context,
        action)"""
        return self._history

    @abstractmethod
    def get_history(self, history_id):
        """Get the preivous context, action and reward with history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        timestamp : float
            pass

        context : {array-like, None}

        action : Action object

        reward : {float, None}
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


class BaseBandit(object):

    """Bandit algorithm

    Parameters
    ----------
    storage : Storage object
        The storage object to store history context, actions and rewards.

    actions : list of Action objects
        List of actions to be chosen from.

    Attributes
    ----------
    storage : Storage object
        The storage object to store history context, actions and rewards.

    actions : list of Action objects
        List of actions to be chosen from.
    """

    def __init__(self, storage, actions, **kwargs):
        self._storage = storage
        self._actions = actions

    @property
    def storage(self):
        """Storage object that stores history"""
        return self._storage

    @property
    def actions(self):
        """List of actions"""
        return self._actions

    @abstractmethod
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
        pass

    @abstractmethod
    def reward(self, history_id):
        """Reward the preivous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        reward : float
            A float representing the feedback given to the action, the higher
            the better.
        """
        pass
