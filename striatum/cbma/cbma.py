"""
Bandit interfaces
"""
from abc import abstractmethod


# TODO: think about how to use this
class Action(object):
    """The action object"""
    @abstractmethod
    def __init__(self):
        pass


class BaseCbma(object):
    """Bandit algorithm

    Parameters
    ----------
    historystorage : historystorage object
        The historystorage object to store history context, actions and rewards.

    modelstorage : modelstorage object
        The modelstorage object to store model parameters.

    actions : list of Action objects
        List of actions to be chosen from.

    Attributes
    ----------
    historystorage : historystorage object
        The historystorage object to store history context, actions and rewards.

    modelstorage : modelstorage object
        The modelstorage object to store model parameters.

    actions : list of Action objects
        List of actions to be chosen from.
    """
    def __init__(self, historystorage, modelstorage, actions):
        self._historystorage = historystorage
        self._modelstorage = modelstorage
        self._actions = actions

    @property
    def historystorage(self):
        """HistoryStorage object that stores history"""
        return self._historystorage

    @property
    def modelstorage(self):
        """ModelStorage object that stores model parameters"""
        return self._modelstorage

    @property
    def actions(self):
        """List of actions"""
        return self._actions

    @abstractmethod
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
            The actions to perform.

        score : dictionary
            The dictionary with actions as key and scores as value.

        """
        pass

    @abstractmethod
    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : list
            A list of float numbers representing the feedback given to actions, the higher
            the better.
        """
        pass
