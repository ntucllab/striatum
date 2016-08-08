"""
Bandit interfaces
"""
from abc import abstractmethod


class Action(object):
    """The action object

    Parameters
    ----------
        action_id: int
            The idx of this action.
        title:
        content:
    """
    def __init__(self, action_id, title, content):
        self.action_id = action_id
        self.title = title
        self.content = content


class BaseBandit(object):
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

    actions_id: list of integers
        List of all action_id's.
    """
    def __init__(self, historystorage, modelstorage, actions):
        self._historystorage = historystorage
        self._modelstorage = modelstorage
        self._actions = actions
        self._actions_id = [actions[i].action_id for i in range(len(actions))]

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
    def get_action(self, context, n_action=1):
        """Return the action to perform

        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context available.

        n_action: int
                Number of actions wanted to recommend users.

        Returns
        -------
        history_id : int
            The history id of the action.

        action : Actions object
            The action to perform.
        """
        pass

    @abstractmethod
    def reward(self, history_id, reward):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        reward : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        pass

