"""
Bandit interfaces
"""
from abc import abstractmethod


class Action(object):
    """The action object

    Parameters
    ----------
    action_id: int
        The index of this action.

    title: string
        The title of this action.

    content: object
        The content of this action.
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
    \_historystorage : historystorage object
        The historystorage object to store history context, actions and rewards.

    \_modelstorage : modelstorage object
        The modelstorage object to store model parameters.

    \_actions : list of Action objects
        List of actions to be chosen from.

    \_action_ids: list of integers
        List of all action_id's.
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

    @property
    def action_ids(self):
        return [self._actions[i].action_id for i in range(len(self._actions))]

    @abstractmethod
    def get_action(self, context, n_actions=1):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {action_id: context} of different actions.

        n_actions: int
            Number of actions wanted to recommend users.

        Returns
        -------
        history_id : int
            The history id of the action.

        action_recommendation : list of dictionaries
            In each dictionary, it will contains {Action object, estimated_reward, uncertainty}.
        """
        pass

    @abstractmethod
    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        pass

    @abstractmethod
    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation
        """
        pass
