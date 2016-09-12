"""
Bandit interfaces
"""

from abc import abstractmethod

from striatum import rewardplot as rplt


class Action(object):
    r"""The action object

    Parameters
    ----------
    action_id: int
        The index of this action.
    """

    def __init__(self, action_id):
        self.action_id = action_id


class BaseBandit(object):
    r"""Bandit algorithm

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
        """List of action ids"""
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
            Each dictionary contains
            {Action object, estimated_reward, uncertainty}.
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

    def calculate_cum_reward(self):
        """Calculate cumulative reward with respect to time.
        """
        return rplt.calculate_cum_reward(self)

    def calculate_avg_reward(self):
        """Calculate average reward with respect to time.
        """
        return rplt.calculate_avg_reward(self)

    def plot_avg_reward(self):
        """Plot average reward with respect to time.
        """
        rplt.plot_avg_reward(self)

    def plot_avg_regret(self):
        """Plot average regret with respect to time.
        """
        rplt.plot_avg_regret(self)

    def get_action_with_id(self, action_id):
        for action in self._actions:
            if action.action_id == action_id:
                return action
        else:
            raise KeyError("action_id {} not found".format(action_id))
