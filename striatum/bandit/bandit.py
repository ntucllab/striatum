"""
Bandit interfaces
"""
from abc import abstractmethod

from striatum import rewardplot as rplt


class BaseBandit(object):
    r"""Bandit algorithm

    Parameters
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    Attributes
    ----------
    \_history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    \_model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    \_action_storage : ActionStorage object
        The ActionStorage object to store actions.

    \_action_ids: list of integers
        List of all action_id's.
    """

    def __init__(self, history_storage, model_storage, action_storage):
        self._history_storage = history_storage
        self._model_storage = model_storage
        self._action_storage = action_storage

    @property
    def history_storage(self):
        return self._history_storage

    @abstractmethod
    def get_action(self, context, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {action_id: context} of different actions.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

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
