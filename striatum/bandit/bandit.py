"""
Bandit interfaces
"""
from abc import abstractmethod

from striatum import rewardplot as rplt
from ..storage import Recommendation


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

    recommendation_cls : class (default: None)
        The class used to initiate the recommendations. If None, then use
        default Recommendation class.

    Attributes
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.
    """

    def __init__(self, history_storage, model_storage, action_storage,
                 recommendation_cls=None):
        self._history_storage = history_storage
        self._model_storage = model_storage
        self._action_storage = action_storage
        if recommendation_cls is None:
            self._recommendation_cls = Recommendation
        else:
            self._recommendation_cls = recommendation_cls

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

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        pass

    def _get_action_with_empty_action_storage(self, context, n_actions):
        if n_actions is None:
            recommendations = None
        else:
            recommendations = []
        history_id = self._history_storage.add_history(context,
                                                       recommendations)
        return history_id, recommendations

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
            A list of Action objects for recommendation.
        """
        pass

    def update_action(self, action):
        """Update action.

        Parameters
        ----------
        action : Action
            The Action object to update.
        """
        self._action_storage.update(action)

    @abstractmethod
    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
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
