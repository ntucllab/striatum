"""
History storage
"""
from abc import abstractmethod
from datetime import datetime


class History(object):
    """action/reward history entry.

    Parameters
    ----------
    history_id : int
    context : {dict of list of float, None}
    recommendations : {Recommendation, list of Recommendation}
    created_at : datetime
    rewards : {float, dict of float, None}
    rewarded_at : {datetime, None}
    """

    def __init__(self, history_id, context, recommendations, created_at,
                 rewarded_at=None):
        self.history_id = history_id
        self.context = context
        self.recommendations = recommendations
        self.created_at = created_at
        self.rewarded_at = rewarded_at

    def update_reward(self, rewards, rewarded_at):
        """Update reward_time and rewards.

        Parameters
        ----------
        rewards : {float, dict of float, None}
        rewarded_at : {datetime, None}
        """
        if not hasattr(self.recommendations, '__iter__'):
            recommendations = (self.recommendations,)
        else:
            recommendations = self.recommendations

        for rec in recommendations:
            try:
                rec.reward = rewards[rec.action.id]
            except KeyError:
                pass
        self.rewarded_at = rewarded_at

    @property
    def rewards(self):
        if not hasattr(self.recommendations, '__iter__'):
            recommendations = (self.recommendations,)
        else:
            recommendations = self.recommendations
        rewards = {}
        for rec in recommendations:
            if rec.reward is None:
                continue
            rewards[rec.action.id] = rec.reward
        return rewards


class HistoryStorage(object):
    """The object to store the history of context, recommendations and rewards.
    """
    @abstractmethod
    def get_history(self, history_id):
        """Get the previous context, recommendations and rewards with
        history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        history: History

        Raise
        -----
        KeyError
        """
        pass

    @abstractmethod
    def get_unrewarded_history(self, history_id):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        history: History

        Raise
        -----
        KeyError
        """
        pass

    @abstractmethod
    def add_history(self, context, recommendations, rewards=None):
        """Add a history record.

        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}

        Raise
        -----
        """
        pass

    @abstractmethod
    def add_reward(self, history_id, rewards):
        """Add reward to a history record.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}

        Raise
        -----
        """
        pass


class MemoryHistoryStorage(HistoryStorage):
    """HistoryStorage that store History objects in memory."""

    def __init__(self):
        self.histories = {}
        self.unrewarded_histories = {}
        self.n_histories = 0

    def get_history(self, history_id):
        """Get the previous context, recommendations and rewards with
        history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        history: History

        Raise
        -----
        KeyError
        """
        return self.histories[history_id]

    def get_unrewarded_history(self, history_id):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.

        Returns
        -------
        history: History

        Raise
        -----
        KeyError
        """
        return self.unrewarded_histories[history_id]

    def add_history(self, context, recommendations, rewards=None):
        """Add a history record.

        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}

        Raise
        -----
        """
        created_at = datetime.now()
        history_id = self.n_histories
        if rewards is None:
            history = History(history_id, context, recommendations, created_at)
            self.unrewarded_histories[history_id] = history
        else:
            rewarded_at = created_at
            history = History(history_id, context, recommendations, created_at,
                              rewards, rewarded_at)
            self.histories[history_id] = history
        self.n_histories += 1
        return history_id

    def add_reward(self, history_id, rewards):
        """Add reward to a history record.

        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}

        Raise
        -----
        """
        rewarded_at = datetime.now()
        history = self.unrewarded_histories.pop(history_id)
        history.update_reward(rewards, rewarded_at)
        self.histories[history.history_id] = history
