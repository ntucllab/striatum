"""
Action storage
"""
from abc import abstractmethod
from copy import deepcopy

import six


class Action(object):
    r"""The action object

    Parameters
    ----------
    action_id: int
        The index of this action.
    """

    def __init__(self, action_id=None):
        self.id = action_id


class ActionStorage(object):

    @abstractmethod
    def get(self, action_id):
        r"""Get action by action id

        Parameters
        ----------
        action_id: int
            The id of the action.

        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        pass

    @abstractmethod
    def add(self, action):
        r"""Add action

        Parameters
        ----------
        action: Action object
            The Action object to add.

        Raises
        ------
        KeyError

        Returns
        -------
        new_action_ids: list of int
            The Action ids of the added Actions.
        """
        pass

    @abstractmethod
    def update(self, action):
        r"""Add action

        Parameters
        ----------
        action: Action object
            The Action object to update.

        Raises
        ------
        KeyError
        """
        pass

    @abstractmethod
    def remove(self, action_id):
        r"""Add action

        Parameters
        ----------
        action_id: int
            The Action id to remove.

        Raises
        ------
        KeyError
        """
        pass

    @abstractmethod
    def count(self):
        r"""Count actions
        """
        pass

    @abstractmethod
    def iterids(self):
        r"""Return iterable of the Action ids.

        Returns
        -------
        action_ids: iterable
            Action ids.
        """


class MemoryActionStorage(object):

    def __init__(self):
        self._actions = {}
        self._next_action_id = 0

    def get(self, action_id):
        r"""Get action by action id

        Parameters
        ----------
        action_id: int
            The id of the action.

        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        return deepcopy(self._actions[action_id])

    def add(self, actions):
        r"""Add actions

        Parameters
        ----------
        action: list of Action objects
            The list of Action objects to add.

        Raises
        ------
        KeyError

        Returns
        -------
        new_action_ids: list of int
            The Action ids of the added Actions.
        """
        new_action_ids = []
        for action in actions:
            if action.id is None:
                action.id = self._next_action_id
                self._next_action_id += 1
            elif action.id in self._actions:
                raise KeyError("Action id {} exists".format(action.id))
            else:
                self._next_action_id = max(self._next_action_id, action.id + 1)
            self._actions[action.id] = action
            new_action_ids.append(action.id)
        return new_action_ids

    def update(self, action):
        r"""Update action

        Parameters
        ----------
        action: Action object
            The Action object to update.

        Raises
        ------
        KeyError
        """
        self._actions[action.id] = action

    def remove(self, action_id):
        r"""Remove action

        Parameters
        ----------
        action_id: int
            The Action id to remove.

        Raises
        ------
        KeyError
        """
        del self._actions[action_id]

    def count(self):
        r"""Count actions

        Returns
        -------
        count: int
            Number of Action in the storage.
        """
        return len(self._actions)

    def iterids(self):
        r"""Return iterable of the Action ids.

        Returns
        -------
        action_ids: iterable
            Action ids.
        """
        return six.viewkeys(self._actions)

    def __iter__(self):
        return iter(six.viewvalues(self._actions))
