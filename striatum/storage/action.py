"""
Action storage
"""
from abc import abstractmethod
from copy import deepcopy


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

    def add(self, action):
        r"""Add action
        Parameters
        ----------
        action: Action object
            The Action object to add.

        Raises
        ------
        KeyError
        """
        pass

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


class MemoryActionStorage(object):
    def __init__(self):
        self.actions = {}
        self.next_action_id = 0

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
        return deepcopy(self.actions[action_id])

    def add(self, action):
        r"""Add action
        Parameters
        ----------
        action: Action object
            The Action object to add.

        Raises
        ------
        KeyError
        """
        if action.id is None:
            action.id = self.next_action_id
            self.next_action_id += 1
        elif action.id in self.actions:
            raise KeyError("Action id {} exists".format(action.id))
        self.actions[action.id] = action

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
        self.actions[action.id] = action

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
        del self.actions[action_id]

