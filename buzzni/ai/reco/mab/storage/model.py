"""
Model storage
"""
from abc import abstractmethod


class ModelStorage(object):
    """The object to store the model."""
    @abstractmethod
    def get_model(self):
        """Get model"""
        pass

    @abstractmethod
    def save_model(self):
        """Save model"""
        pass


class MemoryModelStorage(ModelStorage):
    """Store the model in memory."""
    def __init__(self):
        self._model = None

    def get_model(self):
        return self._model

    def save_model(self, model):
        self._model = model
