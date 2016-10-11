"""Unit test for LinUCB
"""
import unittest

from striatum.bandit import LinUCB
from .base_bandit_test import BaseBanditTest


class TestLinUCB(BaseBanditTest, unittest.TestCase):
    #pylint: disable=protected-access

    def setUp(self):
        super(TestLinUCB, self).setUp()
        self.context_dimension = 2
        self.alpha = 1.
        self.policy = LinUCB(
            self.history_storage, self.model_storage,
            self.action_storage, self.alpha, self.context_dimension)

    def test_initialization(self):
        super(TestLinUCB, self).test_initialization()
        self.assertEqual(self.alpha, self.policy.alpha)

    def test_model_storage(self):
        model = self.policy._model_storage.get_model()
        self.assertEqual(len(model['b']), self.action_storage.count())
        self.assertEqual(len(model['b'][1]), self.context_dimension)
        self.assertEqual(len(model['A']), self.action_storage.count())
        self.assertEqual(model['A'][1].shape,
                         (self.context_dimension, self.context_dimension))
