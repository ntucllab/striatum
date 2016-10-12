"""Unit test for LinUCB
"""
import unittest

import numpy as np

from striatum.bandit import LinUCB
from striatum.storage import Action
from .base_bandit_test import BaseBanditTest, ChangeableActionSetBanditTest


class TestLinUCB(ChangeableActionSetBanditTest,
                 BaseBanditTest,
                 unittest.TestCase):
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

    def test_add_action(self):
        policy = self.policy
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, _ = policy.get_action(context1, 2)
        new_actions = [Action() for i in range(2)]
        policy.add_action(new_actions)
        policy.reward(history_id, {3: 1})
        for action in new_actions:
            self.assertTrue((policy._model_storage.get_model()['A'][action.id]
                             == np.identity(self.context_dimension)).all())

        context2 = {1: [1, 1], 2: [2, 2], 3: [3, 3], 4: [4, 4], 5: [5, 5]}
        history_id2, actions = policy.get_action(context2, 4)
        self.assertEqual(len(actions), 4)
        policy.reward(history_id2, {new_actions[0].id: 4, new_actions[1].id: 5})
        for action in new_actions:
            self.assertFalse((policy._model_storage.get_model()['A'][action.id]
                              == np.identity(2)).all())
