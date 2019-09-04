"""Unit test for LinUCB with Hybrid Linear Models"""

import unittest
import numpy as np
from striatum.bandit import HybLinUCB
from striatum.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action
)
from .base_bandit_test import BaseBanditTest, ChangeableActionSetBanditTest


class TestHybLinUCB(ChangeableActionSetBanditTest, BaseBanditTest, unittest.TestCase):

    def setUp(self):
        super(TestHybLinUCB, self).setUp()
        self.shared_context_dimension = 1
        self.unshared_context_dimension = 1
        self.alpha = 1.
        self.policy = HybLinUCB(self.history_storage, self.model_storage, self.action_storage,
                                shared_context_dimension=self.shared_context_dimension,
                                unshared_context_dimension=self.unshared_context_dimension,
                                alpha=self.alpha)
        self.policy_with_empty_action_storage = HybLinUCB(
            MemoryHistoryStorage(), MemoryModelStorage(), MemoryActionStorage(),
            shared_context_dimension=self.shared_context_dimension,
            unshared_context_dimension=self.unshared_context_dimension,
            alpha=self.alpha)

    def test_initialization(self):
        super(TestHybLinUCB, self).test_initialization()
        policy = self.policy
        self.assertEqual(self.shared_context_dimension, policy.shared_context_dimension)
        self.assertEqual(self.unshared_context_dimension, policy.unshared_context_dimension)
        self.assertEqual(self.alpha, policy.alpha)
        self.assertEqual(self.n_jobs, policy.n_jobs)

    def test_model_storage(self):
        model = self.policy._model_storage.get_model()
        self.assertEqual(len(model['b0']), self.shared_context_dimension)
        self.assertEqual(model['A0'].shape, (self.shared_context_dimension, self.shared_context_dimension))
        self.assertEqual(len(model['b']), self.action_storage.count())
        self.assertEqual(len(model['b'][1]), self.unshared_context_dimension)
        self.assertEqual(len(model['B']), self.action_storage.count())
        self.assertEqual(model['B'][1].shape, (self.unshared_context_dimension, self.shared_context_dimension))
        self.assertEqual(len(model['A']), self.action_storage.count())
        self.assertEqual(model['A'][1].shape, (self.unshared_context_dimension, self.unshared_context_dimension))

    def test_add_action(self):
        policy = self.policy
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, _ = policy.get_action(context1, 2)
        new_actions = [Action() for _ in range(2)]
        policy.add_action(new_actions)
        self.assertEqual(len(new_actions) + len(self.actions),
                         policy._action_storage.count())
        policy.reward(history_id, {3: 1})
        model = policy._model_storage.get_model()
        for action in new_actions:
            self.assertTrue((model['A'][action.id] == np.identity(self.unshared_context_dimension)).all())
            self.assertTrue((model['B'][action.id]
                             == np.zeros((self.unshared_context_dimension, self.shared_context_dimension))).all())

        context2 = {1: [1, 1], 2: [2, 2], 3: [3, 3], 4: [4, 4], 5: [5, 5]}
        history_id2, recommendations = policy.get_action(context2, 4)
        self.assertEqual(len(recommendations), 4)
        policy.reward(history_id2, {new_actions[0].id: 4, new_actions[1].id: 5})
        model = policy._model_storage.get_model()
        for action in new_actions:
            self.assertFalse((model['A'][action.id] == np.identity(self.unshared_context_dimension)).all())
            self.assertFalse((model['B'][action.id]
                             == np.zeros((self.unshared_context_dimension, self.shared_context_dimension))).all())
