"""Unit test for LinUCB
"""

import unittest

import numpy as np

from striatum.bandit import linucb
from striatum.bandit.bandit import Action
from striatum.storage import history, model


class TestLinUcb(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        a1 = Action(1)
        a2 = Action(2)
        a3 = Action(3)
        self.actions = [a1, a2, a3]
        self.alpha = 1.00

    def test_initialization(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        self.assertEqual(self.actions, policy._actions)
        self.assertEqual(1.00, policy.alpha)
        self.assertEqual(2, policy.context_dimension)
        self.assertEqual([1, 2, 3], policy.action_ids)

    def test_get_first_action(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, action = policy.get_action(context, 1)
        self.assertEqual(history_id, 0)
        self.assertIn(action[0]['action'], self.actions)
        self.assertEqual(
            policy._historystorage.get_unrewarded_history(history_id).context, context)

    def test_update_reward(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, _ = policy.get_action(context, 1)
        policy.reward(history_id, {3: 1})
        self.assertEqual(
            policy._historystorage.get_history(history_id).reward, {3: 1})

    def test_model_storage(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, _ = policy.get_action(context, 2)
        policy.reward(history_id, {2: 1, 3: 1})
        self.assertEqual(len(policy._modelstorage.get_model()['b']), 3)
        self.assertEqual(len(policy._modelstorage.get_model()['b'][1]), 2)
        self.assertEqual(len(policy._modelstorage.get_model()['matrix_a']), 3)
        self.assertEqual(
            policy._modelstorage.get_model()['matrix_a'][1].shape, (2, 2))

    def test_delay_reward(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        context2 = {1: [0, 0], 2: [3, 3], 3: [6, 6]}
        history_id1, _ = policy.get_action(context1, 2)
        history_id2, _ = policy.get_action(context2, 1)
        policy.reward(history_id1, {2: 1, 3: 1})
        self.assertEqual(
            policy._historystorage.get_history(history_id1).context, context1)
        self.assertEqual(
            policy._historystorage.get_unrewarded_history(history_id2).context, context2)
        self.assertEqual(
            policy._historystorage.get_history(history_id1).reward,
            {2: 1, 3: 1})
        self.assertEqual(
            policy._historystorage.get_unrewarded_history(history_id2).reward, None)

    def test_reward_order_descending(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        context2 = {1: [0, 0], 2: [3, 3], 3: [6, 6]}
        history_id1, _ = policy.get_action(context1, 2)
        history_id2, _ = policy.get_action(context2, 1)
        policy.reward(history_id2, {3: 1})
        self.assertEqual(
            policy._historystorage.get_unrewarded_history(history_id1).context, context1)
        self.assertEqual(
            policy._historystorage.get_history(history_id2).context, context2)
        self.assertEqual(
            policy._historystorage.get_unrewarded_history(history_id1).reward, None)
        self.assertEqual(
            policy._historystorage.get_history(history_id2).reward, {3: 1})

    def test_add_action(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, _ = policy.get_action(context1, 2)
        a4 = Action(4)
        a5 = Action(5)
        policy.add_action([a4, a5])
        policy.reward(history_id, {3: 1})
        self.assertTrue((policy._modelstorage.get_model()['matrix_a'][4] == np.identity(2)).all())

        context2 = {1: [1, 1], 2: [2, 2], 3: [3, 3], 4: [4, 4], 5: [5, 5]}
        history_id2, _ = policy.get_action(context2, 1)
        policy.reward(history_id2, {4: 4, 5: 5})
        self.assertFalse((policy._modelstorage.get_model()['matrix_a'][4] == np.identity(2)).all())


if __name__ == '__main__':
    unittest.main()
