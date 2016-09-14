"""Unit test for Exp3
"""

import unittest

from striatum.bandit import exp3
from striatum.bandit.bandit import Action
from striatum.storage import history, model


class TestExp3(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        a1 = Action(1)
        a2 = Action(2)
        a3 = Action(3)
        a4 = Action(4)
        a5 = Action(5)
        self.actions = [a1, a2, a3, a4, a5]
        self.gamma = 0.5

    def test_initialization(self):
        policy = exp3.Exp3(self.actions, self.historystorage,
                           self.modelstorage, self.gamma)
        self.assertEqual(self.actions, policy.actions)
        self.assertEqual(0.5, policy.gamma)

    def test_get_first_action(self):
        policy = exp3.Exp3(self.actions, self.historystorage,
                           self.modelstorage, self.gamma)
        history_id, action = policy.get_action(context=None, n_actions=1)
        self.assertEqual(history_id, 0)
        self.assertIn(action[0]['action'], self.actions)

    def test_update_reward(self):
        policy = exp3.Exp3(self.actions, self.historystorage,
                           self.modelstorage, self.gamma)
        history_id, _ = policy.get_action(context=None, n_actions=1)
        policy.reward(history_id, {1: 1.0})
        self.assertEqual(policy._historystorage.get_history(history_id).reward,
                         {1: 1.0})

    def test_model_storage(self):
        policy = exp3.Exp3(self.actions, self.historystorage,
                           self.modelstorage, self.gamma)
        history_id, _ = policy.get_action(context=None, n_actions=1)
        policy.reward(history_id, {1: 1.0})
        self.assertEqual(len(policy._modelstorage._model['w']), 5)
        self.assertEqual(len(policy._modelstorage._model['query_vector']), 5)

    def test_delay_reward(self):
        policy = exp3.Exp3(self.actions, self.historystorage,
                           self.modelstorage, self.gamma)
        history_id1, action1 = policy.get_action(context=None, n_actions=1)
        history_id2, action2 = policy.get_action(context=None, n_actions=1)
        policy.reward(history_id1, {1: 1.0})
        self.assertEqual(policy._historystorage.get_history(history_id1).reward, {1: 1.0})
        self.assertEqual(policy._historystorage.get_unrewarded_history(history_id2).reward, None)

    def test_reward_order_descending(self):
        policy = exp3.Exp3(self.actions, self.historystorage,
                           self.modelstorage, self.gamma)
        history_id1, _ = policy.get_action(context=None, n_actions=1)
        history_id2, _ = policy.get_action(context=None, n_actions=1)
        policy.reward(history_id2, {1: 1.0, 2: 0.0})
        self.assertEqual(policy._historystorage.get_unrewarded_history(history_id1).reward,
                         None)
        self.assertEqual(policy._historystorage.get_history(history_id2).reward,
                         {1: 1.0, 2: 0.0})

    def test_add_action(self):
        policy = exp3.Exp3(self.actions, self.historystorage,
                           self.modelstorage, self.gamma)
        history_id, _ = policy.get_action(context=None, n_actions=1)
        a6 = Action(6)
        a7 = Action(7)
        policy.add_action([a6, a7])
        policy.reward(history_id, {3: 1})
        self.assertEqual(len(policy._actions), 7)
        self.assertEqual(policy.action_ids, [1, 2, 3, 4, 5, 6, 7])


if __name__ == '__main__':
    unittest.main()
