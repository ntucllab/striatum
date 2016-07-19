import numpy as np
import unittest
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linucb


class TestLinUcb(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        self.actions = [1, 2, 3]
        self.alpha = 1.00

    def test_initialization(self):
        policy = linucb.LinUCB(self.actions, self.historystorage, self.modelstorage, 1.00, 2)
        self.assertEqual(self.actions, policy._actions)
        self.assertEqual(1.00, policy.alpha)
        self.assertEqual(2, policy.d)

    def test_get_first_action(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)
        self.assertEqual(policy._historystorage.get_history(history_id).context, [[1, 1], [2, 2], [3, 3]])

    def test_update_reward(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        policy.reward(history_id, 1)
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1)

    def test_model_storage(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        policy.reward(history_id, 1)
        self.assertEqual(len(policy._modelstorage.get_model()['b']), 3)
        self.assertEqual(len(policy._modelstorage.get_model()['b'][1]), 2)
        self.assertEqual(len(policy._modelstorage.get_model()['matrix_a']), 3)
        self.assertEqual(policy._modelstorage.get_model()['matrix_a'][1].shape, (2, 2))

    def test_delay_reward(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        history_id_2, action_2 = policy.get_action([[0, 0], [3, 3], [6, 6]])
        policy.reward(history_id, 1)
        self.assertEqual(policy._historystorage.get_history(history_id).context, [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(policy._historystorage.get_history(history_id_2).context, [[0, 0], [3, 3], [6, 6]])
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        history_id_2, action_2 = policy.get_action([[0, 0], [3, 3], [6, 6]])
        policy.reward(history_id_2, 1)
        self.assertEqual(policy._historystorage.get_history(history_id).context, [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(policy._historystorage.get_history(history_id_2).context, [[0, 0], [3, 3], [6, 6]])
        self.assertEqual(policy._historystorage.get_history(history_id).reward, None)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, 1)


if __name__ == '__main__':
    unittest.main()
