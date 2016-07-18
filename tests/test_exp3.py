import numpy as np
import unittest
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import exp3


class TestExp3(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        self.actions = [1, 2, 3, 4, 5]
        self.gamma = 0.5

    def test_initialization(self):
        policy = exp3.Exp3(self.actions, self.historystorage, self.modelstorage, self.gamma)
        self.assertEqual(self.actions, policy.actions)
        self.assertEqual(0.5, policy.gamma)

    def test_get_first_action(self):
        policy = exp3.Exp3(self.actions, self.historystorage, self.modelstorage, self.gamma)
        history_id, action = policy.get_action([1, 1])
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)

    def test_update_reward(self):
        policy = exp3.Exp3(self.actions, self.historystorage, self.modelstorage, self.gamma)
        history_id, action = policy.get_action([1, 1])
        policy.reward(history_id, 1.0)
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1.0)

    def test_model_storage(self):
        policy = exp3.Exp3(self.actions, self.historystorage, self.modelstorage, self.gamma)
        history_id, action = policy.get_action([1, 1])
        policy.reward(history_id, 1.0)
        self.assertEqual(len(policy._modelstorage._model['w']), 5)
        self.assertEqual(len(policy._modelstorage._model['query_vector']), 5)

    def test_delay_reward(self):
        policy = exp3.Exp3(self.actions, self.historystorage, self.modelstorage, self.gamma)
        history_id, action = policy.get_action([1, 1])
        history_id_2, action_2 = policy.get_action([3, 3])
        policy.reward(history_id, 1)
        self.assertTrue(
            (policy._historystorage.get_history(history_id).context == np.transpose(np.array([[1, 1]]))).all())
        self.assertTrue(
            (policy._historystorage.get_history(history_id_2).context == np.transpose(np.array([[3, 3]]))).all())
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        policy = exp3.Exp3(self.actions, self.historystorage, self.modelstorage, self.gamma)
        history_id, action = policy.get_action([1, 1])
        history_id_2, action_2 = policy.get_action([3, 3])
        policy.reward(history_id_2, 1)
        self.assertTrue(
            (policy._historystorage.get_history(history_id).context == np.transpose(np.array([[1, 1]]))).all())
        self.assertTrue(
            (policy._historystorage.get_history(history_id_2).context == np.transpose(np.array([[3, 3]]))).all())
        self.assertEqual(policy._historystorage.get_history(history_id).reward, None)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, 1)


if __name__ == '__main__':
    unittest.main()
