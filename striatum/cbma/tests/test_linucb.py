import numpy as np
import unittest
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.cbma import linucb


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
        context = [[1, 1], [2, 2], [3, 3]]
        history_id, actions, score = policy.get_action(1, context)
        self.assertEqual(history_id, 0)
        self.assertEqual(policy._historystorage.get_history(history_id).context, [[1, 1], [2, 2], [3, 3]])

    def test_update_reward(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, actions, score = policy.get_action(1, [[1, 1], [2, 2], [3, 3]])
        policy.reward(history_id, {1: 1})
        self.assertEqual(policy._historystorage.get_history(history_id).reward, {1: 1})

    def test_model_storage(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action, score = policy.get_action(1, [[1, 1], [2, 2], [3, 3]])
        policy.reward(history_id, {1: 1})
        self.assertEqual(len(policy._modelstorage.get_model()['b']), 3)
        self.assertEqual(len(policy._modelstorage.get_model()['b'][1]), 2)
        self.assertEqual(len(policy._modelstorage.get_model()['matrix_a']), 3)
        self.assertEqual(policy._modelstorage.get_model()['matrix_a'][1].shape, (2, 2))

    def test_delay_reward(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action, score = policy.get_action(1, [[1, 1], [2, 2], [3, 3]])
        history_id_2, action_2, score2 = policy.get_action(2, [[0, 0], [3, 3], [6, 6]])
        policy.reward(history_id, {1: 1})
        self.assertEqual(policy._historystorage.get_history(history_id).context, [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(policy._historystorage.get_history(history_id_2).context, [[0, 0], [3, 3], [6, 6]])
        self.assertEqual(policy._historystorage.get_history(history_id).reward, {1: 1})
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action, score = policy.get_action(1, [[1, 1], [2, 2], [3, 3]])
        history_id_2, action_2, score2 = policy.get_action(2, [[0, 0], [3, 3], [6, 6]])
        policy.reward(history_id_2, {3: 1, 2: 0})
        self.assertEqual(policy._historystorage.get_history(history_id).context, [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(policy._historystorage.get_history(history_id_2).context, [[0, 0], [3, 3], [6, 6]])
        self.assertEqual(policy._historystorage.get_history(history_id).reward, None)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, {3: 1, 2: 0})

    def test_add_action(self):
        policy = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action, score = policy.get_action(1, [[1, 1], [2, 2], [3, 3]])
        policy.add_action([4, 5])
        policy.reward(history_id, {1: 1})
        self.assertEqual(policy._actions, [1, 2, 3, 4, 5])
        self.assertTrue((policy._modelstorage.get_model()['matrix_a'][4] == np.identity(2)).all())
        history_id_2, action_2, score_2 = policy.get_action(2, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        policy.reward(history_id_2, {5: 1, 4: 1})
        history_id_3, action_3, score_3 = policy.get_action(2, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        print(score_3)
        print(action_3)
        #policy.reward(history_id2, 1)
        #self.assertFalse((policy._modelstorage.get_model()['matrix_a'][action_2] == np.identity(2)).all())


if __name__ == '__main__':
    unittest.main()
