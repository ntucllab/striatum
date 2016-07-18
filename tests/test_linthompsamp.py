import unittest
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linthompsamp


class TestLinThompSamp(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        self.actions = [1, 2, 3]
        self.d = 2
        self.delta = 0.5
        self.R = 0.5
        self.epsilon = 0.1

    def test_initialization(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.d, self.delta, self.R, self.epsilon)
        self.assertEqual(self.actions, policy.actions)
        self.assertEqual(self.d, policy.d)
        self.assertEqual(self.R, policy.R)
        self.assertEqual(self.epsilon, policy.epsilon)

    def test_get_first_action(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)
        self.assertTrue((policy._historystorage.get_history(history_id).context == [[1, 1], [2, 2], [3, 3]]))

    def test_update_reward(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        policy.reward(history_id, 1.0)
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1)

    def test_model_storage(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        policy.reward(history_id, 1.0)
        self.assertTrue((policy._modelstorage._model['B'].shape == (2, 2)) == True)
        self.assertEqual(len(policy._modelstorage._model['muhat']), 2)
        self.assertEqual(len(policy._modelstorage._model['f']), 2)

    def test_delay_reward(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        history_id_2, action_2 = policy.get_action([[0, 1], [2, 3], [7, 5]])
        policy.reward(history_id, 1)
        self.assertEqual(policy._historystorage.get_history(history_id).context, [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(policy._historystorage.get_history(history_id_2).context, [[0, 1], [2, 3], [7, 5]])
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = policy.get_action([[1, 1], [2, 2], [3, 3]])
        history_id_2, action_2 = policy.get_action([[0, 1], [2, 3], [7, 5]])
        policy.reward(history_id_2, 1)
        self.assertEqual(policy._historystorage.get_history(history_id).context, [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(policy._historystorage.get_history(history_id_2).context, [[0, 1], [2, 3], [7, 5]])
        self.assertEqual(policy._historystorage.get_history(history_id).reward, None)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, 1)


if __name__ == '__main__':
    unittest.main()
