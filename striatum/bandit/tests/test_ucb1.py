import unittest
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import ucb1


class Ucb1(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        self.actions = [1, 2, 3, 4, 5]
        self.alpha = 1.00

    def test_initialization(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        self.assertEqual(self.actions, policy._actions)

    def test_get_first_action(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None)
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)

    def test_update_reward(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None)
        policy.reward(history_id, 1)
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1)

    def test_model_storage(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None)
        policy.reward(history_id, 1)
        self.assertEqual(policy._modelstorage._model['empirical_reward'][action], 2)
        self.assertEqual(policy._modelstorage._model['n_actions'][action], 2.0)
        self.assertEqual(policy._modelstorage._model['n_total'], 6.0)

    def test_delay_reward(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None)
        history_id_2, action_2 = policy.get_action(context=None)
        policy.reward(history_id, 1)
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1.0)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None)
        history_id_2, action_2 = policy.get_action(context=None)
        policy.reward(history_id_2, 1)
        self.assertEqual(policy._historystorage.get_history(history_id).reward, None)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, 1.0)


if __name__ == '__main__':
    unittest.main()
