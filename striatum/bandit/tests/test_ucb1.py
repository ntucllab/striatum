import unittest
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import ucb1
from striatum.bandit.bandit import Action


class Ucb1(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        a1 = Action(1, 'a1', 'i love u')
        a2 = Action(2, 'a2', 'i hate u')
        a3 = Action(3, 'a3', 'i do not understand')
        a4 = Action(4, 'a4', 'i love u very much')
        a5 = Action(5, 'a5', 'i hate u very nuch')
        self.actions = [a1, a2, a3, a4, a5]
        self.alpha = 1.00

    def test_initialization(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        self.assertEqual(self.actions, policy._actions)

    def test_get_first_action(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None, n_actions=1)
        self.assertEqual(history_id, 0)
        self.assertIn(action[0]['action'], self.actions)

    def test_update_reward(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None, n_actions=1)
        policy.reward(history_id, {1: 0})
        self.assertEqual(policy._historystorage.get_history(history_id).reward, {1: 0})

    def test_model_storage(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None, n_actions=1)
        policy.reward(history_id, {action[0]['action'].action_id: 1.0})
        self.assertEqual(policy._modelstorage._model['empirical_reward'][action[0]['action'].action_id], 2)
        self.assertEqual(policy._modelstorage._model['action_times'][action[0]['action'].action_id], 2.0)
        self.assertEqual(policy._modelstorage._model['total_time'], 6.0)

    def test_delay_reward(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None, n_actions=1)
        history_id_2, action_2 = policy.get_action(context=None, n_actions=1)
        policy.reward(history_id, {1: 0})
        self.assertEqual(policy._historystorage.get_history(history_id).reward, {1: 0})
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None, n_actions=1)
        history_id_2, action_2 = policy.get_action(context=None, n_actions=2)
        policy.reward(history_id_2, {1: 0, 2: 0})
        self.assertEqual(policy._historystorage.get_history(history_id).reward, None)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, {1: 0, 2: 0})

    def test_add_action(self):
        policy = ucb1.UCB1(self.actions, self.historystorage, self.modelstorage)
        history_id, action = policy.get_action(context=None, n_actions=1)
        a6 = Action(6, 'a6', 'how are you?')
        a7 = Action(7, 'a7', 'i am fine')
        policy.add_action([a6, a7])
        policy.reward(history_id, {3: 1})
        self.assertEqual(len(policy._actions), 7)
        self.assertEqual(policy._actions_id, [1, 2, 3, 4, 5, 6, 7])


if __name__ == '__main__':
    unittest.main()
