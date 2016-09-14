import unittest
import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linthompsamp
from striatum.bandit.bandit import Action

class TestLinThompSamp(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        a1 = Action(1)
        a2 = Action(2)
        a3 = Action(3)
        self.actions = [a1, a2, a3]
        self.context_dimension = 2
        self.delta = 0.5
        self.R = 0.5
        self.epsilon = 0.1

    def test_initialization(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.context_dimension, self.delta, self.R, self.epsilon)
        self.assertEqual(self.actions, policy._actions)
        self.assertEqual(self.context_dimension, policy.context_dimension)
        self.assertEqual(self.R, policy.R)
        self.assertEqual(self.epsilon, policy.epsilon)

    def test_get_first_action(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.context_dimension, self.delta, self.R, self.epsilon)
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, action = policy.get_action(context, 1)
        self.assertEqual(history_id, 0)
        self.assertIn(action[0]['action'], self.actions)
        self.assertEqual(policy._historystorage.get_unrewarded_history(history_id).context, context)

    def test_update_reward(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.context_dimension, self.delta, self.R, self.epsilon)
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, action = policy.get_action(context, 1)
        policy.reward(history_id, {3: 1})
        self.assertEqual(policy._historystorage.get_history(history_id).reward, {3: 1})

    def test_model_storage(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.context_dimension, self.delta, self.R, self.epsilon)
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, action = policy.get_action(context, 2)
        policy.reward(history_id, {2: 1, 3: 1})
        self.assertTrue((policy._modelstorage._model['B'].shape == (2, 2)) == True)
        self.assertEqual(len(policy._modelstorage._model['mu_hat']), 2)
        self.assertEqual(len(policy._modelstorage._model['f']), 2)

    def test_delay_reward(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.context_dimension, self.delta, self.R, self.epsilon)
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        context2 = {1: [0, 0], 2: [3, 3], 3: [6, 6]}
        history_id1, action1 = policy.get_action(context1, 2)
        history_id2, action2 = policy.get_action(context2, 1)
        policy.reward(history_id1, {2: 1, 3: 1})
        self.assertEqual(policy._historystorage.get_history(history_id1).context, context1)
        self.assertEqual(policy._historystorage.get_unrewarded_history(history_id2).context, context2)
        self.assertEqual(policy._historystorage.get_history(history_id1).reward, {2: 1, 3: 1})
        self.assertEqual(policy._historystorage.get_unrewarded_history(history_id2).reward, None)

    def test_reward_order_descending(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.context_dimension, self.delta, self.R, self.epsilon)
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        context2 = {1: [0, 0], 2: [3, 3], 3: [6, 6]}
        history_id1, action1 = policy.get_action(context1, 2)
        history_id2, action2 = policy.get_action(context2, 1)
        policy.reward(history_id2, {3: 1})
        self.assertEqual(policy._historystorage.get_unrewarded_history(history_id1).context, context1)
        self.assertEqual(policy._historystorage.get_history(history_id2).context, context2)
        self.assertEqual(policy._historystorage.get_unrewarded_history(history_id1).reward, None)
        self.assertEqual(policy._historystorage.get_history(history_id2).reward, {3: 1})

    def test_add_action(self):
        policy = linthompsamp.LinThompSamp(self.actions, self.historystorage,
                                           self.modelstorage, self.context_dimension, self.delta, self.R, self.epsilon)
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, action = policy.get_action(context1, 2)
        a4 = Action(4)
        a5 = Action(5)
        policy.add_action([a4, a5])
        policy.reward(history_id, {3: 1})
        self.assertEqual(len(policy._actions), 5)
        self.assertEqual(policy.action_ids, [1, 2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
