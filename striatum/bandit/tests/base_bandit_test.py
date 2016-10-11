"""Unit test for LinUCB
"""
from striatum.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
)


class BaseBanditTest(object):
    #pylint: disable=protected-access

    def setUp(self):  # pylint: disable=invalid-name
        self.model_storage = MemoryModelStorage()
        self.history_storage = MemoryHistoryStorage()
        self.action_storage = MemoryActionStorage()
        self.action_storage.add([Action(i + 1) for i in range(3)])

    def test_initialization(self):
        policy = self.policy
        self.assertEqual(self.model_storage, policy._model_storage)
        self.assertEqual(self.history_storage, policy._history_storage)
        self.assertEqual(self.history_storage, policy.history_storage)
        self.assertEqual(self.action_storage, policy._action_storage)
        self.assertEqual(self.context_dimension, policy.context_dimension)

    def test_get_first_action(self):
        policy = self.policy
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, action = policy.get_action(context, 1)
        self.assertEqual(history_id, 0)
        self.assertEqual(len(action), 1)
        self.assertIn(action[0]['action'].id, self.action_storage.iterids())
        self.assertEqual(
            policy._history_storage.get_unrewarded_history(history_id).context,
            context)

    def test_get_multiple_action(self):
        policy = self.policy
        n_actions = 2
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, actions = policy.get_action(context, n_actions)
        self.assertEqual(history_id, 0)
        self.assertEqual(len(actions), n_actions)
        for action in actions:
            self.assertIn(action['action'].id, self.action_storage.iterids())
        self.assertEqual(
            policy._history_storage.get_unrewarded_history(history_id).context,
            context)

    def test_update_reward(self):
        policy = self.policy
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, _ = policy.get_action(context, 1)
        policy.reward(history_id, {3: 1})
        self.assertEqual(
            policy._history_storage.get_history(history_id).reward,
            {3: 1})

    def test_delay_reward(self):
        policy = self.policy
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        context2 = {1: [0, 0], 2: [3, 3], 3: [6, 6]}
        history_id1, actions1 = policy.get_action(context1, 2)
        self.assertEqual(len(actions1), 2)
        history_id2, actions2 = policy.get_action(context2, 1)
        self.assertEqual(len(actions2), 1)
        policy.reward(history_id1, {2: 1, 3: 1})
        self.assertDictEqual(
            policy._history_storage.get_history(history_id1).context, context1)
        self.assertDictEqual(
            policy._history_storage.get_unrewarded_history(history_id2).context,
            context2)
        self.assertDictEqual(
            policy._history_storage.get_history(history_id1).reward,
            {2: 1, 3: 1})
        self.assertIsNone(
            policy._history_storage.get_unrewarded_history(history_id2).reward)

    def test_reward_order_descending(self):
        policy = self.policy
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        context2 = {1: [0, 0], 2: [3, 3], 3: [6, 6]}
        history_id1, _ = policy.get_action(context1, 2)
        history_id2, _ = policy.get_action(context2, 1)
        policy.reward(history_id2, {3: 1})
        self.assertDictEqual(
            policy._history_storage.get_unrewarded_history(history_id1).context,
            context1)
        self.assertDictEqual(
            policy._history_storage.get_history(history_id2).context, context2)
        self.assertIsNone(
            policy._history_storage.get_unrewarded_history(history_id1).reward)
        self.assertDictEqual(
            policy._history_storage.get_history(history_id2).reward, {3: 1})
