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
        self.actions = [Action(i + 1) for i in range(3)]
        self.action_storage.add(self.actions)

    def test_initialization(self):
        policy = self.policy
        self.assertEqual(self.model_storage, policy._model_storage)
        self.assertEqual(self.history_storage, policy._history_storage)
        self.assertEqual(self.history_storage, policy.history_storage)
        self.assertEqual(self.action_storage, policy._action_storage)

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

    def test_get_action_with_n_actions_none(self):
        policy = self.policy
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, action = policy.get_action(context, None)
        self.assertEqual(history_id, 0)
        self.assertIsInstance(action, dict)
        self.assertIn(action['action'].id, self.action_storage.iterids())
        self.assertEqual(
            policy._history_storage.get_unrewarded_history(history_id).context,
            context)

    def test_get_all_action(self):
        policy = self.policy
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, actions = policy.get_action(context, -1)
        self.assertEqual(history_id, 0)
        self.assertEqual(len(actions), len(self.actions))
        for action in actions:
            self.assertIn(action['action'].id, self.action_storage.iterids())
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
            policy._history_storage.get_history(history_id).rewards,
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
            policy._history_storage.get_history(history_id1).rewards,
            {2: 1, 3: 1})
        self.assertIsNone(
            policy._history_storage.get_unrewarded_history(history_id2).rewards)

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
            policy._history_storage.get_unrewarded_history(history_id1).rewards)
        self.assertDictEqual(
            policy._history_storage.get_history(history_id2).rewards, {3: 1})


class ChangeableActionSetBanditTest(object):

    def test_add_action_change_storage(self):
        policy = self.policy
        new_actions = [Action() for i in range(2)]
        policy.add_action(new_actions)
        self.assertEqual(set(a.id for a in self.actions + new_actions),
                         set(self.action_storage.iterids()))
