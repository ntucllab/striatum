"""Unit test for LinUCB
"""
from striatum.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
    Recommendation,
)


class BaseBanditTest(object):
    # pylint: disable=protected-access

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

    def test_get_action_with_empty_storage(self):
        policy = self.policy_with_empty_action_storage
        context = {}
        history_id, recommendations = policy.get_action(context, 1)
        self.assertEqual(history_id, 0)
        self.assertEqual(len(recommendations), 0)
        self.assertDictEqual(
            policy._history_storage.get_unrewarded_history(history_id).context,
            context)

    def test_get_first_action(self):
        policy = self.policy
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, recommendations = policy.get_action(context, 1)
        self.assertEqual(history_id, 0)
        self.assertEqual(len(recommendations), 1)
        self.assertIn(recommendations[0].action.id,
                      self.action_storage.iterids())
        self.assertEqual(
            policy._history_storage.get_unrewarded_history(history_id).context,
            context)

    def test_get_action_with_n_actions_none(self):
        policy = self.policy
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, recommendations = policy.get_action(context, None)
        self.assertEqual(history_id, 0)
        self.assertIsInstance(recommendations, Recommendation)
        self.assertIn(recommendations.action.id,
                      self.action_storage.iterids())
        self.assertEqual(
            policy._history_storage.get_unrewarded_history(history_id).context,
            context)

    def test_get_all_action(self):
        policy = self.policy
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, recommendations = policy.get_action(context, -1)
        self.assertEqual(history_id, 0)
        self.assertEqual(len(recommendations), len(self.actions))
        for rec in recommendations:
            self.assertIn(rec.action.id, self.action_storage.iterids())
        self.assertEqual(
            policy._history_storage.get_unrewarded_history(history_id).context,
            context)

    def test_get_multiple_action(self):
        policy = self.policy
        n_actions = 2
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, recommendations = policy.get_action(context, n_actions)
        self.assertEqual(history_id, 0)
        self.assertEqual(len(recommendations), n_actions)
        for rec in recommendations:
            self.assertIn(rec.action.id, self.action_storage.iterids())
        self.assertEqual(
            policy._history_storage.get_unrewarded_history(history_id).context,
            context)

    def test_update_reward(self):
        policy = self.policy
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, recommendations = policy.get_action(context, 1)
        rewards = {recommendations[0].action.id: 1.}
        policy.reward(history_id, rewards)
        self.assertEqual(
            policy._history_storage.get_history(history_id).rewards, rewards)

    def test_delay_reward(self):
        policy = self.policy
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        context2 = {1: [0, 0], 2: [3, 3], 3: [6, 6]}
        history_id1, recommendations1 = policy.get_action(context1, 2)
        self.assertEqual(len(recommendations1), 2)
        history_id2, recommendations2 = policy.get_action(context2, 1)
        self.assertEqual(len(recommendations2), 1)

        rewards = {
            recommendations1[0].action.id: 0.,
            recommendations1[1].action.id: 1.,
        }
        policy.reward(history_id1, rewards)
        self.assertDictEqual(
            policy._history_storage.get_history(history_id1).context, context1)
        self.assertDictEqual(
            policy._history_storage.get_unrewarded_history(history_id2).context,
            context2)
        self.assertDictEqual(
            policy._history_storage.get_history(history_id1).rewards, rewards)
        self.assertDictEqual(
            policy._history_storage.get_unrewarded_history(history_id2).rewards,
            {})

    def test_reward_order_descending(self):
        policy = self.policy
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        context2 = {1: [0, 0], 2: [3, 3], 3: [6, 6]}
        history_id1, _ = policy.get_action(context1, 2)
        history_id2, recommendations2 = policy.get_action(context2)
        rewards = {recommendations2.action.id: 1.}
        policy.reward(history_id2, rewards)
        self.assertDictEqual(
            policy._history_storage.get_unrewarded_history(history_id1).context,
            context1)
        self.assertDictEqual(
            policy._history_storage.get_history(history_id2).context, context2)
        self.assertDictEqual(
            policy._history_storage.get_unrewarded_history(history_id1).rewards,
            {})
        self.assertDictEqual(
            policy._history_storage.get_history(history_id2).rewards, rewards)

    def test_update_action(self):
        action = self.actions[1]
        action.action_type = "text"
        action.action_text = "hello"
        self.policy.update_action(action)
        updated_action = self.action_storage.get(action.id)
        self.assertEqual(updated_action.action_type, action.action_type)
        self.assertEqual(updated_action.action_text, action.action_text)


class ChangeableActionSetBanditTest(object):
    # pylint: disable=protected-access

    def test_add_action_change_storage(self):
        policy = self.policy
        new_actions = [Action() for i in range(2)]
        policy.add_action(new_actions)
        self.assertEqual(set(a.id for a in self.actions + new_actions),
                         set(self.action_storage.iterids()))

    def test_add_action_from_empty_change_storage(self):
        policy = self.policy_with_empty_action_storage
        new_actions = [Action() for i in range(2)]
        policy.add_action(new_actions)
        self.assertEqual(set(a.id for a in new_actions),
                         set(policy._action_storage.iterids()))

    def test_remove_action_change_storage(self):
        policy = self.policy
        removed_action = self.actions[1]
        policy.remove_action(removed_action.id)
        new_action_ids = set(a.id for a in self.actions
                             if a.id != removed_action.id)
        self.assertEqual(new_action_ids,
                         set(self.action_storage.iterids()))

    def test_remove_and_get_action_and_reward(self):
        policy = self.policy
        removed_action = self.actions[1]
        policy.remove_action(removed_action.id)

        context = {1: [1, 1], 3: [3, 3]}
        history_id, recommendations = policy.get_action(context, 1)
        self.assertEqual(history_id, 0)
        self.assertEqual(len(recommendations), 1)
        self.assertIn(recommendations[0].action.id,
                      self.action_storage.iterids())

        rewards = {recommendations[0].action.id: 1.}
        policy.reward(history_id, rewards)
        self.assertEqual(
            policy._history_storage.get_history(history_id).rewards, rewards)
