import unittest

from striatum.bandit import UCB1
from striatum.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
)
from .base_bandit_test import BaseBanditTest, ChangeableActionSetBanditTest


class TestUCB1(ChangeableActionSetBanditTest,
               BaseBanditTest,
               unittest.TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        super(TestUCB1, self).setUp()
        self.policy = UCB1(
            self.history_storage, self.model_storage, self.action_storage)
        self.policy_with_empty_action_storage = UCB1(
            MemoryHistoryStorage(), MemoryModelStorage(), MemoryActionStorage())

    def test_model_storage(self):
        policy = self.policy
        history_id, recommendations = policy.get_action(context=None,
                                                        n_actions=1)
        policy.reward(history_id, {recommendations[0].action.id: 1.0})
        model = policy._model_storage.get_model()
        self.assertEqual(model['total_action_reward'][recommendations[0]
                                                      .action.id],
                         2.)
        self.assertEqual(model['action_times'][recommendations[0].action.id], 2)
        self.assertEqual(model['n_rounds'], len(self.actions) + 1)

    def test_add_action(self):
        policy = self.policy
        history_id, _ = policy.get_action(context=None, n_actions=2)
        new_actions = [Action() for i in range(2)]
        policy.add_action(new_actions)
        self.assertEqual(len(new_actions) + len(self.actions),
                         policy._action_storage.count())
        policy.reward(history_id, {3: 1})
        model = policy._model_storage.get_model()
        for action in new_actions:
            self.assertEqual(model['total_action_reward'][action.id], 1.0)
            self.assertEqual(model['action_times'][action.id], 1)
            self.assertEqual(model['n_rounds'],
                             len(self.actions) + len(new_actions) + 1)

        history_id2, recommendations = policy.get_action(context=None,
                                                         n_actions=4)
        self.assertEqual(len(recommendations), 4)
        policy.reward(history_id2, {new_actions[0].id: 4, new_actions[1].id: 5})
        model = policy._model_storage.get_model()
        for action in new_actions:
            self.assertNotEqual(model['total_action_reward'][action.id], 1.0)
            self.assertEqual(model['action_times'][action.id], 2)
            self.assertEqual(model['n_rounds'],
                             len(self.actions) + len(new_actions) + 1 + 2)
