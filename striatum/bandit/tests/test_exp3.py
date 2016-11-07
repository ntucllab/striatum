import unittest

from striatum.bandit import Exp3
from striatum.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
)
from .base_bandit_test import BaseBanditTest, ChangeableActionSetBanditTest


class TestExp3(ChangeableActionSetBanditTest,
               BaseBanditTest,
               unittest.TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        super(TestExp3, self).setUp()
        self.gamma = 0.5
        self.policy = Exp3(self.history_storage, self.model_storage,
                           self.action_storage, gamma=self.gamma)
        self.policy_with_empty_action_storage = Exp3(
            MemoryHistoryStorage(), MemoryModelStorage(), MemoryActionStorage(),
            gamma=self.gamma)

    def test_initialization(self):
        super(TestExp3, self).test_initialization()
        policy = self.policy
        self.assertEqual(self.gamma, policy.gamma)

    def test_model_storage(self):
        policy = self.policy
        history_id, recommendations = policy.get_action(context=None,
                                                        n_actions=1)
        policy.reward(history_id, {recommendations[0].action.id: 1.0})
        model = policy._model_storage.get_model()
        self.assertEqual(len(model['w']), len(self.actions))
        self.assertGreater(model['w'][recommendations[0].action.id], 1.)

    def test_add_action(self):
        policy = self.policy
        history_id, recommendations = policy.get_action(context=None,
                                                        n_actions=2)
        new_actions = [Action() for i in range(2)]
        policy.add_action(new_actions)
        self.assertEqual(len(new_actions) + len(self.actions),
                         policy._action_storage.count())
        policy.reward(history_id, {recommendations[0].action.id: 1.})
        model = policy._model_storage.get_model()
        for action in new_actions:
            self.assertEqual(model['w'][action.id], 1.0)

        history_id2, recommendations2 = policy.get_action(context=None,
                                                          n_actions=-1)
        self.assertEqual(len(recommendations2),
                         len(new_actions) + len(self.actions))
        policy.reward(history_id2, {new_actions[0].id: 4, new_actions[1].id: 5})
        model = policy._model_storage.get_model()
        for action in new_actions:
            self.assertGreater(model['w'][action.id], 1.0)
