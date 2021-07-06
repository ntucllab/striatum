import unittest

from striatum.bandit import LinThompSamp
from striatum.storage import (
    MemoryHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    Action,
)
from .base_bandit_test import BaseBanditTest, ChangeableActionSetBanditTest


class TestLinThompSamp(ChangeableActionSetBanditTest,
                       BaseBanditTest,
                       unittest.TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        super(TestLinThompSamp, self).setUp()
        self.context_dimension = 2
        self.delta = 0.5
        self.R = 0.5  # pylint: disable=invalid-name
        self.epsilon = 0.1
        self.policy = LinThompSamp(
            self.history_storage, self.model_storage,
            self.action_storage, context_dimension=self.context_dimension,
            delta=self.delta, R=self.R, epsilon=self.epsilon)
        self.policy_with_empty_action_storage = LinThompSamp(
            MemoryHistoryStorage(), MemoryModelStorage(), MemoryActionStorage(),
            context_dimension=self.context_dimension,
            delta=self.delta, R=self.R, epsilon=self.epsilon)

    def test_initialization(self):
        super(TestLinThompSamp, self).test_initialization()
        policy = self.policy
        self.assertEqual(self.context_dimension, policy.context_dimension)
        self.assertEqual(self.R, policy.R)
        self.assertEqual(self.epsilon, policy.epsilon)

    def test_model_storage(self):
        policy = self.policy
        model = self.policy._model_storage.get_model()
        context = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, _ = policy.get_action(context, 2)
        policy.reward(history_id, {2: 1, 3: 1})
        self.assertTupleEqual(model['B'].shape, (self.context_dimension,
                                                 self.context_dimension))
        self.assertEqual(len(model['mu_hat']), self.context_dimension)
        self.assertEqual(len(model['f']), self.context_dimension)

    def test_add_action(self):
        policy = self.policy
        context1 = {1: [1, 1], 2: [2, 2], 3: [3, 3]}
        history_id, _ = policy.get_action(context1, 2)
        new_actions = [Action() for i in range(2)]
        policy.add_action(new_actions)
        self.assertEqual(len(new_actions) + len(self.actions),
                         policy._action_storage.count())
        policy.reward(history_id, {3: 1})

        context2 = {1: [1, 1], 2: [2, 2], 3: [3, 3], 4: [4, 4], 5: [5, 5]}
        history_id2, recommendations = policy.get_action(context2, 4)
        self.assertEqual(len(recommendations), 4)
        policy.reward(history_id2, {new_actions[0].id: 4, new_actions[1].id: 5})
