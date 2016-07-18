import numpy as np
import unittest
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import sys

sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import exp4p


class Exp4P(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        self.actions = [1, 2, 3, 4, 5]
        self.history_context = np.random.uniform(0, 5, (1000, 2))
        self.history_action = np.zeros(1000)
        for t in range(1000):
            for i in range(5):
                if i * 2 < sum(self.history_context[t, :]) <= (i + 1) * 2:
                    self.history_action[t] = self.actions[i]
        self.LogReg = OneVsRestClassifier(LogisticRegression())
        self.MNB = OneVsRestClassifier(MultinomialNB())
        self.LogReg.fit(self.history_context, self.history_action)
        self.MNB.fit(self.history_context, self.history_action)

    def test_initialization(self):
        policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage,
                             [self.LogReg, self.MNB], delta=0.1, pmin=None)
        self.assertEqual(self.actions, policy.actions)
        self.assertEqual(0.1, policy.delta)

    def test_get_first_action(self):
        policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage,
                             [self.LogReg, self.MNB], delta=0.1, pmin=None)
        history_id, action = policy.get_action([1, 1])
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)

    def test_update_reward(self):
        policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage,
                             [self.LogReg, self.MNB], delta=0.1, pmin=None)
        history_id, action = policy.get_action([1, 1])
        policy.reward(history_id, 1.0)
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1.0)

    def test_model_storage(self):
        policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage,
                             [self.LogReg, self.MNB], delta=0.1, pmin=None)
        history_id, action = policy.get_action([1, 1])
        policy.reward(history_id, 1.0)
        self.assertEqual(len(policy._modelstorage._model['w']), 2)
        self.assertEqual(len(policy._modelstorage._model['query_vector']), 5)
        self.assertEqual(np.shape(policy._modelstorage._model['advice']), (2, 5))

    def test_delay_reward(self):
        policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage,
                             [self.LogReg, self.MNB], delta=0.1, pmin=None)
        history_id, action = policy.get_action([1, 1])
        history_id_2, action_2 = policy.get_action([3, 3])
        policy.reward(history_id, 1)
        self.assertTrue(
            (policy._historystorage.get_history(history_id).context == np.transpose(np.array([[1, 1]]))).all())
        self.assertTrue(
            (policy._historystorage.get_history(history_id_2).context == np.transpose(np.array([[3, 3]]))).all())
        self.assertEqual(policy._historystorage.get_history(history_id).reward, 1)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage,
                             [self.LogReg, self.MNB], delta=0.1, pmin=None)
        history_id, action = policy.get_action([1, 1])
        history_id_2, action_2 = policy.get_action([3, 3])
        policy.reward(history_id_2, 1)
        self.assertTrue(
            (policy._historystorage.get_history(history_id).context == np.transpose(np.array([[1, 1]]))).all())
        self.assertTrue(
            (policy._historystorage.get_history(history_id_2).context == np.transpose(np.array([[3, 3]]))).all())
        self.assertEqual(policy._historystorage.get_history(history_id).reward, None)
        self.assertEqual(policy._historystorage.get_history(history_id_2).reward, 1)


if __name__ == '__main__':
    unittest.main()
