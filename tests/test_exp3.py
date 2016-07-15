import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import exp3
import numpy as np
import unittest

class TestExp3(unittest.TestCase):
    def setUp(self):
        self.ModelStorage = model.MemoryModelStorage()
        self.HistoryStorage = history.MemoryHistoryStorage()
        self.actions = [1,2,3,4,5]
        self.gamma = 0.5

    def test_initialization(self):
        EXP3 = exp3.Exp3(self.actions, self.HistoryStorage, self.ModelStorage, self.gamma)

    def test_get_first_action(self):
        EXP3 = exp3.Exp3(self.actions, self.HistoryStorage, self.ModelStorage, self.gamma)
        history_id, action = EXP3.get_action([1, 1])
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)

    def test_update_reward(self):
        EXP3 = exp3.Exp3(self.actions, self.HistoryStorage, self.ModelStorage, self.gamma)
        history_id, action = EXP3.get_action([1, 1])
        EXP3.reward(history_id, 1.0)
        self.assertEqual(EXP3._HistoryStorage.get_history(history_id).reward, 1.0)

    def test_model_storage(self):
        EXP3 = exp3.Exp3(self.actions, self.HistoryStorage, self.ModelStorage, self.gamma)
        history_id, action = EXP3.get_action([1, 1])
        EXP3.reward(history_id, 1.0)
        self.assertEqual(len(EXP3._ModelStorage._model['w']), 5)
        self.assertEqual(len(EXP3._ModelStorage._model['query_vector']), 5)

    def test_delay_reward(self):
        EXP3 = exp3.Exp3(self.actions, self.HistoryStorage, self.ModelStorage, self.gamma)
        history_id, action = EXP3.get_action([1, 1])
        history_id_2, action_2 = EXP3.get_action([3,3])
        EXP3.reward(history_id, 1)
        self.assertTrue((EXP3 ._HistoryStorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((EXP3._HistoryStorage.get_history(history_id_2).context
                        == np.transpose(np.array([[3,3]]))).all())
        self.assertEqual(EXP3._HistoryStorage.get_history(history_id).reward, 1)
        self.assertEqual(EXP3._HistoryStorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        EXP3 = exp3.Exp3(self.actions, self.HistoryStorage, self.ModelStorage, self.gamma)
        history_id, action = EXP3.get_action([1,1])
        history_id_2, action_2 = EXP3.get_action([3,3])
        EXP3.reward(history_id_2, 1)
        self.assertTrue((EXP3._HistoryStorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((EXP3._HistoryStorage.get_history(history_id_2).context
                        == np.transpose(np.array([[3,3]]))).all())
        self.assertEqual(EXP3._HistoryStorage.get_history(history_id).reward, None)
        self.assertEqual(EXP3._HistoryStorage.get_history(history_id_2).reward, 1)

if __name__ == '__main__':
    unittest.main()
