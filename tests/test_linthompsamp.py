import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linthompsamp
import numpy as np
import unittest

class TestLinThompSamp(unittest.TestCase):
    def setUp(self):
        self.ModelStorage = model.MemoryModelStorage()
        self.HistoryStorage = history.MemoryHistoryStorage()
        self.actions = [1,2,3]
        self.d = 2
        self.delta = 0.5
        self.R = 0.5
        self.epsilon = 0.1

    def test_initialization(self):
        LINTHOMPSAMP = linthompsamp.LinThompSamp(self.actions, self.HistoryStorage,
                                                self.ModelStorage, self.d, self.delta, self.R, self.epsilon)

    def test_get_first_action(self):
        LINTHOMPSAMP = linthompsamp.LinThompSamp(self.actions, self.HistoryStorage,
                                                 self.ModelStorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = LINTHOMPSAMP.get_action([[1,1],[2,2],[3,3]])
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)
        self.assertTrue((LINTHOMPSAMP._HistoryStorage.get_history(history_id).context
                        == [[1,1],[2,2],[3,3]]))

    '''def test_update_reward(self):
        LINTHOMPSAMP = linthompsamp.LinThompSamp(self.actions, self.HistoryStorage,
                                                 self.ModelStorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = LINTHOMPSAMP.get_action([1,1])
        LINTHOMPSAMP.reward(history_id, 1.0)
        self.assertEqual(LINTHOMPSAMP._HistoryStorage.get_history(history_id).reward, 1)

    def test_model_storage(self):
        LINTHOMPSAMP = linthompsamp.LinThompSamp(self.actions, self.HistoryStorage,
                                                 self.ModelStorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = LINTHOMPSAMP.get_action([1, 1])
        LINTHOMPSAMP.reward(history_id, 1.0)
        self.assertTrue((LINTHOMPSAMP._ModelStorage._model['B'].shape == (2,2))==True)
        self.assertEqual(len(LINTHOMPSAMP._ModelStorage._model['muhat']),2)
        self.assertEqual(len(LINTHOMPSAMP._ModelStorage._model['f']), 2)

    def test_delay_reward(self):
        LINTHOMPSAMP = linthompsamp.LinThompSamp(self.actions, self.HistoryStorage,
                                                 self.ModelStorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = LINTHOMPSAMP.get_action([1,1])
        history_id_2, action_2 = LINTHOMPSAMP.get_action([0,0])
        LINTHOMPSAMP.reward(history_id, 1)
        self.assertTrue((LINTHOMPSAMP._HistoryStorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((LINTHOMPSAMP._HistoryStorage.get_history(history_id_2).context
                        == np.transpose(np.array([[0,0]]))).all())
        self.assertEqual(LINTHOMPSAMP._HistoryStorage.get_history(history_id).reward, 1)
        self.assertEqual(LINTHOMPSAMP._HistoryStorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        LINTHOMPSAMP = linthompsamp.LinThompSamp(self.actions, self.HistoryStorage,
                                                 self.ModelStorage, self.d, self.delta, self.R, self.epsilon)
        history_id, action = LINTHOMPSAMP.get_action([1,1])
        history_id_2, action_2 = LINTHOMPSAMP.get_action([0,0])
        LINTHOMPSAMP.reward(history_id_2, 1)
        self.assertTrue((LINTHOMPSAMP._HistoryStorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((LINTHOMPSAMP._HistoryStorage.get_history(history_id_2).context
                        == np.transpose(np.array([[0,0]]))).all())
        self.assertEqual(LINTHOMPSAMP._HistoryStorage.get_history(history_id).reward, None)
        self.assertEqual(LINTHOMPSAMP._HistoryStorage.get_history(history_id_2).reward, 1)'''


if __name__ == '__main__':
    unittest.main()
