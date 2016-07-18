# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:04:26 2016

@author: David Huang
"""

import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linucb
import numpy as np
import unittest

class TestLinUcb(unittest.TestCase):
    def setUp(self):
        self.modelstorage = model.MemoryModelStorage()
        self.historystorage = history.MemoryHistoryStorage()
        self.actions = [1,2,3,4,5]
        self.alpha = 1.00

    def test_initialization(self):
        LINUCB = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)

    def test_get_first_action(self):
        LINUCB = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = LINUCB.get_action([1,1])
        self.assertEqual(history_id, 0)
        self.assertIn(action,self.actions)
        self.assertTrue((LINUCB._historystorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())

    def test_update_reward(self):
        LINUCB = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = LINUCB.get_action([1,1])
        LINUCB.reward(history_id, 1)
        self.assertEqual(LINUCB._historystorage.get_history(history_id).reward, 1)

    def test_model_storage(self):
        LINUCB = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = LINUCB.get_action([1,1])
        LINUCB.reward(history_id, 1)
        self.assertTrue((LINUCB._modelstorage._model['ba'][action]
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((LINUCB._modelstorage._model['Aa'][action]
                        == np.array([[2.,1.],[1.,2.]])).all())

    def test_delay_reward(self):
        LINUCB = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = LINUCB.get_action([1,1])
        history_id_2, action_2 = LINUCB.get_action([0,0])
        LINUCB.reward(history_id, 1)
        self.assertTrue((LINUCB._historystorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((LINUCB._historystorage.get_history(history_id_2).context
                        == np.transpose(np.array([[0,0]]))).all())
        self.assertEqual(LINUCB._historystorage.get_history(history_id).reward, 1)
        self.assertEqual(LINUCB._historystorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        LINUCB = linucb.LinUCB(self.actions, self.historystorage,
                               self.modelstorage, 1.00, 2)
        history_id, action = LINUCB.get_action([1,1])
        history_id_2, action_2 = LINUCB.get_action([0,0])
        LINUCB.reward(history_id_2, 1)
        self.assertTrue((LINUCB._historystorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((LINUCB._historystorage.get_history(history_id_2).context
                        == np.transpose(np.array([[0,0]]))).all())
        self.assertEqual(LINUCB._historystorage.get_history(history_id).reward, None)
        self.assertEqual(LINUCB._historystorage.get_history(history_id_2).reward, 1)


if __name__ == '__main__':
    unittest.main()
