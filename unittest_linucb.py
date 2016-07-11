# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:04:26 2016

@author: user
"""

import striatum
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import linucb
import numpy as np
import unittest

class test_linucb(unittest.TestCase):
    def setUp(self):
        self.ModelStorage = model.MemoryModelStorage()
        self.HistoryStorage = history.MemoryHistoryStorage()
        self.actions = [1,2,3,4,5]
        self.alpha = 1.00
    
    def test_initialization(self):
        LINUCB = linucb.LinUCB(self.actions, self.HistoryStorage,
                               self.ModelStorage, 1.00, 2)                

    def test_get_first_action(self):
        LINUCB = linucb.LinUCB(self.actions, self.HistoryStorage,
                               self.ModelStorage, 1.00, 2)  
        history_id, action = LINUCB.get_action([1,1])
        self.assertEqual(history_id, 0)
        self.assertIn(action,self.actions)
        self.assertEqual(LINUCB.HistoryStorage.get_history(history_id).context,
                         np.transpose(np.array([[1,1]])))
        
    


if __name__ == '__main__':
    unittest.main() 