# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 18:04:26 2016

@author: David Huang
"""

import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import exp4p
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
import unittest

class Exp4P(unittest.TestCase):
    def setUp(self):
        self.ModelStorage = model.MemoryModelStorage()
        self.HistoryStorage = history.MemoryHistoryStorage()
        self.actions = [1,2,3,4,5]
        self.history_context = np.random.uniform(0, 5, (1000, 2))
        self.history_action = np.zeros(1000)
        for t in range(1000):
            for i in range(5):
                if (i *  2 < sum(self.history_context[t, :]) <= (i + 1) * 2):
                    self.history_action[t] = self.actions[i]
        self.LogReg = OneVsRestClassifier(LogisticRegression())
        self.MNB = OneVsRestClassifier(MultinomialNB())
        self.LogReg.fit(self.history_context, self.history_action)
        self.MNB.fit(self.history_context, self.history_action)

    def test_initialization(self):
        EXP4P = exp4p.Exp4P(self.actions, self.HistoryStorage, self.ModelStorage,
                          [self.LogReg, self.MNB], delta=0.1, pmin=None)

    def test_get_first_action(self):
        EXP4P = exp4p.Exp4P(self.actions, self.HistoryStorage, self.ModelStorage,
                            [self.LogReg, self.MNB], delta=0.1, pmin=None)
        history_id, action = EXP4P.get_action([1, 1])
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)

    def test_update_reward(self):
        EXP4P = exp4p.Exp4P(self.actions, self.HistoryStorage, self.ModelStorage,
                            [self.LogReg, self.MNB], delta=0.1, pmin=None)
        history_id, action = EXP4P.get_action([1, 1])
        EXP4P.reward(history_id, 1.0)
        self.assertEqual(EXP4P._HistoryStorage.get_history(history_id).reward, 1.0)

    def test_model_storage(self):
        EXP4P = exp4p.Exp4P(self.actions, self.HistoryStorage, self.ModelStorage,
                            [self.LogReg, self.MNB], delta=0.1, pmin=None)
        history_id, action = EXP4P.get_action([1, 1])
        EXP4P.reward(history_id, 1.0)
        self.assertEqual(len(EXP4P._ModelStorage._model['w']), 2)
        self.assertEqual(len(EXP4P._ModelStorage._model['query_vector']), 5)
        self.assertEqual(np.shape(EXP4P._ModelStorage._model['advice']), (2,5))

    '''def test_delay_reward(self):
        LINUCB = linucb.LinUCB(self.actions, self.HistoryStorage,
                               self.ModelStorage, 1.00, 2)
        history_id, action = LINUCB.get_action([1,1])
        history_id_2, action_2 = LINUCB.get_action([0,0])
        LINUCB.reward(history_id, 1)
        self.assertTrue((LINUCB._HistoryStorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((LINUCB._HistoryStorage.get_history(history_id_2).context
                        == np.transpose(np.array([[0,0]]))).all())
        self.assertEqual(LINUCB._HistoryStorage.get_history(history_id).reward, 1)
        self.assertEqual(LINUCB._HistoryStorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        LINUCB = linucb.LinUCB(self.actions, self.HistoryStorage,
                               self.ModelStorage, 1.00, 2)
        history_id, action = LINUCB.get_action([1,1])
        history_id_2, action_2 = LINUCB.get_action([0,0])
        LINUCB.reward(history_id_2, 1)
        self.assertTrue((LINUCB._HistoryStorage.get_history(history_id).context
                        == np.transpose(np.array([[1,1]]))).all())
        self.assertTrue((LINUCB._HistoryStorage.get_history(history_id_2).context
                        == np.transpose(np.array([[0,0]]))).all())
        self.assertEqual(LINUCB._HistoryStorage.get_history(history_id).reward, None)
        self.assertEqual(LINUCB._HistoryStorage.get_history(history_id_2).reward, 1)'''


if __name__ == '__main__':
    unittest.main()
