# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 18:04:26 2016

@author: David Huang
"""

import sys
sys.path.append("..")
from striatum.storage import history as history
from striatum.storage import model as model
from striatum.bandit import ucb1
import numpy as np
import unittest

class Ucb1(unittest.TestCase):
    def setUp(self):
        self.ModelStorage = model.MemoryModelStorage()
        self.HistoryStorage = history.MemoryHistoryStorage()
        self.actions = [1,2,3,4,5]
        self.alpha = 1.00

    def test_initialization(self):
        UCB = ucb1.UCB1(self.actions, self.HistoryStorage, self.ModelStorage)

    def test_get_first_action(self):
        UCB = ucb1.UCB1(self.actions, self.HistoryStorage, self.ModelStorage)
        history_id, action = UCB.get_action(context = None)
        self.assertEqual(history_id, 0)
        self.assertIn(action, self.actions)

    def test_update_reward(self):
        UCB = ucb1.UCB1(self.actions, self.HistoryStorage, self.ModelStorage)
        history_id, action = UCB.get_action(context=None)
        UCB.reward(history_id, 1)
        self.assertEqual(UCB._HistoryStorage.get_history(history_id).reward, 1)

    def test_model_storage(self):
        UCB = ucb1.UCB1(self.actions, self.HistoryStorage, self.ModelStorage)
        history_id, action = UCB.get_action(context=None)
        UCB.reward(history_id, 1)
        self.assertEqual(UCB._ModelStorage._model['empirical_reward'][action], 2)
        self.assertEqual(UCB._ModelStorage._model['n_actions'][action], 2.0)
        self.assertEqual(UCB._ModelStorage._model['n_total'], 6.0)

    def test_delay_reward(self):
        UCB = ucb1.UCB1(self.actions, self.HistoryStorage, self.ModelStorage)
        history_id, action = UCB.get_action(context=None)
        history_id_2, action_2 = UCB.get_action(context=None)
        UCB.reward(history_id, 1)
        self.assertEqual(UCB._HistoryStorage.get_history(history_id).reward, 1.0)
        self.assertEqual(UCB._HistoryStorage.get_history(history_id_2).reward, None)

    def test_reward_order_descending(self):
        UCB = ucb1.UCB1(self.actions, self.HistoryStorage, self.ModelStorage)
        history_id, action = UCB.get_action(context=None)
        history_id_2, action_2 = UCB.get_action(context=None)
        UCB.reward(history_id_2, 1)
        self.assertEqual(UCB._HistoryStorage.get_history(history_id).reward, None)
        self.assertEqual(UCB._HistoryStorage.get_history(history_id_2).reward, 1.0)


if __name__ == '__main__':
    unittest.main()
