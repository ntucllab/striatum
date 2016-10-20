# import numpy as np
# import unittest
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.multiclass import OneVsRestClassifier
# import sys
# sys.path.append("..")
# from striatum.storage import history as history
# from striatum.storage import model as model
# from striatum.bandit import exp4p
# from striatum.bandit.bandit import Action


# class Exp4P(unittest.TestCase):
#     def setUp(self):
#         self.modelstorage = model.MemoryModelStorage()
#         self.historystorage = history.MemoryHistoryStorage()
#         a1 = Action(1)
#         a2 = Action(2)
#         a3 = Action(3)
#         a4 = Action(4)
#         a5 = Action(5)
#         self.actions = [a1, a2, a3, a4, a5]
#         self.action_ids = [1, 2, 3, 4, 5]

#     def test_initialization(self):
#         policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage, delta=0.1, p_min=None)
#         self.assertEqual(self.actions, policy._actions)
#         self.assertEqual(0.1, policy.delta)

#     def test_get_first_action(self):
#         policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage, delta=0.1, p_min=None)
#         prob1 = {1: 0.82, 2: 0.03, 3: 0.05, 4: 0.04, 5: 0.06}
#         prob2 = {1: 0.72, 2: 0.07, 3: 0.07, 4: 0.07, 5: 0.07}
#         context = {1: prob1, 2: prob2}
#         history_id, action = policy.get_action(context, 1)
#         self.assertEqual(history_id, 0)
#         self.assertIn(action[0]['action'], self.actions)
#         self.assertEqual(policy._historystorage.get_unrewarded_history(history_id).context, context)

#     def test_update_reward(self):
#         policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage, delta=0.1, p_min=None)
#         prob1 = {1: 0.82, 2: 0.03, 3: 0.05, 4: 0.04, 5: 0.06}
#         prob2 = {1: 0.72, 2: 0.07, 3: 0.07, 4: 0.07, 5: 0.07}
#         context = {1: prob1, 2: prob2}
#         history_id, action = policy.get_action(context, 1)
#         policy.reward(history_id, {1: 1.0})
#         self.assertEqual(policy._historystorage.get_history(history_id).reward, {1: 1.0})

#     def test_model_storage(self):
#         policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage, delta=0.1, p_min=None)
#         prob1 = {1: 0.82, 2: 0.03, 3: 0.05, 4: 0.04, 5: 0.06}
#         prob2 = {1: 0.72, 2: 0.07, 3: 0.07, 4: 0.07, 5: 0.07}
#         context = {1: prob1, 2: prob2}
#         history_id, action = policy.get_action(context, 1)
#         policy.reward(history_id, {1: 1.0})
#         self.assertEqual(len(policy._modelstorage._model['w']), 2)
#         self.assertEqual(len(policy._modelstorage._model['action_probs']), 5)

#     def test_delay_reward(self):
#         policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage, delta=0.1, p_min=None)
#         prob1 = {1: 0.82, 2: 0.03, 3: 0.05, 4: 0.04, 5: 0.06}
#         prob2 = {1: 0.72, 2: 0.07, 3: 0.07, 4: 0.07, 5: 0.07}
#         context1 = {1: prob1, 2: prob2}
#         history_id1, action1 = policy.get_action(context1, 1)
#         prob1 = {1: 0.32, 2: 0.51, 3: 0.05, 4: 0.06, 5: 0.06}
#         prob2 = {1: 0.32, 2: 0.42, 3: 0.12, 4: 0.07, 5: 0.07}
#         context2 = {1: prob1, 2: prob2}
#         history_id2, action2 = policy.get_action(context2, 2)
#         policy.reward(history_id1, {1: 1.0})
#         self.assertEqual(policy._historystorage.get_history(history_id1).context, context1)
#         self.assertEqual(policy._historystorage.get_unrewarded_history(history_id2).context, context2)
#         self.assertEqual(policy._historystorage.get_history(history_id1).reward, {1: 1.0})
#         self.assertEqual(policy._historystorage.get_unrewarded_history(history_id2).reward, None)

#     def test_reward_order_descending(self):
#         policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage, delta=0.1, p_min=None)
#         prob1 = {1: 0.82, 2: 0.03, 3: 0.05, 4: 0.04, 5: 0.06}
#         prob2 = {1: 0.72, 2: 0.07, 3: 0.07, 4: 0.07, 5: 0.07}
#         context1 = {1: prob1, 2: prob2}
#         history_id1, action1 = policy.get_action(context1, 1)
#         prob1 = {1: 0.32, 2: 0.51, 3: 0.05, 4: 0.06, 5: 0.06}
#         prob2 = {1: 0.32, 2: 0.42, 3: 0.12, 4: 0.07, 5: 0.07}
#         context2 = {1: prob1, 2: prob2}
#         history_id2, action2 = policy.get_action(context2, 2)
#         policy.reward(history_id2, {1: 1.0, 2: 1.0})
#         self.assertEqual(policy._historystorage.get_unrewarded_history(history_id1).reward, None)
#         self.assertEqual(policy._historystorage.get_history(history_id2).reward, {1: 1.0, 2: 1.0})

#     def test_add_action(self):
#         policy = exp4p.Exp4P(self.actions, self.historystorage, self.modelstorage, delta=0.1, p_min=None)
#         prob1 = {1: 0.82, 2: 0.03, 3: 0.05, 4: 0.04, 5: 0.06}
#         prob2 = {1: 0.72, 2: 0.07, 3: 0.07, 4: 0.07, 5: 0.07}
#         context = {1: prob1, 2: prob2}
#         history_id, action = policy.get_action(context, 2)
#         a6 = Action(6)
#         a7 = Action(7)
#         policy.add_action([a6, a7])
#         policy.reward(history_id, {3: 1})
#         self.assertEqual(len(policy._actions), 7)
#         self.assertEqual(policy.action_ids, [1, 2, 3, 4, 5, 6, 7])
