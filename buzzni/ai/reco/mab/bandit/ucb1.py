"""Upper Confidence Bound 1
This module contains a class that implements UCB1 algorithm, a famous
multi-armed bandit algorithm without context.
"""
from __future__ import division
import numpy as np
import six

from striatum.bandit.bandit import BaseBandit


class UCB1(BaseBandit):
    r"""Upper Confidence Bound 1

    Parameters
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    recommendation_cls : class (default: None)
        The class used to initiate the recommendations. If None, then use
        default Recommendation class.

    References
    ----------
    .. [1]  Peter Auer, et al. "Finite-time Analysis of the Multiarmed Bandit
            Problem." Machine Learning, 47. 2002.
    """

    def __init__(self, history_storage, model_storage, action_storage,
                 recommendation_cls=None):
        super(UCB1, self).__init__(history_storage, model_storage,
                                   action_storage, recommendation_cls)
        total_action_reward = {}
        action_times = {}
        for action_id in self._action_storage.iterids():
            total_action_reward[action_id] = 1.0
            action_times[action_id] = 1
        n_rounds = self._action_storage.count()
        self._model_storage.save_model({
            'total_action_reward': total_action_reward,
            'action_times': action_times,
            'n_rounds': n_rounds,
        })

    def _ucb1_score(self):
        model = self._model_storage.get_model()
        total_action_reward = model['total_action_reward']
        action_times = model['action_times']
        n_rounds = model['n_rounds']

        estimated_reward_dict = {}
        uncertainty_dict = {}
        score_dict = {}
        for action_id in self._action_storage.iterids():
            estimated_reward = (total_action_reward[action_id]
                                / action_times[action_id])
            uncertainty = np.sqrt(2 * np.log(n_rounds)
                                  / action_times[action_id])
            estimated_reward_dict[action_id] = estimated_reward
            uncertainty_dict[action_id] = uncertainty
            score_dict[action_id] = estimated_reward + uncertainty
        return estimated_reward_dict, uncertainty_dict, score_dict

    def get_action(self, context=None, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context available.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        if self._action_storage.count() == 0:
            return self._get_action_with_empty_action_storage(context,
                                                              n_actions)

        estimated_reward, uncertainty, score = self._ucb1_score()
        if n_actions == -1:
            n_actions = self._action_storage.count()

        if n_actions is None:
            recommendation_id = max(score, key=score.get)
            recommendations = self._recommendation_cls(
                action=self._action_storage.get(recommendation_id),
                estimated_reward=estimated_reward[recommendation_id],
                uncertainty=uncertainty[recommendation_id],
                score=score[recommendation_id],
            )
        else:
            recommendation_ids = sorted(score, key=score.get,
                                        reverse=True)[:n_actions]
            recommendations = []  # pylint: disable=redefined-variable-type
            for action_id in recommendation_ids:
                recommendations.append(self._recommendation_cls(
                    action=self._action_storage.get(action_id),
                    estimated_reward=estimated_reward[action_id],
                    uncertainty=uncertainty[action_id],
                    score=score[action_id],
                ))

        history_id = self._history_storage.add_history(context, recommendations)
        return history_id, recommendations

    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """

        # Update the model
        model = self._model_storage.get_model()
        total_action_reward = model['total_action_reward']
        action_times = model['action_times']
        for action_id, reward in six.viewitems(rewards):
            total_action_reward[action_id] += reward
            action_times[action_id] += 1
            model['n_rounds'] += 1
        self._model_storage.save_model(model)
        # Update the history
        self._history_storage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation
        """
        self._action_storage.add(actions)

        model = self._model_storage.get_model()
        total_action_reward = model['total_action_reward']
        action_times = model['action_times']

        for action in actions:
            total_action_reward[action.id] = 1.0
            action_times[action.id] = 1
            model['n_rounds'] += 1

        self._model_storage.save_model(model)

    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        model = self._model_storage.get_model()
        model['n_rounds'] -= model['action_times'][action_id]
        del model['total_action_reward'][action_id]
        del model['action_times'][action_id]
        self._model_storage.save_model(model)
        self._action_storage.remove(action_id)
