"""
Bandit interfaces
"""
from abc import abstractmethod

import numpy as np
import six

from buzzni.ai.reco.mab import rewardplot as rplt
from buzzni.ai.reco.mab.storage import Recommendation


class BaseBandit(object):
    r"""Bandit algorithm

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

    Attributes
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.
    """

    def __init__(self, model_name, history_storage, model_storage, action_storage,
                 recommendation_cls=None):
        self._model_name = model_name
        self._history_storage = history_storage
        self._model_storage = model_storage
        self._action_storage = action_storage
        if recommendation_cls is None:
            self._recommendation_cls = Recommendation
        else:
            self._recommendation_cls = recommendation_cls

    @property
    def history_storage(self):
        return self._history_storage

    @abstractmethod
    def get_action(self, context, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {action_id: context} of different actions.

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
        pass

    def _get_action_with_empty_action_storage(self, context, n_actions):
        if n_actions is None:
            recommendations = None
        else:
            recommendations = []
        history_id = self._history_storage.add_history(context,
                                                       recommendations)
        return history_id, recommendations

    @abstractmethod
    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        pass

    @abstractmethod
    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation.
        """
        pass

    def update_action(self, action):
        """Update action.

        Parameters
        ----------
        action : Action
            The Action object to update.
        """
        self._action_storage.update(action)

    @abstractmethod
    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        pass

    def calculate_cum_reward(self):
        """Calculate cumulative reward with respect to time.
        """
        return rplt.calculate_cum_reward(self)

    def calculate_avg_reward(self):
        """Calculate average reward with respect to time.
        """
        return rplt.calculate_avg_reward(self)

    def plot_avg_reward(self):
        """Plot average reward with respect to time.
        """
        rplt.plot_avg_reward(self)

    def plot_avg_regret(self):
        """Plot average regret with respect to time.
        """
        rplt.plot_avg_regret(self)


class LinearBandit(BaseBandit):
    def __init__(self, model_name, history_storage, model_storage, action_storage, context_dimension, gamma,
                 recommendation_cls=None):
        BaseBandit.__init__(self, model_name=model_name,
                            history_storage=history_storage,
                            model_storage=model_storage,
                            action_storage=action_storage,
                            recommendation_cls=recommendation_cls)

        self.context_dimension = context_dimension
        self.gamma = gamma

        # Initialize LinUCB Model Parameters
        model = {'A': {}, 'b': {}}
        for action_id in self._action_storage.iterids():
            self._init_action_model(model, action_id)

        self._model_storage.save_model(model)

    def _init_action_model(self, model, action_id):
        '''
        tf-agents 구현체 따름
        # variable 선언
            - https://github.com/tensorflow/agents/blob/3719a8780d7054984ddc528dbfe99b51b9f10062/tf_agents/bandits/agents/linear_bandit_agent.py#L81
        # 계산
            - https://github.com/tensorflow/agents/blob/3719a8780d7054984ddc528dbfe99b51b9f10062/tf_agents/bandits/policies/linear_bandit_policy.py#L251
        '''
        model['A'][action_id] = np.zeros(shape=(self.context_dimension, self.context_dimension))
        model['b'][action_id] = np.zeros((self.context_dimension, 1))

    @abstractmethod
    def _score(self, context, model):
        """Return the estimated_reward, uncertainty, score
        """
        pass

    def get_action(self, context, n_actions=None):
        """Return the action to perform

        Parameters
        ----------
        context : dict
            Contexts {action_id: context} of different actions.

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

        if not isinstance(context, dict):
            raise ValueError("LinUCB requires context dict for all actions!")
        if n_actions == -1:
            n_actions = self._action_storage.count()

        model = self._model_storage.get_model()
        estimated_reward, uncertainty, score = self._score(context, model)

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
        context = (self._history_storage
                   .get_unrewarded_history(history_id)
                   .context)

        # Update the model
        model = self._model_storage.get_model()
        A = model['A']  # pylint: disable=invalid-name
        b = model['b']

        '''
        tf-agents 구현체 따름
        # gamma:
            - https://github.com/tensorflow/agents/blob/3719a8780d7054984ddc528dbfe99b51b9f10062/tf_agents/bandits/agents/linear_bandit_agent.py#L110
        '''

        for action_id, reward in six.viewitems(rewards):
            action_context = np.reshape(context[action_id], (-1, 1))
            A[action_id] = self.gamma * A[action_id] + action_context.dot(action_context.T)
            b[action_id] = self.gamma * b[action_id] + reward * action_context

        self._model_storage.save_model({'A': A, 'b': b})

        # Update the history
        self._history_storage.add_reward(history_id, rewards)


    def add_action(self, actions):
        """ Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation
        """
        new_action_ids = self._action_storage.add(actions)
        model = self._model_storage.get_model()

        for action_id in new_action_ids:
            self._init_action_model(model, action_id)

        self._model_storage.save_model(model)

    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        model = self._model_storage.get_model()
        del model['A'][action_id]
        del model['b'][action_id]
        self._model_storage.save_model(model)
        self._action_storage.remove(action_id)