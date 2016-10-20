from six.moves import range
import numpy as np
import matplotlib.pyplot as plt

from .utils import get_random_state


def simulate_data(n_rounds, context_dimension, action_storage, algorithm=None,
                  random_state=None):
    """Simulate dataset for the contextual bandit problem.

    Parameters
    ----------
    n_rounds: int
        Total number of (context, reward) tuples you want to generate.

    context_dimension: int
        Dimension of the context.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    algorithm: string
        The bandit algorithm you want to use.

    random_state: int, np.random.RandomState (default: None)
        If int, np.random.RandomState will used it as seed. If None, a random
        seed will be used.

    Return
    ---------
    context: dict
        The dict stores contexts (dict with {action_id: context_dimension
        ndarray}) at each iteration.

    desired_actions: dict
        The action which will receive reward 1 ({history_id: action_id}).
    """
    random_state = get_random_state(random_state)

    action_ids = list(action_storage.iterids())
    context = {}
    desired_actions = {}

    if algorithm == 'Exp4P':
        for t in range(n_rounds):
            context[t] = random_state.uniform(0, 1, context_dimension)
            context_sum = context[t].sum()
            for action_i, action_id in enumerate(action_ids):
                if (action_i * context_dimension / action_storage.count()
                        < context_sum
                        <= ((action_i + 1) * context_dimension
                            / action_storage.count())):
                    desired_actions[t] = action_id

    else:
        for t in range(n_rounds):
            context[t] = {}
            for action_id in action_ids:
                context[t][action_id] = random_state.uniform(0, 1,
                                                             context_dimension)
            desired_actions[t] = max(
                context[t],
                key=lambda action_id: context[t][action_id].sum())
    return context, desired_actions


def evaluate_policy(policy, context, desired_actions):
    """Evaluate a given policy based on a (context, desired_actions) dataset.

    Parameters
    ----------
    policy: bandit object
        The bandit algorithm you want to evaluate.

    context: {array, dictionary}
        The contexts for evaluation.

    desired_actions:
         The desired_actions for evaluation.

    Return
    ---------
    cum_regret: array
        The cumulative regret at each iteration.
    """

    n_rounds = len(desired_actions)
    cum_regret = np.empty(shape=n_rounds)
    for t in range(n_rounds):
        history_id, action = policy.get_action(context[t], 1)
        action_id = action[0]['action'].id
        if desired_actions[t] != action_id:
            policy.reward(history_id, {action_id: 0})
            if t == 0:
                cum_regret[t] = 1.
            else:
                cum_regret[t] = cum_regret[t - 1] + 1.
        else:
            policy.reward(history_id, {action_id: 1})
            if t == 0:
                cum_regret[t] = 0.
            else:
                cum_regret[t] = cum_regret[t - 1]
    return cum_regret


def plot_tuning_curve(tuning_region, ctr_tuning, label):
    """Draw the parameter tuning plot

    Parameters
    ----------
    tuning_region: array
        The region for tuning parameter.

    ctr_tuning: array
        The resulted ctrs for each number of the tuning parameter.

    label: string
        The name of label want to show.
    """

    plt.plot(tuning_region, ctr_tuning, 'ro-', label=label)
    plt.xlabel('parameter value')
    plt.ylabel('CTR')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Parameter Tunning Curve")
    plt.show()
