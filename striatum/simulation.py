from six.moves import range
import numpy as np
import matplotlib.pyplot as plt


def simulate_data(times, d, actions, algorithm=None):
    """Simulate dataset for the contextual bandit problem.

    Parameters
    ----------
    times: int
        Total number of (context, reward) tuples you want to generate.

    d: int
        Dimension of the context.

    actions : list of Action objects
        List of actions to be chosen from.

    algorithm: string
        The bandit algorithm you want to use.

    Return
    ---------
    context: dictionary
        The dictionary stores contexts (dictionary with n_actions d-by-1 action) at each iteration.

    desired_action:
        The action which will receive reward 1.
    """

    action_ids = [action.action_id for action in actions]
    context = {}
    desired_action = np.empty(shape=(times,), dtype=np.int)

    if algorithm == 'Exp4P':
        for t in range(times):
            context[t] = np.random.uniform(0, 1, d)
            for i in range(len(actions)):
                if i * d / len(actions) < sum(context[t]) <= (i + 1) * d / len(actions):
                    desired_action[t] = action_ids[i]

    else:
        for t in range(times):
            context[t] = {}
            for action_id in action_ids:
                context[t][action_id] = np.random.uniform(0, 1, d)
            desired_action[t] = max(
                context[t],
                key=lambda action_id: context[t][action_id].sum())
    return context, desired_action


def evaluate_policy(policy, context, desired_action):
    """Evaluate a given policy based on a (context, desired_action) dataset.

    Parameters
    ----------
    policy: bandit object
        The bandit algorithm you want to evaluate.

    context: {array, dictionary}
        The contexts for evaluation.

    desired_action:
         The desired_action for evaluation.

    Return
    ---------
    cum_regret: array
        The cumulative regret at each iteration.
    """

    times = len(desired_action)
    cum_regret = np.zeros(shape=(times, 1))
    for t in range(times):
        history_id, action = policy.get_action(context[t], 1)
        if desired_action[t][0] != action[0]['action'].action_id:
            policy.reward(history_id, {action[0]['action'].action_id: 0})
            if t == 0:
                cum_regret[t] = 1.0
            else:
                cum_regret[t] = cum_regret[t - 1] + 1.0
        else:
            policy.reward(history_id, {action[0]['action'].action_id: 1})
            if t > 0:
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
