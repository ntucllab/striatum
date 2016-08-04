import numpy as np
import matplotlib.pyplot as plt


def data_simulation(times, d, actions):

    """Simulate dataset for linucb and linthompsamp algorithms.

        Parameters
        ----------
        times: int
            Total number of (context, reward) tuples you want to generate.

        d: int
            Dimension of the context.

        actions : list of Action objects
            List of actions to be chosen from.

        Return
        ---------
        context: dictionary
            The dictionary stores contexts (n_actions-by-d matrix) at each iteration.

        desired_action:
            The action which will receive reward 1.
    """

    context = {}
    desired_action = np.zeros(shape=(times, 1))
    n_actions = len(actions)
    for t in range(times):
        context[t] = np.random.uniform(0, 1, (n_actions, d))
        desired_action[t] = actions[np.argmax(np.sum(context[t], axis=1))]
    return context, desired_action


def data_simulation2(times, d, actions):

    """Simulate dataset for exp4p.

        Parameters
        ----------
        times: int
            Total number of (context, reward) tuples you want to generate.

        d: int
            Dimension of the context.

        actions : list of Action objects
            List of actions to be chosen from.

        Return
        ---------
        context: array-like
            Contexts (d-by-1 array) at each iteration.

        desired_action:
            The action which will receive reward 1.
    """

    context = np.random.uniform(0, 1, (times, d))
    desired_action = np.zeros(shape=(times, 1))
    n_actions = len(actions)
    for t in range(times):
        for i in range(n_actions):
            if i * d / n_actions < sum(context[t, :]) <= (i + 1) * d / n_actions:
                desired_action[t] = actions[i]
    return context, desired_action


def policy_evaluation(policy, context, desired_action):

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

        seq_error: array
            The cumulative regret at each iteration.

    """

    times = len(desired_action)
    seq_error = np.zeros(shape=(times, 1))
    for t in range(times):
        history_id, action = policy.get_action(context[t])
        if desired_action[t][0] != action:
            policy.reward(history_id, 0)
            if t == 0:
                seq_error[t] = 1.0
            else:
                seq_error[t] = seq_error[t - 1] + 1.0
        else:
            policy.reward(history_id, 1)
            if t > 0:
                seq_error[t] = seq_error[t - 1]
    return seq_error


def regret_calculation(seq_error):

    """Evaluate a given policy based on a (context, desired_action) dataset.

        Parameters
        ----------
        seq_error: array
            The cumulative regret at each iteration.

        Return
        ---------

        regret: array
            The average cumulative regret at each iteration.

    """

    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def tuning_plot(tunning_region, ctr_tunning, label):

    """Draw the parameter tunning plot

        Parameters
        ----------
        tunning_region: array
            The region for tunning parameter.

        ctr_tunning: array
            The resulted ctrs for each number of the tunning parameter.

        label: string
            The name of label want to show.
            
    """

    plt.plot(tunning_region, ctr_tunning, 'ro-', label=label)
    plt.xlabel('parameter value')
    plt.ylabel('CTR')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Parameter Tunning Curve")
    plt.show()


def regret_plot(times, regret, label):

    """Draw the parameter tunning plot

        Parameters
        ----------
        times: int
            Total number of (context, reward) tuples you want to generate.

        regret: array
            The average cumulative regret at each iteration.

        label: string
            The name of label want to show.
    """

    plt.plot(range(times), regret, 'r-', label=label)
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Regret Bound with respect to T")
    plt.show()
