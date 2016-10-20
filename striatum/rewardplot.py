import six
from six.moves import range
import matplotlib.pyplot as plt


def calculate_cum_reward(policy):

    """Calculate cumulative reward with respect to time.

        Parameters
        ----------
        policy: bandit object
            The bandit algorithm you want to evaluate.

        Return
        ---------
        cum_reward: dict
            The dict stores {history_id: cumulative reward} .

        cum_n_actions: dict
            The dict stores
            {history_id: cumulative number of recommended actions}.
    """
    cum_reward = {-1: 0.0}
    cum_n_actions = {-1: 0}
    for i in range(policy.history_storage.n_histories):
        reward = policy.history_storage.get_history(i).reward
        cum_n_actions[i] = cum_n_actions[i - 1] + len(reward)
        cum_reward[i] = cum_reward[i - 1] + sum(six.viewvalues(reward))
    return cum_reward, cum_n_actions


def calculate_avg_reward(policy):

    """Calculate average reward with respect to time.

        Parameters
        ----------
        policy: bandit object
            The bandit algorithm you want to evaluate.

        Return
        ---------
        avg_reward: dict
            The dict stores {history_id: average reward} .
    """
    cum_reward, cum_n_actions = calculate_cum_reward(policy)
    avg_reward = {}
    for i in range(policy.history_storage.n_histories):
        avg_reward[i] = cum_reward[i] / cum_n_actions[i]
    return avg_reward


def plot_avg_reward(policy):

    """Plot average reward with respect to time.

        Parameters
        ----------
        policy: bandit object
            The bandit algorithm you want to evaluate.
    """

    avg_reward = calculate_avg_reward(policy)
    plt.plot(avg_reward.keys(), avg_reward.values(), 'r-',
             label="average reward")
    plt.xlabel('time')
    plt.ylabel('avg reward')
    plt.legend()
    plt.title("Average Reward with respect to Time")


def plot_avg_regret(policy):

    """Plot average regret with respect to time.

        Parameters
        ----------
        policy: bandit object
            The bandit algorithm you want to evaluate.
    """

    avg_reward = calculate_avg_reward(policy)
    points = sorted(six.viewitems(avg_reward), key=lambda x: x[0])
    x, y = zip(*points)
    plt.plot(x, [1 - reward for reward in y], 'r-', label="average regret")
    plt.xlabel('time')
    plt.ylabel('avg regret')
    plt.legend()
    plt.title("Average Regret with respect to Time")
