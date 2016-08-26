import matplotlib.pyplot as plt


def calculate_cum_reward(policy):

    """Calculate cumulative reward with respect to time.

        Parameters
        ----------
        policy: bandit object
            The bandit algorithm you want to evaluate.

        Return
        ---------
        cum_reward: dictionary
            The dictionary stores {history_id: cumulative reward} .

        time: dictionary
            The dictionary stores {history_id: cumulative number of recommended actions} .
    """

    last_history_id = policy.historystorage.n_histories
    cum_reward = {-1: 0.0}
    time = {-1: 0.0}
    for i in range(0, last_history_id):
        reward = policy.historystorage.get_history(i).reward
        time[i] = time[i - 1] + len(reward.values())
        cum_reward[i] = cum_reward[i - 1] + sum(reward.values())
    return cum_reward, time


def calculate_avg_reward(policy):

    """Calculate average reward with respect to time.

        Parameters
        ----------
        policy: bandit object
            The bandit algorithm you want to evaluate.

        Return
        ---------
        avg_reward: dictionary
            The dictionary stores {history_id: average reward} .
    """

    last_history_id = policy.historystorage.n_histories
    cum_reward, time = calculate_cum_reward(policy)
    avg_reward = {}
    for i in range(0, last_history_id):
        avg_reward[i] = cum_reward[i] / time[i]
    return avg_reward


def plot_avg_reward(policy):

    """Plot average reward with respect to time.

        Parameters
        ----------
        policy: bandit object
            The bandit algorithm you want to evaluate.
    """

    avg_reward = calculate_avg_reward(policy)
    plt.plot(avg_reward.keys(), avg_reward.values(), 'r-', label="average reward")
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
    plt.plot(avg_reward.keys(), [1.0 - reward for reward in avg_reward.values()], 'r-', label="average regret")
    plt.xlabel('time')
    plt.ylabel('avg regret')
    plt.legend()
    plt.title("Average Regret with respect to Time")
