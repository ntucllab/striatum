import numpy as np
import matplotlib.pyplot as plt


def calculate_cum_reward(policy, last_history_id):
    cum_reward = {0: 0.0}
    for i in range(1, last_history_id + 1):
        cum_reward[i] = cum_reward[i - 1] + policy.historystorage.get_history(i).reward
    return cum_reward


def calculate_avg_reward(policy, last_history_id):
    cum_reward = calculate_cum_reward(policy, last_history_id)
    avg_reward = {}
    for i in range(1, last_history_id + 1):
        avg_reward[i] = cum_reward[i] / i
    return avg_reward


def plot_avg_reward(policy, last_history_id):
    avg_reward = calculate_avg_reward(policy, last_history_id)
    plt.plot(avg_reward.keys, avg_reward.values(), 'r-', label="average reward")
    plt.xlabel('time')
    plt.ylabel('avg reward')
    plt.legend()
    plt.title("Average Reward with respect to Time")
    plt.show()
