import numpy as np
import matplotlib.pyplot as plt


def calculate_cum_reward(policy, last_history_id):
    cum_reward = {-1: 0.0}
    time = {-1: 0.0}
    for i in range(0, last_history_id + 1):
        reward = policy.historystorage.get_history(i).reward
        time[i] = time[i - 1] + len(reward.values())
        cum_reward[i] = cum_reward[i - 1] + sum(reward.values())
    return cum_reward, time


def calculate_avg_reward(policy, last_history_id):
    cum_reward, time = calculate_cum_reward(policy, last_history_id)
    avg_reward = {}
    for i in range(0, last_history_id + 1):
        avg_reward[i] = cum_reward[i] / time[i]
    return avg_reward


def plot_avg_reward(policy, last_history_id):
    avg_reward = calculate_avg_reward(policy, last_history_id)
    plt.plot(avg_reward.keys(), avg_reward.values(), 'r-', label="average reward")
    plt.xlabel('time')
    plt.ylabel('avg reward')
    plt.legend()
    plt.title("Average Reward with respect to Time")
    plt.show()


def plot_avg_regret(policy, last_history_id):
    avg_reward = calculate_avg_reward(policy, last_history_id)
    plt.plot(avg_reward.keys(), 1 - avg_reward.values(), 'r-', label="average regret")
    plt.xlabel('time')
    plt.ylabel('avg regret')
    plt.legend()
    plt.title("Average Reward with respect to Time")
    plt.show()
