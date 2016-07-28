"""
The script uses real-world data to conduct contextual bandit experiments. Here we use
MovieLens 1M Dataset, which is released by GroupLens at 2/2003. Please fist download
the dataset from http://grouplens.org/datasets/movielens/, then unzipped the file
"ml-1m.zip" to the examples folder.
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from striatum.storage import history
from striatum.storage import model
from striatum.bandit import linucb
from striatum.bandit import linthompsamp


def movie_preprocessing(movie):
    movie_col = ["movie_id", "movie_name", "tag"]
    movie_tag = [doc.split("|") for doc in movie['tag']]
    tag_table = {token: idx for idx, token in enumerate(set(itertools.chain.from_iterable(movie_tag)))}
    movie_tag = pd.DataFrame(movie_tag)
    tag_table = pd.DataFrame(tag_table.items())
    tag_table.columns = ["Tag", "Index"]

    # use one-hot encoding for movie genres (here called tag)
    tag_dummy = np.zeros([len(movie), len(tag_table)])

    for i in range(len(movie)):
        for j in range(len(tag_table)):
            if tag_table['Tag'][j] in list(movie_tag.iloc[i, :]):
                tag_dummy[i, j] = 1

    # combine the tag_dummy one-hot encoding table to original movie files
    movie = pd.concat([movie, pd.DataFrame(tag_dummy)], 1)
    movie_col.extend(["tag" + str(i) for i in range(len(tag_table))])
    movie.columns = movie_col
    movie = movie.drop("tag", 1)
    return movie


def feature_extraction(data):
    # actions: we use top 50 movies as our actions for recommendations
    actions = data.groupby('movie_id').size().sort_values(ascending=False)[:50]
    actions = list(actions.index)

    # user_feature: tags they've watched for non-top-50 movies normalized per user
    user_feature = data[~data['movie_id'].isin(actions)]
    user_feature = user_feature.groupby('user_id').aggregate(np.sum)
    user_feature = user_feature.drop(['movie_id', 'rating', 'timestamp'], 1)
    user_feature = user_feature.div(user_feature.sum(axis=1), axis=0)

    # streaming_batch: the result for testing bandit algrorithms
    top50_data = data[data['movie_id'].isin(actions)]
    top50_data = top50_data.sort('timestamp', ascending=1)
    streaming_batch = top50_data['user_id']

    # reward_list: if rating >=3, the user will watch the movie
    top50_data['reward'] = np.where(top50_data['rating'] >= 3, 1, 0)
    reward_list = top50_data[['user_id', 'movie_id', 'reward']]
    reward_list = reward_list[reward_list['reward'] == 1]
    return streaming_batch, user_feature, actions, reward_list


def policy_evaluation(bandit, streaming_batch, user_feature, movie_context, actions, reward_list):
    times = len(streaming_batch)
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    seq_error = np.zeros(shape=(times, 1))
    context = movie_context.drop(['movie_id', 'movie_name'], axis=1)

    # TODO: implement exp4p
    if bandit == 'LinUCB':
        policy = linucb.LinUCB(actions, historystorage, modelstorage, 0.5, 40)

    if bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(actions, historystorage, modelstorage,
                                           d=40, delta=0.9, r=0.01, epsilon=0.5)
    for t in range(times):
        feature = user_feature[streaming_batch.iloc[t] == user_feature.index]

        # full_context: combine "user features" and "tag_dummy for each action"
        full_context = pd.DataFrame(np.repeat(np.array(feature), 50, axis=0))
        full_context.index = context.index
        full_context = pd.concat([context, full_context], axis=1)
        full_context = full_context.as_matrix()
        history_id, action = policy.get_action(full_context)

        watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[t]]

        if action not in list(watched_list['movie_id']):
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
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def main():
    # read and preprocess the movie data
    movie = pd.read_table("movies.dat", sep="::", names=["movie_id", "movie_name", "tag"], engine='python')
    movie = movie_preprocessing(movie)
    movie.index = movie['movie_id']

    # read the ratings data and merge it with movie data
    rating = pd.read_table("ratings.dat", sep="::",
                           names=["user_id", "movie_id", "rating", "timestamp"], engine='python')
    data = pd.merge(rating, movie, on="movie_id")

    # extract feature from our data set
    streaming_batch, user_feature, actions, reward_list = feature_extraction(data)
    streaming_batch_small = streaming_batch.iloc[1:10000]
    movie_context = movie[movie['movie_id'].isin(actions)]

    # conduct regret analyses for LinUCB and LinThompSamp
    seq_error1 = policy_evaluation('LinUCB', streaming_batch_small,
                                   user_feature, movie_context, actions, reward_list)
    seq_error2 = policy_evaluation('LinThompSamp', streaming_batch_small,
                                   user_feature, movie_context, actions, reward_list)
    regret1 = regret_calculation(seq_error1)
    regret2 = regret_calculation(seq_error2)

    # plot the result
    plt.plot(range(len(regret1)), regret1, 'r-', label="LinUCB, alpha =0.5")
    plt.plot(range(len(regret2)), regret2, 'g-', label="LinThompSamp, delta=0.9,\nr=0.01, epsilon=0.5")
    plt.xlabel('time')
    plt.ylabel('regret')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Regret Bound with respect to T")
    plt.show()


if __name__ == '__main__':
    main()
