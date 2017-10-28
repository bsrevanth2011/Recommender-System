
# coding: utf-8

# In[326]:


import pandas as pd
import numpy as np

MAX_ITERATIONS = 500


# In[327]:


def load_csv(filepath = 'ml-100k/u.data'):
    names = ['user_id', 'item_id', 'ratings', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep = '\t', names = names)
    no_of_users = df.user_id.unique().shape[0]
    no_of_items = df.item_id.unique().shape[0]
    ratings = np.zeros((no_of_users, no_of_items))
    ratings[df['user_id']-1, df['item_id']-1] = df['ratings']
    return ratings


# In[328]:


def randomize_centroids(ratings, k = 10):
    return ratings[np.random.choice(range(ratings.shape[0]), size = k, replace = False)]


# In[329]:


def assign_clusters(ratings, clusters, centroids, k = 10):
    for user in ratings:
        index = min([(i, np.linalg.norm(user - centroids[i])) for i, _ in enumerate(centroids)], key = (lambda t: t[1]))[0]
        clusters[index].append(user)        
    for cluster in clusters:
        if not cluster:
            cluster.append(ratings[np.random.randint(0, ratings.shape[0])])
    return clusters   


# In[330]:


def has_converged(centroids, old_centroids, iterations):
    if iterations >= MAX_ITERATIONS:
        return False
    return np.array_equal(centroids, old_centroids)


# In[331]:


def kmeans(ratings, k = 10):
    centroids = randomize_centroids(ratings, k)
    old_centroids = [[] for _ in range(k)]
    iterations = 0
    clusters = []
    while not has_converged(centroids, old_centroids, iterations):
        iterations += 1
        old_centroids = centroids.copy()
        clusters = [[] for _ in range(k)]
        clusters = assign_clusters(ratings, clusters, centroids, k)
        for i in range(k):
            centroids[i] = np.mean(clusters[i], axis = 0)
    print iterations
    return clusters

