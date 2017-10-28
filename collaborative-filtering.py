
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# In[2]:


def load_csv(filepath = 'ml-100k/u.data'):
    names = ['user_id', 'item_id', 'ratings', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep = '\t', names = names)
    no_of_users = df.user_id.unique().shape[0]
    no_of_items = df.item_id.unique().shape[0]
    ratings = np.zeros((no_of_users, no_of_items))
    ratings[df['user_id']-1, df['item_id']-1] = df['ratings']
    return ratings


# In[3]:


def train_test_split_slow(ratings):
    train_data = ratings.copy()
    test_data = np.zeros(ratings.shape)
    for i in range(ratings.shape[0]):
        indices = np.random.choice(ratings[i, :].nonzero()[0], size=10, replace=False)
        test_data[i, indices] = ratings[i, indices]
        train_data[i, indices] = 0
    return train_data, test_data


# In[4]:


def train_test_split_fast(ratings):
    train_data = ratings.copy();
    test_data = np.zeros(ratings.shape)
    indices = np.random.randint(0, 2, size=ratings.shape, dtype=bool)
    test_data[indices] = ratings[indices]
    train_data[indices] = 0
    return train_data, test_data


# In[5]:


def similarity(ratings, epsilon = 1e-9):
    sim = ratings.dot(ratings.T)
    norm = np.array(np.sqrt(np.diagonal(sim)))
    sim = sim / norm / norm.T
    return sim


# In[6]:


def predict(ratings, sim, kind = 'user'):
    return sim.dot(ratings) / np.array([np.abs(sim).sum(axis=1)]).T


# In[7]:


def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# In[126]:


def predict_topk(ratings, sim, k = 40):
    pred = np.zeros(ratings.shape)
    for i in range(ratings.shape[0]):
        top_k_users = [np.argsort(sim[:,i])[:-k-1:-1]]
        for j in range(ratings.shape[1]):
            pred[i, j] = sim[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
            pred[i, j] /= np.sum(np.abs(sim[i, :][top_k_users]))

    return pred


# In[128]:


def predict_topk_nobais(ratings, sim, k = 40):
    pred = np.zeros(ratings.shape)
    user_bias = np.mean(ratings, axis=1)
    ratings -= user_bias[:, np.newaxis]
    for i in range(ratings.shape[0]):
        top_k = [np.argsort(sim[:, i])[:-(k+1): -1]]
        for j in range(ratings.shape[1]):
            pred[i, j] = sim[i, top_k].dot(ratings[top_k, j].T)
            pred[i, j] /= np.sum(np.abs(sim[i, top_k]))
    pred += user_bias[:, np.newaxis]

    return pred


#%%
from matplotlib import pyplot as plt

x = np.arange(5, 20, 5)
ratings = load_csv('ml-100k/u.data')
train_set, test_set = train_test_split_fast(ratings)
sim = similarity(train_set)
y = []
for k in x:
    pred = predict_topk_nobais(train_set, sim, k)
    y += [get_mse(pred, test_set)]
print(len(x), len(y))
plt.figure()
plt.plot(x,y)